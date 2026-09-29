"""The `hibiki-asr` command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import API_VERSION
from .diagnostics.report import format_report
from .engine import Engine, engine_version
from .models.catalog import CatalogError, Ref
from .models.manager import ModelsError
from .models.schema import DownloadRequest, DownloadState
from .provision.commands import InstallerMissing, Runner, detect_installer, run_command
from .provision.cuda_libs import prepare_environment
from .provision.install import (
    SetupRefused,
    build_plan,
    check_installable,
    choose_variant,
    distribution_installed,
    execute,
)
from .provision.pins import lockfile_path, lockfile_sha256
from .provision.service import (
    ACTIONS,
    ServiceUnsupported,
    apply_plan,
    describe,
    engine_argv,
    plan_service,
    restart_hint,
    windows_user,
)
from .provision.state import read_lockfile_sha256, read_variant, variant_from_env, write_variant
from .provision.update import UpdateRefused, detect_source, plan_update, read_direct_url
from .settings import Settings, config_file_path, load_settings, write_config_value

_SECRETS = ("token", "hf_token")

# Commands that change the environment go through this, so tests can swap it and never install anything.
command_runner: Runner = run_command


def _mib(n: float) -> str:
    return f"{n / 2**20:.0f} MB"


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# -- serve -----------------------------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .api.app import create_app

    try:
        settings.validate_for_serving()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    engine = Engine(settings)
    where = f"http://{'127.0.0.1' if settings.host in ('0.0.0.0', '::') else settings.host}:{settings.port}"
    print(f"hibiki-asr {engine_version()} listening on {settings.host}:{settings.port}", file=sys.stderr)
    print(
        f"  connect Hibiki to: {where}"
        + (f"   token: {settings.token}" if settings.token else "   (no token needed on loopback)"),
        file=sys.stderr,
    )
    print(f"  models: {settings.resolved_models_dir}", file=sys.stderr)
    uvicorn.run(
        create_app(engine), host=settings.host, port=settings.port, log_level=settings.log_level.lower()
    )
    return 0


# -- doctor ----------------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    engine = Engine(settings)
    try:
        diagnostics = engine.diagnostics(refresh=True)
    finally:
        engine.shutdown()
    if args.json:
        _print_json(diagnostics.model_dump(mode="json"))
    else:
        print(format_report(diagnostics))
    return 1 if any(f.severity.value == "error" for f in diagnostics.findings) else 0


# -- setup -----------------------------------------------------------------------------------------------


def _platform() -> str:
    return sys.platform


def cmd_setup(args: argparse.Namespace, settings: Settings) -> int:
    engine = Engine(settings)
    try:
        return _setup(args, settings, engine)
    finally:
        engine.shutdown()


def _setup(args: argparse.Namespace, settings: Settings, engine: Engine) -> int:
    try:
        choice = choose_variant(
            args.variant, engine.hardware(), _platform(), allow_experimental=args.allow_experimental
        )
        variant = choice.variant
        check_installable(variant, _platform(), args.allow_experimental)
        installer = detect_installer(sys.executable)
    except (SetupRefused, InstallerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if (
        args.if_changed
        and read_variant(settings.data_dir) == variant.id
        and read_lockfile_sha256(settings.data_dir) == lockfile_sha256(variant)
    ):
        print(f"The {variant.id} runtime is already installed from the current lockfile; nothing to do.")
        return 0

    with lockfile_path(variant) as lockfile:
        plan = build_plan(choice, installer, lockfile, installed=distribution_installed)
        print(f"Runtime variant: {variant.id} ({variant.title})")
        print(f"  why: {plan.reason}")
        if variant.experimental:
            print(f"  experimental: {variant.note}")
        print("Commands:")
        for command in plan.commands:
            print(f"  {command.display()}")
        if args.dry_run:
            print("Dry run: nothing was changed.")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    "error: not running in a terminal, so cannot ask; pass --yes to install", file=sys.stderr
                )
                return 2
            if input("Install now? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Cancelled; nothing was changed.")
                return 1
        code = execute(plan, command_runner)
    if code != 0:
        return code

    write_variant(settings.data_dir, variant.id, plan.lockfile_sha256)
    prepare_environment()  # the CUDA libraries the install just added
    print("\nInstalled. Checking what the engine sees now:\n")
    diagnostics = engine.diagnostics(refresh=True)
    print(format_report(diagnostics))
    return 1 if any(f.severity.value == "error" for f in diagnostics.findings) else 0


# -- update ----------------------------------------------------------------------------------------------


def cmd_update(args: argparse.Namespace, settings: Settings) -> int:
    if variant_from_env():
        print(
            "error: HIBIKI_ASR_VARIANT is set, as it is in the Docker images. A container is updated by "
            "pulling a newer image, not from inside.",
            file=sys.stderr,
        )
        return 2
    try:
        installer = detect_installer(sys.executable)
        source = detect_source(Path(sys.prefix), read_direct_url())
        plan = plan_update(
            source,
            installer,
            read_variant(settings.data_dir),
            config=args.config.resolve() if args.config else None,
        )
    except (UpdateRefused, InstallerMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    commands = [plan.upgrade, *([plan.setup] if plan.setup else [])]
    print("Commands:")
    for command in commands:
        print(f"  {command.display()}")
    for note in plan.notes:
        print(note)
    if args.dry_run:
        print("Dry run: nothing was changed.")
        return 0

    for command in commands:
        print(f"+ {command.display()}")
        code = command_runner(command.argv)
        if code != 0:
            print(f"error: the command exited with status {code}", file=sys.stderr)
            if command is plan.setup:
                print(
                    "The engine itself was upgraded. Fix the problem above and run "
                    "`hibiki-asr setup --variant <id> --yes` again (see `hibiki-asr doctor`).",
                    file=sys.stderr,
                )
            return code
    print(restart_hint(_platform()))
    return 0


# -- service ---------------------------------------------------------------------------------------------


def cmd_service(args: argparse.Namespace, settings: Settings) -> int:
    config = args.config.resolve() if args.config else None
    try:
        plan = plan_service(
            args.action,
            _platform(),
            engine_argv(shutil.which, sys.executable, config),
            config_home=Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"),
            data_dir=settings.data_dir,
            user=windows_user(os.environ),
        )
    except ServiceUnsupported as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("\n".join(describe(plan)))
        print("Dry run: nothing was changed.")
        return 0
    code = apply_plan(plan, command_runner)
    if code == 0 and args.action != "status":
        print("\n".join(plan.notes))
    return code


# -- config ----------------------------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    path = config_file_path()
    if args.action == "path":
        print(path)
    elif args.action == "show":
        data = settings.model_dump(mode="json")
        data["models_dir"] = str(settings.resolved_models_dir)
        for key in _SECRETS:
            if data.get(key):
                data[key] = "********"
        print(f"# config file: {path}{'' if path.exists() else ' (not created yet)'}", file=sys.stderr)
        _print_json(data)
    else:
        if args.key is None or args.value is None:
            print(
                "usage: hibiki-asr config set KEY VALUE   (e.g. device cpu, vad.threshold 0.4, hf_endpoint https://hf-mirror.com)",
                file=sys.stderr,
            )
            return 2
        try:
            write_config_value(path, args.key, args.value)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{args.key} = {args.value}   (saved to {path}; restart the engine to apply)")
    return 0


# -- models ----------------------------------------------------------------------------------------------


def _ref(text: str) -> Ref:
    if "@" not in text:
        raise CatalogError(f"give a version too, as ID@VERSION (see `hibiki-asr models list`), got {text!r}")
    return Ref.parse(text)


class _Bar:
    """A one-line progress display for the terminal; plain lines when output is piped."""

    def __init__(self) -> None:
        self.tty = sys.stderr.isatty()
        self._last_line = ""

    def show(
        self,
        state: str,
        stage: str,
        source: str | None,
        done: int,
        total: int | None,
        rate: float,
        file: str | None,
    ) -> None:
        pct = f"{100 * done / total:5.1f}%" if total else "  ... "
        speed = f"{_mib(rate)}/s" if rate else ""
        line = (
            f"{stage:<11} {pct}  {_mib(done)}"
            + (f" / {_mib(total)}" if total else "")
            + f"  {speed:<10} {file or ''}  [{source or '-'}]"
        )
        if self.tty:
            print("\r" + line[:118].ljust(118), end="", file=sys.stderr, flush=True)
        elif line != self._last_line and (not total or int(100 * done / total) % 10 == 0):
            print(line, file=sys.stderr)
        self._last_line = line

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr)


def cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    engine = Engine(settings)
    manager = engine.models
    try:
        if args.action == "list":
            models = manager.list_models()
            if args.json:
                _print_json([m.model_dump(mode="json") for m in models])
                return 0
            for m in models:
                update = "  (update available)" if m.update_available else ""
                print(
                    f"{m.id}  {m.display_name}  [{m.task}: {'/'.join(m.source_languages)} -> {'/'.join(m.output_languages)}]{update}"
                )
                for v in m.versions:
                    size = f"{_mib(v.size_bytes)}" if v.size_bytes else "?"
                    print(
                        f"    {v.version:<8} {v.status.value:<14} {size:>9}{'  <- active' if v.active else ''}"
                    )
        elif args.action == "sources":
            report = manager.sources(refresh=True, extra=args.endpoint or [])
            if args.json:
                _print_json(report.model_dump(mode="json"))
                return 0
            print(f"Hugging Face directly reachable: {'yes' if report.huggingface_reachable else 'NO'}")
            for s in report.sources:
                outcome = f"ok, {s.latency_ms} ms" if s.reachable else f"unreachable ({s.error})"
                print(f"  {s.endpoint:<32} {s.kind.value:<10} {outcome}")
            print(f"recommended: {report.recommended_endpoint or 'none reachable'}")
        elif args.action == "download":
            request = DownloadRequest(
                endpoint=args.endpoint, fallback=not args.no_fallback, threads=args.threads
            )
            status = manager.start_download(_ref(args.ref), request)
            bar = _Bar()
            import time as _time

            try:
                while status.state in (DownloadState.queued, DownloadState.running):
                    bar.show(
                        status.state.value,
                        status.stage,
                        status.source,
                        status.bytes_done,
                        status.bytes_total,
                        status.bytes_per_second,
                        status.file,
                    )
                    _time.sleep(0.25)
                    status = manager.get_download(status.id)
            except KeyboardInterrupt:
                manager.cancel_download(status.id)
                bar.finish()
                print("cancelled; run the same command to resume", file=sys.stderr)
                return 130
            bar.finish()
            if status.state is not DownloadState.succeeded:
                print(f"error: {status.error or status.state.value}", file=sys.stderr)
                return 1
            print(f"installed {args.ref}")
        elif args.action == "verify":
            result = manager.verify(_ref(args.ref))
            print("ok" if result.ok else "PROBLEMS:\n  " + "\n  ".join(result.problems))
            return 0 if result.ok else 1
        elif args.action == "use":
            info = manager.set_active(args.ref.partition("@")[0], args.ref.partition("@")[2] or "")
            print(f"{info.id}: active version is now {info.active_version}")
        elif args.action == "delete":
            manager.delete(_ref(args.ref))
            print(f"deleted {args.ref}")
        elif args.action == "refresh":
            refreshed = manager.refresh_catalog()
            print(refreshed.message)
            return 0 if refreshed.refreshed else 1
    except (ModelsError, CatalogError) as exc:
        print(f"error: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 1
    finally:
        engine.shutdown()
    return 0


# -- parser ----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hibiki-asr", description="Local Whisper speech recognition engine."
    )
    parser.add_argument(
        "--version", action="version", version=f"hibiki-asr {engine_version()} (api {API_VERSION})"
    )
    parser.add_argument("--config", type=Path, help="config file (default: see `hibiki-asr config path`)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    doctor = sub.add_parser(
        "doctor", help="show the hardware, the device in use, and how to fix a CPU fallback"
    )
    doctor.add_argument("--json", action="store_true")

    setup = sub.add_parser(
        "setup",
        help="install the runtime that fits this machine (CPU, NVIDIA CUDA) into this Python environment",
    )
    setup.add_argument(
        "--variant",
        default="auto",
        metavar="auto|ID",
        help="runtime variant to install; auto follows the hardware and says why (default). IDs: see variants.toml",
    )
    setup.add_argument("--dry-run", action="store_true", help="print the exact commands and change nothing")
    setup.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    setup.add_argument(
        "--allow-experimental",
        action="store_true",
        help="install a variant that is marked experimental (what is unverified is printed first)",
    )
    setup.add_argument(
        "--if-changed",
        action="store_true",
        help="do nothing when this runtime was already installed from the current lockfile (used by `update`)",
    )

    service = sub.add_parser(
        "service",
        help="start the engine automatically (systemd user unit on Linux, Task Scheduler on Windows)",
    )
    service.add_argument("action", choices=ACTIONS)
    service.add_argument("--dry-run", action="store_true", help="print the file and commands, change nothing")

    update = sub.add_parser(
        "update", help="upgrade the engine, and its runtime when the pinned versions changed"
    )
    update.add_argument("--dry-run", action="store_true", help="print the commands and change nothing")

    config = sub.add_parser("config", help="show or change settings")
    config.add_argument("action", choices=["show", "path", "set"])
    config.add_argument("key", nargs="?")
    config.add_argument("value", nargs="?")

    models = sub.add_parser("models", help="install and manage models")
    ms = models.add_subparsers(dest="action", required=True)
    lst = ms.add_parser("list")
    lst.add_argument("--json", action="store_true")
    src = ms.add_parser("sources", help="check whether Hugging Face or a mirror can be reached")
    src.add_argument("--endpoint", action="append", help="also test this endpoint (repeatable)")
    src.add_argument("--json", action="store_true")
    dl = ms.add_parser("download", help="download ID@VERSION with the components it needs")
    dl.add_argument("ref", metavar="ID@VERSION")
    dl.add_argument("--endpoint", help="download from this endpoint first, e.g. https://hf-mirror.com")
    dl.add_argument("--no-fallback", action="store_true", help="use only --endpoint")
    dl.add_argument("--threads", type=int, help="parallel connections per large file")
    for name, helptext in (
        ("verify", "re-hash an installed version"),
        ("delete", "remove an installed version"),
    ):
        ms.add_parser(name, help=helptext).add_argument("ref", metavar="ID@VERSION")
    ms.add_parser("use", help="make ID@VERSION the default version").add_argument("ref", metavar="ID@VERSION")
    ms.add_parser("refresh", help="fetch a newer model catalog")
    return parser


_COMMANDS = {
    "serve": cmd_serve,
    "doctor": cmd_doctor,
    "setup": cmd_setup,
    "service": cmd_service,
    "update": cmd_update,
    "config": cmd_config,
    "models": cmd_models,
}
_STARTS_RUNTIME = {"serve", "doctor", "setup"}  # they probe or run the inference stack in a child process


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overrides = {k: getattr(args, k, None) for k in ("host", "port")}
        settings = load_settings(overrides=overrides, config_path=args.config)
    except ValueError as exc:
        print(f"error: invalid settings: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=getattr(logging, settings.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore"):  # one line per HTTP request would drown a multi-gigabyte download
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))
    if args.command in _STARTS_RUNTIME:
        prepare_environment()  # pip-installed CUDA libraries must be on the library path before children start
    return _COMMANDS[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
