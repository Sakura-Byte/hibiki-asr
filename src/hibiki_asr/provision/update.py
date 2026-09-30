"""`hibiki-asr update`: upgrade the engine, then bring the runtime back in line with its lockfile.

Which command upgrades the engine depends on how it was installed, so that is detected first (``detect_source``);
an install that cannot be updated safely (a source checkout) is refused rather than overwritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from .commands import Command, Installer
from .variants import load_variants

PACKAGE = "hibiki-asr"
REPO_URL = "https://github.com/Sakura-Byte/hibiki-asr"


class UpdateRefused(Exception):
    """The engine cannot be updated this way. The message is meant for the user."""


@dataclass(frozen=True)
class InstallSource:
    kind: str  # "uv-tool" | "git" | "index" | "editable" | "local"
    url: str | None = None
    revision: str | None = None  # the branch, tag or commit a git install was made from


@dataclass(frozen=True)
class UpdatePlan:
    upgrade: Command
    setup: Command | None  # None: no runtime variant is recorded, so there is nothing to re-apply
    notes: tuple[str, ...]


def read_direct_url() -> str | None:
    """pip and uv record where a package came from in ``direct_url.json`` (absent for an index install)."""
    try:
        return metadata.distribution(PACKAGE).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None


def detect_source(prefix: Path, direct_url: str | None) -> InstallSource:
    if (prefix / "uv-receipt.toml").is_file():  # uv writes one into every tool environment
        return InstallSource("uv-tool")
    if not direct_url:
        return InstallSource("index")
    try:
        data = json.loads(direct_url)
        url = str(data["url"])
    except (ValueError, KeyError, TypeError):
        return InstallSource("local")
    vcs = data.get("vcs_info")
    if isinstance(vcs, dict) and vcs.get("vcs") == "git":
        return InstallSource("git", url, vcs.get("requested_revision"))
    if isinstance(data.get("dir_info"), dict) and data["dir_info"].get("editable"):
        return InstallSource("editable", url)
    return InstallSource("local", url)


def upgrade_command(source: InstallSource, installer: Installer) -> Command:
    if source.kind == "uv-tool":
        if not installer.uv:
            raise UpdateRefused(
                "hibiki-asr is installed as a uv tool but `uv` is not on PATH. Add it to PATH and retry, "
                "or run `uv tool upgrade hibiki-asr` yourself."
            )
        return Command((installer.uv, "tool", "upgrade", PACKAGE))
    if source.kind == "editable":
        raise UpdateRefused(
            f"hibiki-asr is installed in editable mode from {source.url}; update that checkout with git instead."
        )
    if source.kind == "local":
        raise UpdateRefused(
            f"hibiki-asr was installed from {source.url or 'a local file'}, which `update` cannot refresh. "
            "Reinstall it from the new source."
        )
    if source.kind == "git":
        ref = f"@{source.revision}" if source.revision else ""
        return installer.upgrade(f"{PACKAGE} @ git+{source.url or REPO_URL}{ref}")
    return installer.upgrade(PACKAGE)


def plan_update(
    source: InstallSource,
    installer: Installer,
    variant_id: str | None,
    *,
    config: Path | None = None,
) -> UpdatePlan:
    upgrade = upgrade_command(source, installer)
    notes: list[str] = []
    if source.revision:
        notes.append(
            f"Installed from '{source.revision}'; a tag or commit stays where it is, a branch moves."
        )

    if variant_id is None:
        notes.append("No runtime variant is recorded here (setup was never run): run `hibiki-asr setup`.")
        return UpdatePlan(upgrade, None, tuple(notes))

    # The new version's `setup` decides what to install, so it runs as a fresh process: this one still has the
    # old code loaded.
    variant = load_variants().get(variant_id)
    argv = [installer.python, "-m", "hibiki_asr.cli", *(["--config", str(config)] if config else [])]
    argv += ["setup", "--variant", variant_id, "--yes"]
    if variant is not None and variant.experimental:
        argv.append("--allow-experimental")  # it was installed with that consent
    if source.kind == "uv-tool":
        notes.append(
            "uv rebuilds a tool's environment when it upgrades it, which removes the runtime packages `setup` "
            f"installed, so `setup` runs again for '{variant_id}'."
        )
    else:
        argv.append("--if-changed")
        notes.append(f"`setup` runs again for '{variant_id}' only if its lockfile changed.")
    return UpdatePlan(upgrade, Command(tuple(argv)), tuple(notes))
