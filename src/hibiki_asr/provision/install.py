"""`hibiki-asr setup`: pick a runtime variant and work out the commands that install it.

Choosing and planning are pure. Running the commands is the caller's job (see ``commands.run_command``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ..diagnostics.schema import GpuInfo, HardwareProbe
from .commands import Command, Installer, Runner, conflicting_distributions
from .pins import pinned_names, sha256_bytes
from .variants import Variant, get_variant, recommend_variant, setup_command


class SetupRefused(Exception):
    """The requested setup cannot or should not be done. The message is meant for the user."""


@dataclass(frozen=True)
class Choice:
    variant: Variant
    reason: str


@dataclass(frozen=True)
class SetupPlan:
    variant: Variant
    reason: str
    commands: tuple[Command, ...]
    lockfile_sha256: str


def describe_gpu(gpu: GpuInfo) -> str:
    bits = [
        f"driver {gpu.driver}" if gpu.driver else "",
        f"compute capability {gpu.compute_capability}" if gpu.compute_capability else "",
        gpu.gfx or "",
        "integrated" if gpu.integrated else "",
    ]
    return f"{gpu.name} ({', '.join(b for b in bits if b)})" if any(bits) else gpu.name


def explain_recommendation(probe: HardwareProbe, variant: Variant) -> str:
    gpus = [g for g in probe.gpus if g.vendor in ("nvidia", "amd")]
    if variant.gpu_vendor == "none":
        if not gpus:
            return "no NVIDIA or AMD GPU was detected"
        return "no GPU runtime fits " + "; ".join(describe_gpu(g) for g in gpus)
    gpu = next(g for g in gpus if g.vendor == variant.gpu_vendor)
    return f"{describe_gpu(gpu)} is served by the {variant.title} runtime"


def check_installable(variant: Variant, platform: str, allow_experimental: bool) -> None:
    """Raise SetupRefused unless ``variant`` can be installed here, and say why."""
    if not variant.supports_os(platform):
        raise SetupRefused(
            f"{variant.id} ({variant.title}) is not available on {platform}; it supports {', '.join(variant.os)}."
        )
    if not variant.installable:
        raise SetupRefused(
            f"{variant.id} ({variant.title}) cannot be installed by `hibiki-asr setup`: "
            f"it is experimental and no runtime is pinned for it yet.\n{variant.note}"
        )
    if variant.experimental and not allow_experimental:
        raise SetupRefused(
            f"{variant.id} ({variant.title}) is experimental.\n{variant.note}\n"
            "Run the same command with --allow-experimental to install it anyway."
        )


def choose_variant(
    requested: str, probe: HardwareProbe, platform: str, *, allow_experimental: bool
) -> Choice:
    """``auto`` follows the hardware probe (an experimental or unpinned choice falls back to the CPU runtime)."""
    if requested != "auto":
        try:
            variant = get_variant(requested)
        except KeyError as exc:
            raise SetupRefused(str(exc.args[0])) from None
        return Choice(variant, f"chosen with --variant {requested}")

    recommended = recommend_variant(probe, platform)
    why = explain_recommendation(probe, recommended)
    if recommended.installable and (allow_experimental or not recommended.experimental):
        return Choice(recommended, f"automatic choice: {why}")
    cpu = get_variant("cpu")
    if not recommended.installable:
        blocked = f"has no pinned runtime yet (`{setup_command(recommended)}` explains what is missing)"
    else:
        blocked = f"is experimental (`{setup_command(recommended)}` installs it anyway)"
    return Choice(cpu, f"automatic choice: {why}, but {recommended.id} {blocked}; installing the CPU runtime")


def distribution_installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def build_plan(
    choice: Choice,
    installer: Installer,
    lockfile: Path,
    *,
    installed: Callable[[str], bool] = distribution_installed,
) -> SetupPlan:
    """The commands that make this environment match ``choice.variant``'s lockfile."""
    data = lockfile.read_bytes()
    clashing = conflicting_distributions(pinned_names(data.decode("utf-8")), installed)
    commands = [installer.uninstall(clashing)] if clashing else []
    commands.append(installer.install_locked(lockfile))
    return SetupPlan(choice.variant, choice.reason, tuple(commands), sha256_bytes(data))


def execute(plan: SetupPlan, runner: Runner, say: Callable[[str], None] = print) -> int:
    """Run the plan's commands in order; stop at the first failure and return its exit code."""
    for command in plan.commands:
        say(f"+ {command.display()}")
        code = runner(command.argv)
        if code != 0:
            say(f"error: the command exited with status {code}; the runtime was not changed further")
            return code
    return 0
