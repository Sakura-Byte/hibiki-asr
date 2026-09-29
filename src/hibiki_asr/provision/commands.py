"""Commands that change the Python environment, built as data so they can be printed, tested and only then run.

Nothing here runs a command. ``run_command`` is the single place that does, and callers take it as a parameter
so tests never install anything.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class InstallerMissing(RuntimeError):
    """Neither uv nor pip can be found."""


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]

    def display(self) -> str:
        return subprocess.list2cmdline(self.argv) if os.name == "nt" else shlex.join(self.argv)


Runner = Callable[[Sequence[str]], int]


def run_command(argv: Sequence[str]) -> int:
    """Run a command with the terminal attached and return its exit code (127 when it does not exist)."""
    try:
        return subprocess.run(list(argv), check=False).returncode
    except OSError as exc:
        print(f"error: cannot run {argv[0]}: {exc}")
        return 127


# Distributions that install the same import package. Two of them side by side corrupt each other, and removing
# one afterwards deletes files the other needs, so the ones a lockfile does not ask for are removed first.
CONFLICT_GROUPS: tuple[tuple[str, ...], ...] = (("onnxruntime", "onnxruntime-gpu"),)


def conflicting_distributions(pinned: set[str], installed: Callable[[str], bool]) -> list[str]:
    """Installed distributions that would clash with the ones ``pinned`` installs."""
    out: list[str] = []
    for group in CONFLICT_GROUPS:
        if any(name in pinned for name in group):
            out += [name for name in group if name not in pinned and installed(name)]
    return out


@dataclass(frozen=True)
class Installer:
    """uv when it is on PATH, otherwise ``python -m pip``; always aimed at ``python``'s own environment."""

    python: str
    uv: str | None = None

    def install_locked(self, lockfile: Path) -> Command:
        # --no-deps: a lockfile is a complete resolution. Without it the installer would add back the
        # dependencies the lockfile leaves out on purpose (the CPU onnxruntime next to onnxruntime-gpu).
        if self.uv:
            return Command(
                (self.uv, "pip", "install", "--python", self.python, "--no-deps", "-r", str(lockfile))
            )
        return Command((self.python, "-m", "pip", "install", "--no-deps", "-r", str(lockfile)))

    def uninstall(self, names: Sequence[str]) -> Command:
        if self.uv:
            return Command((self.uv, "pip", "uninstall", "--python", self.python, *names))
        return Command((self.python, "-m", "pip", "uninstall", "-y", *names))

    def upgrade(self, requirement: str) -> Command:
        if self.uv:
            return Command((self.uv, "pip", "install", "--python", self.python, "--upgrade", requirement))
        return Command((self.python, "-m", "pip", "install", "--upgrade", requirement))


def has_pip() -> bool:
    return importlib.util.find_spec("pip") is not None


def detect_installer(
    python: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    pip_available: Callable[[], bool] = has_pip,
) -> Installer:
    uv = which("uv")
    if uv or pip_available():
        return Installer(python, uv)
    raise InstallerMissing(
        "neither uv nor pip is available in this environment. Install uv (https://docs.astral.sh/uv/) and retry."
    )
