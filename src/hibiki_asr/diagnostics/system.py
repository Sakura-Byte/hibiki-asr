"""Thin access to the operating system, so hardware probing can be tested with fakes."""

from __future__ import annotations

import os
import platform as _platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


def _run(argv: Sequence[str], timeout: float = 10.0) -> tuple[int, str]:
    """Run a command; (127, "") when it does not exist, (124, "") on timeout."""
    try:
        completed = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError:
        return 126, ""
    return completed.returncode, completed.stdout


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


@dataclass
class SystemAccess:
    platform: str = field(default_factory=lambda: sys.platform)
    machine: str = field(default_factory=_platform.machine)
    cpu_count: Callable[[], int | None] = os.cpu_count
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    run: Callable[[Sequence[str], float], tuple[int, str]] = _run
    read_text: Callable[[str], str | None] = _read_text
    exists: Callable[[str], bool] = os.path.exists
    listdir: Callable[[str], list[str]] = lambda path: sorted(os.listdir(path)) if os.path.isdir(path) else []
    which: Callable[[str], str | None] = shutil.which

    def is_windows(self) -> bool:
        return self.platform.startswith("win")

    def is_linux(self) -> bool:
        return self.platform.startswith("linux")
