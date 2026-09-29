"""Make the CUDA libraries that pip installs (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, ...) findable.

Those wheels put their shared libraries under ``site-packages/nvidia/<package>/lib`` (``bin`` on Windows), which
is on no library search path, so CTranslate2 and onnxruntime cannot load them. The engine puts the directories
on ``LD_LIBRARY_PATH`` (``PATH`` on Windows) for itself and for the probe and worker processes it starts. That
must happen before those processes start: the loader reads the variable once at process start-up.
"""

from __future__ import annotations

import os
import site
import sys
import sysconfig
from collections.abc import Iterable, MutableMapping
from pathlib import Path


def library_variable(platform: str) -> str:
    return "PATH" if platform.startswith("win") else "LD_LIBRARY_PATH"


def nvidia_library_dirs(site_dirs: Iterable[Path], platform: str) -> list[Path]:
    """Every ``nvidia/<package>/lib`` (``bin`` on Windows) directory found in ``site_dirs``, without duplicates."""
    subdir = "bin" if platform.startswith("win") else "lib"
    found: list[Path] = []
    for site_dir in site_dirs:
        root = site_dir / "nvidia"
        if not root.is_dir():
            continue
        for package in sorted(root.iterdir()):
            candidate = package / subdir
            if candidate.is_dir() and candidate not in found:
                found.append(candidate)
    return found


def prepend_library_path(env: MutableMapping[str, str], dirs: Iterable[Path], platform: str) -> bool:
    """Put ``dirs`` in front of the library search path in ``env``. Returns whether anything changed."""
    variable = library_variable(platform)
    current = env.get(variable, "")
    present = current.split(os.pathsep) if current else []
    missing = [str(d) for d in dirs if str(d) not in present]
    if not missing:
        return False
    env[variable] = os.pathsep.join([*missing, *present])
    return True


def _site_dirs() -> list[Path]:
    paths = {
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
        *getattr(site, "getsitepackages", list)(),
    }
    return [Path(p) for p in sorted(paths)]


def prepare_environment(env: MutableMapping[str, str] | None = None, platform: str | None = None) -> bool:
    """Call before starting anything that imports ctranslate2 or onnxruntime. Idempotent."""
    platform = platform or sys.platform
    return prepend_library_path(
        os.environ if env is None else env, nvidia_library_dirs(_site_dirs(), platform), platform
    )
