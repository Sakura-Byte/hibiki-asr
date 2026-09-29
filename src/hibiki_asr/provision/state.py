"""Which runtime variant `hibiki-asr setup` installed (or the image was built with), and from which lockfile."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

ENV_VARIANT = "HIBIKI_ASR_VARIANT"  # set by the Docker images, which have no `setup` step at run time


def variant_file(data_dir: Path) -> Path:
    return data_dir / "variant.json"


def _read_state(data_dir: Path) -> dict[str, Any]:
    try:
        state = json.loads(variant_file(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def variant_from_env() -> str | None:
    return os.environ.get(ENV_VARIANT) or None


def read_variant(data_dir: Path) -> str | None:
    if variant_from_env():
        return variant_from_env()
    value = _read_state(data_dir).get("variant")
    return value if isinstance(value, str) else None


def read_lockfile_sha256(data_dir: Path) -> str | None:
    """The hash of the lockfile the runtime was installed from, or None when unknown."""
    value = _read_state(data_dir).get("lockfile_sha256")
    return value if isinstance(value, str) else None


def write_variant(data_dir: Path, variant: str, lockfile_sha256: str | None = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"variant": variant}
    if lockfile_sha256:
        state["lockfile_sha256"] = lockfile_sha256
    fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=".variant-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    os.replace(tmp, variant_file(data_dir))  # atomic: a crash never leaves a truncated state file
