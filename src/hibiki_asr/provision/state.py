"""Which runtime variant `hibiki-asr setup` installed (or the image was built with)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VARIANT = "HIBIKI_ASR_VARIANT"  # set by the Docker images, which have no `setup` step at run time


def variant_file(data_dir: Path) -> Path:
    return data_dir / "variant.json"


def read_variant(data_dir: Path) -> str | None:
    if os.environ.get(ENV_VARIANT):
        return os.environ[ENV_VARIANT]
    try:
        value = json.loads(variant_file(data_dir).read_text(encoding="utf-8")).get("variant")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def write_variant(data_dir: Path, variant: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    variant_file(data_dir).write_text(json.dumps({"variant": variant}), encoding="utf-8")
