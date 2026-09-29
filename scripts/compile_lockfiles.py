#!/usr/bin/env python3
"""Regenerate the pinned runtime lockfiles from requirements/<variant>.in (needs `uv` and network access).

    python scripts/compile_lockfiles.py             # every variant that has a lockfile
    python scripts/compile_lockfiles.py cpu cuda12  # only these

The lockfiles are universal (one file for Linux, Windows, macOS and Python 3.10 to 3.13) and carry sha256 hashes,
so `hibiki-asr setup` installs exactly what was resolved here and nothing else. Commit the result.
A variant without a lockfile in variants.toml has no verified pins and is skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hibiki_asr.provision.variants import load_variants

ROOT = Path(__file__).resolve().parents[1]
LOCKFILES = ROOT / "src" / "hibiki_asr" / "provision" / "lockfiles"


def compile_command(variant_id: str, lockfile: str, sources: str) -> list[str]:
    command = [
        "uv",
        "pip",
        "compile",
        f"requirements/{variant_id}.in",
        "--universal",
        "--python-version",
        "3.10",  # the oldest supported Python; newer ones get their own versions through markers
        "--generate-hashes",
        "-o",
        f"src/hibiki_asr/provision/lockfiles/{lockfile}",
    ]
    if "onnxruntime-gpu" in sources:
        # faster-whisper depends on the CPU onnxruntime, which must not be installed next to onnxruntime-gpu.
        command += ["--no-emit-package", "onnxruntime"]
    return command


def main(argv: list[str]) -> int:
    variants = [v for v in load_variants().values() if v.installable and (not argv or v.id in argv)]
    unknown = set(argv) - {v.id for v in variants}
    if unknown:
        print(f"no lockfile is defined for: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    for variant in variants:
        source = ROOT / "requirements" / f"{variant.id}.in"
        command = compile_command(variant.id, variant.lockfile, source.read_text(encoding="utf-8"))
        print("+", " ".join(command), file=sys.stderr)
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
