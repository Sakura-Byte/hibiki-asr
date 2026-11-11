#!/usr/bin/env python3
"""Regenerate the pinned runtime lockfiles from requirements/<variant>.in (needs `uv` and network access).

    python scripts/compile_lockfiles.py             # every variant that has a lockfile
    python scripts/compile_lockfiles.py cpu cuda12  # only these
    python scripts/compile_lockfiles.py --upgrade  # update every pin to the newest compatible release

The lockfiles are universal (one file for Linux, Windows, macOS and Python 3.10 to 3.13) and carry sha256 hashes,
so `hibiki-asr setup` installs exactly what was resolved here and nothing else. Commit the result.
A variant without a lockfile in variants.toml has no verified pins and is skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hibiki_asr.provision.variants import load_variants

ROOT = Path(__file__).resolve().parents[1]
LOCKFILES = ROOT / "src" / "hibiki_asr" / "provision" / "lockfiles"


def compile_command(variant_id: str, lockfile: str, sources: str, *, upgrade: bool = False) -> list[str]:
    command = [
        "uv",
        "pip",
        "compile",
        f"requirements/{variant_id}.in",
        "--index-url",
        "https://pypi.org/simple",
        "--universal",
        "--python-version",
        "3.10",  # the oldest supported Python; newer ones get their own versions through markers
        "--generate-hashes",
        "-o",
        f"src/hibiki_asr/provision/lockfiles/{lockfile}",
    ]
    if upgrade:
        command += ["--upgrade", "--refresh"]
    if "onnxruntime-gpu" in sources:
        # faster-whisper depends on the CPU onnxruntime, which must not be installed next to onnxruntime-gpu.
        command += ["--no-emit-package", "onnxruntime"]
    if "mlx-whisper" in sources:
        # mlx-whisper needs torch only for its checkpoint converter, which the engine does not use.
        command += ["--no-emit-package", "torch"]
        # The stack it shares with the CPU runtime is the CPU runtime's, version for version: MLX only adds to it.
        command += ["--constraints", "src/hibiki_asr/provision/lockfiles/cpu.txt"]
    return command


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upgrade", action="store_true", help="Update pins to the newest compatible releases."
    )
    parser.add_argument("variants", nargs="*", help="Variant IDs (default: all installable variants).")
    args = parser.parse_args(argv)
    variants = [
        v
        for v in load_variants().values()
        if v.installable and not v.wheel_archive_url and (not args.variants or v.id in args.variants)
    ]
    unknown = set(args.variants) - {v.id for v in variants}
    if unknown:
        print(f"no lockfile is defined for: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    for variant in variants:
        source = ROOT / "requirements" / f"{variant.id}.in"
        command = compile_command(
            variant.id, variant.lockfile, source.read_text(encoding="utf-8"), upgrade=args.upgrade
        )
        print("+", " ".join(command), file=sys.stderr)
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
