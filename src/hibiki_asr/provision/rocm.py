"""Install the matching wheel from a pinned, hash-checked official CTranslate2 ROCm release archive.

`setup` installs the CPU baseline first, then invokes this module in the target Python environment. ROCm runtime
libraries and the host GPU driver must already be installed. No commands or network requests run during planning.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .commands import Command, Installer, run_command
from .variants import get_variant


def select_wheel(names: list[str], version: str, python_tag: str, os_name: str, machine: str) -> str:
    if machine.lower() not in ("amd64", "x86_64") or os_name not in ("linux", "win32"):
        raise ValueError("The pinned ROCm wheels require Linux or Windows on x86_64.")
    prefix = f"ctranslate2-{version}-{python_tag}-{python_tag}-"
    candidates = [
        name
        for name in names
        if PurePosixPath(name).name.startswith(prefix)
        and (
            name.endswith("-win_amd64.whl")
            if os_name == "win32"
            else "manylinux" in PurePosixPath(name).name and name.endswith("_x86_64.whl")
        )
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one ROCm wheel for {python_tag} on {os_name}; found {len(candidates)}.")
    return candidates[0]


def extract_verified_wheel(
    archive: Path,
    destination: Path,
    sha256: str,
    version: str,
    python_tag: str,
    os_name: str,
    machine: str,
) -> Path:
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError("The CTranslate2 ROCm archive failed its SHA256 check; nothing was installed.")
    with zipfile.ZipFile(archive) as bundle:
        member = select_wheel(bundle.namelist(), version, python_tag, os_name, machine)
        target = destination / PurePosixPath(member).name
        # Copy only the selected file to a basename we chose. Never extract archive paths onto the filesystem.
        with bundle.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    return target


def wheel_install_command(installer: Installer, wheel: Path) -> Command:
    # Reinstall even when the version is unchanged: the baseline installed the PyPI CPU/CUDA wheel of that version.
    if installer.uv:
        return Command(
            (
                installer.uv,
                "pip",
                "install",
                "--python",
                installer.python,
                "--no-deps",
                "--reinstall",
                str(wheel),
            )
        )
    return Command((installer.python, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--uv")
    args = parser.parse_args(argv)
    variant = get_variant(args.variant)
    if not variant.wheel_archive_url or not variant.wheel_archive_sha256 or not variant.wheel_version:
        parser.error(f"{variant.id} has no pinned ROCm archive")
    if not variant.supports_os(sys.platform) or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error(f"{variant.id} requires its supported OS and an x86_64 Python interpreter")
    try:
        with tempfile.TemporaryDirectory(prefix="hibiki-asr-rocm-") as directory:
            root = Path(directory)
            archive = root / "wheels.zip"
            print(f"Downloading the pinned ROCm wheels: {variant.wheel_archive_url}", flush=True)
            with (
                urllib.request.urlopen(variant.wheel_archive_url, timeout=60) as response,
                archive.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            wheel = extract_verified_wheel(
                archive,
                root,
                variant.wheel_archive_sha256,
                variant.wheel_version,
                tag,
                sys.platform,
                platform.machine(),
            )
            command = wheel_install_command(Installer(sys.executable, args.uv), wheel)
            print(f"+ {command.display()}", flush=True)
            return run_command(command.argv)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
