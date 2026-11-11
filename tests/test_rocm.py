"""Pinned ROCm archives: platform selection, integrity checks and replacement of the baseline wheel."""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from pathlib import Path

import pytest

from hibiki_asr.provision.commands import Installer
from hibiki_asr.provision.install import Choice, build_plan
from hibiki_asr.provision.pins import lockfile_sha256, runtime_sha256
from hibiki_asr.provision.rocm import extract_verified_wheel, select_wheel, wheel_install_command
from hibiki_asr.provision.variants import get_variant, load_variants


@pytest.mark.parametrize("python_tag", ["cp310", "cp311", "cp312", "cp313"])
@pytest.mark.parametrize(
    "os_name,platform_tag",
    [
        ("linux", "manylinux_2_27_x86_64.manylinux_2_28_x86_64"),
        ("win32", "win_amd64"),
    ],
)
def test_matching_wheel_is_selected(python_tag, os_name, platform_tag) -> None:
    names = [
        f"nested/ctranslate2-4.9.0-{tag}-{tag}-{platform_tag}.whl"
        for tag in ("cp310", "cp311", "cp312", "cp313")
    ]
    chosen = select_wheel(names, "4.9.0", python_tag, os_name, "x86_64")
    assert f"-{python_tag}-{python_tag}-" in chosen
    with pytest.raises(ValueError, match="found 0"):
        select_wheel(names, "4.9.0", "cp39", os_name, "x86_64")
    with pytest.raises(ValueError, match="found 2"):
        select_wheel([*names, chosen], "4.9.0", python_tag, os_name, "x86_64")


@pytest.mark.parametrize("os_name,machine", [("darwin", "x86_64"), ("linux", "arm64"), ("win32", "ARM64")])
def test_unsupported_platform_is_refused(os_name, machine) -> None:
    with pytest.raises(ValueError, match="x86_64"):
        select_wheel([], "4.9.0", "cp312", os_name, machine)


def test_hash_is_checked_before_extracting_and_only_the_selected_wheel_is_written(tmp_path: Path) -> None:
    archive, output = tmp_path / "bundle.zip", tmp_path / "output"
    output.mkdir()
    wheel = "ctranslate2-4.9.0-cp312-cp312-win_amd64.whl"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"../../{wheel}", b"selected wheel")
        bundle.writestr("../../unrelated", b"must not be extracted")
    with pytest.raises(ValueError, match="SHA256"):
        extract_verified_wheel(archive, output, "0" * 64, "4.9.0", "cp312", "win32", "AMD64")
    assert list(output.iterdir()) == []
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    selected = extract_verified_wheel(archive, output, digest, "4.9.0", "cp312", "win32", "AMD64")
    assert selected == output / wheel and selected.read_bytes() == b"selected wheel"
    assert list(output.iterdir()) == [selected] and not (tmp_path.parent / "unrelated").exists()


def test_rocm_replaces_the_same_version_baseline_without_resolving_dependencies() -> None:
    wheel = Path("/tmp/ctranslate2.whl")
    uv = wheel_install_command(Installer("python", "uv"), wheel).argv
    pip = wheel_install_command(Installer("python"), wheel).argv
    assert "--reinstall" in uv and "--no-deps" in uv
    assert "--force-reinstall" in pip and "--no-deps" in pip
    assert uv[-1] == pip[-1] == str(wheel)


def test_setup_installs_the_baseline_then_the_archive_and_updates_track_both(tmp_path: Path) -> None:
    variant = get_variant("rocm-linux")
    lockfile = tmp_path / "cpu.txt"
    lockfile.write_text("ctranslate2==4.9.0\n")
    plan = build_plan(
        Choice(variant, "explicit"), Installer("python", "uv"), lockfile, installed=lambda _: False
    )
    assert len(plan.commands) == 2
    assert plan.commands[0].argv[-1] == str(lockfile)
    assert plan.commands[1].argv == ("python", "-m", "hibiki_asr.provision.rocm", "rocm-linux", "--uv", "uv")
    updated = dataclasses.replace(variant, wheel_archive_sha256="0" * 64)
    assert plan.lockfile_sha256 != runtime_sha256(updated, lockfile.read_bytes())
    assert lockfile_sha256(variant) != lockfile_sha256(get_variant("cpu"))


def test_every_pinned_rocm_variant_has_a_complete_release_reference() -> None:
    for variant in load_variants().values():
        if variant.wheel_archive_url:
            assert variant.gpu_vendor == "amd" and variant.experimental
            assert f"/releases/download/v{variant.wheel_version}/" in variant.wheel_archive_url
            assert len(variant.wheel_archive_sha256) == 64
            assert variant.lockfile == "cpu.txt"
