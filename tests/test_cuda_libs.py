"""Finding the CUDA libraries that pip installs under site-packages/nvidia."""

from __future__ import annotations

import os
from pathlib import Path

from hibiki_asr.provision import cuda_libs
from hibiki_asr.provision.cuda_libs import nvidia_library_dirs, prepare_environment, prepend_library_path


def make_site(root: Path, layout: dict[str, str]) -> Path:
    """``{"cublas": "lib", "cudnn": "bin"}`` -> site-packages/nvidia/cublas/lib, .../nvidia/cudnn/bin."""
    for package, subdir in layout.items():
        (root / "nvidia" / package / subdir).mkdir(parents=True)
    return root


def test_linux_wheels_keep_their_libraries_in_lib(tmp_path: Path) -> None:
    site = make_site(tmp_path, {"cudnn": "lib", "cublas": "lib", "cuda_runtime": "lib"})
    (site / "nvidia" / "cublas" / "include").mkdir()
    found = nvidia_library_dirs([site], "linux")
    assert found == [
        site / "nvidia" / p / "lib" for p in ("cublas", "cuda_runtime", "cudnn")
    ]  # sorted, lib only


def test_windows_wheels_keep_their_dlls_in_bin(tmp_path: Path) -> None:
    site = make_site(tmp_path, {"cublas": "bin", "cudnn": "bin"})
    (site / "nvidia" / "cublas" / "lib").mkdir()
    assert nvidia_library_dirs([site], "win32") == [
        site / "nvidia" / "cublas" / "bin",
        site / "nvidia" / "cudnn" / "bin",
    ]


def test_no_nvidia_packages_means_nothing_to_add(tmp_path: Path) -> None:
    (tmp_path / "numpy").mkdir()
    assert nvidia_library_dirs([tmp_path, tmp_path / "missing"], "linux") == []
    env: dict[str, str] = {"LD_LIBRARY_PATH": "/usr/lib"}
    assert prepend_library_path(env, [], "linux") is False and env == {"LD_LIBRARY_PATH": "/usr/lib"}


def test_a_library_found_in_two_site_directories_is_listed_once(tmp_path: Path) -> None:
    a = make_site(tmp_path / "a", {"cublas": "lib"})
    assert nvidia_library_dirs([a, a], "linux") == [a / "nvidia" / "cublas" / "lib"]


def test_directories_go_in_front_of_what_is_already_on_the_path() -> None:
    env = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}
    dirs = [Path("/venv/nvidia/cublas/lib"), Path("/venv/nvidia/cudnn/lib")]
    assert prepend_library_path(env, dirs, "linux") is True
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [*map(str, dirs), "/usr/local/cuda/lib64"]


def test_the_variable_is_created_when_it_is_missing_and_windows_uses_path() -> None:
    env: dict[str, str] = {}
    assert prepend_library_path(env, [Path("/x/lib")], "linux") is True
    assert env == {"LD_LIBRARY_PATH": str(Path("/x/lib"))}

    env = {"PATH": "C:\\Windows"}
    assert prepend_library_path(env, [Path("/x/bin")], "win32") is True
    assert env["PATH"] == os.pathsep.join([str(Path("/x/bin")), "C:\\Windows"])
    assert "LD_LIBRARY_PATH" not in env


def test_preparing_twice_changes_nothing_the_second_time(tmp_path: Path, monkeypatch) -> None:
    site = make_site(tmp_path, {"cublas": "lib"})
    monkeypatch.setattr(cuda_libs, "_site_dirs", lambda: [site])
    env: dict[str, str] = {}
    assert prepare_environment(env, "linux") is True
    assert prepare_environment(env, "linux") is False
    assert env["LD_LIBRARY_PATH"] == str(site / "nvidia" / "cublas" / "lib")
