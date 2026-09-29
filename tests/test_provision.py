"""The variant table and the pinned runtime lockfiles that `hibiki-asr setup` installs."""

from __future__ import annotations

import importlib.util
import re
import sys
from importlib import resources
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from helpers import RTX5090, hw, rt
from hibiki_asr.diagnostics.findings import evaluate_findings
from hibiki_asr.diagnostics.selection import select_runtime
from hibiki_asr.provision.pins import lockfile_sha256, lockfile_text, normalize_name, pinned_names
from hibiki_asr.provision.variants import get_variant, load_variants, setup_command

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
INSTALLABLE = [v for v in load_variants().values() if v.installable]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pins(text: str) -> dict[str, list[Version]]:
    """name -> every pinned version (one per marker branch) in a lockfile."""
    out: dict[str, list[Version]] = {}
    for match in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", text, re.MULTILINE):
        out.setdefault(normalize_name(match.group(1)), []).append(Version(match.group(2)))
    return out


# --- the variant table ------------------------------------------------------------------------


def test_variant_rows_say_what_is_unverified() -> None:
    for variant in load_variants().values():
        assert variant.status in ("stable", "experimental")
        if variant.experimental:
            assert len(variant.note) > 40, (
                f"{variant.id}: an experimental variant must say what is unverified"
            )
        else:
            assert variant.installable and not variant.note
        if not variant.installable:
            assert variant.experimental, f"{variant.id}: no lockfile means it cannot be called stable"


def test_docker_images_match_the_table() -> None:
    variants = load_variants().values()
    tags = [v.docker_tag for v in variants if v.docker]
    assert sorted(tags) == ["cpu", "cuda", "rocm"] and len(set(tags)) == len(tags)
    assert all(v.docker_tag == "" for v in variants if not v.docker)


def test_what_is_verified_and_what_is_not() -> None:
    by_id = load_variants()
    assert {v.id for v in by_id.values() if not v.experimental} == {"cpu", "cuda12"}
    assert {v.id for v in by_id.values() if not v.installable} == {
        "cuda11",
        "rocm-linux",
        "rocm-win-gfx101x",
        "rocm-win-gfx103x",
        "rocm-win-gfx110x",
        "rocm-win-gfx120x",
    }
    assert by_id["cuda12-blackwell"].installable and by_id["cuda12-blackwell"].experimental


def test_setup_command_names_the_flag_an_experimental_install_needs() -> None:
    assert setup_command(get_variant("cuda12")) == "hibiki-asr setup --variant cuda12"
    assert (
        setup_command(get_variant("cuda12-blackwell"))
        == "hibiki-asr setup --variant cuda12-blackwell --allow-experimental"
    )
    # nothing to allow: setup explains that no runtime is pinned instead
    assert setup_command(get_variant("rocm-linux")) == "hibiki-asr setup --variant rocm-linux"


def test_hints_for_a_blackwell_card_name_the_experimental_flag() -> None:
    probe, runtime = hw(RTX5090), rt(1)
    selection = select_runtime(probe, runtime, "auto", "auto", "cuda12")
    findings = evaluate_findings(probe, runtime, selection, variant="cuda12")
    hint = next(f.hint for f in findings if f.code == "BLACKWELL_NEEDS_CU128")
    assert "hibiki-asr setup --variant cuda12-blackwell --allow-experimental" in hint


def test_hint_for_a_blackwell_card_in_a_container_says_to_build_the_image() -> None:
    probe, runtime = hw(RTX5090, container=True), rt(1)
    selection = select_runtime(probe, runtime, "auto", "auto", "cuda12")
    findings = evaluate_findings(probe, runtime, selection, variant="cuda12")
    hint = next(f.hint for f in findings if f.code == "BLACKWELL_NEEDS_CU128")
    assert "docker build -f docker/Dockerfile.cuda" in hint
    assert "HIBIKI_ASR_VARIANT=cuda12-blackwell" in hint and "--allow-experimental" in hint


# --- lockfiles --------------------------------------------------------------------------------


def test_the_lockfile_directory_holds_exactly_the_variants_lockfiles() -> None:
    shipped = {p.name for p in resources.files("hibiki_asr.provision").joinpath("lockfiles").iterdir()}
    assert shipped == {v.lockfile for v in INSTALLABLE}


@pytest.mark.parametrize("variant", INSTALLABLE, ids=lambda v: v.id)
def test_every_requirement_is_pinned_and_hashed(variant) -> None:
    text = lockfile_text(variant)
    entries = re.split(r"^(?=[A-Za-z0-9])", text, flags=re.MULTILINE)[1:]
    assert len(entries) > 15
    for entry in entries:
        name = entry.split("\\", 1)[0].strip()
        assert re.match(r"^[A-Za-z0-9._-]+==\S+", name), f"{variant.id}: {name!r} is not pinned"
        assert "--hash=sha256:" in entry, f"{variant.id}: {name} has no hash"
    assert len(lockfile_sha256(variant)) == 64


@pytest.mark.parametrize("variant", INSTALLABLE, ids=lambda v: v.id)
def test_lockfiles_have_a_source_and_are_compiled_from_it(variant) -> None:
    source = ROOT / "requirements" / f"{variant.id}.in"
    assert source.is_file()
    assert f"uv pip compile requirements/{variant.id}.in --universal" in lockfile_text(variant)


def test_a_lockfile_satisfies_the_engines_own_dependencies() -> None:
    """The pins must not fight what pyproject.toml asks for, or `setup` would break the API's environment."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = [*project["dependencies"], *project["optional-dependencies"]["runtime"]]
    for variant in INSTALLABLE:
        locked = pins(lockfile_text(variant))
        for line in wanted:
            requirement = Requirement(line)
            for version in locked.get(normalize_name(requirement.name), []):
                assert requirement.specifier.contains(version, prereleases=True), (
                    f"{variant.id}: {requirement.name}=={version} breaks '{line}'"
                )


def test_gpu_lockfiles_never_ship_the_cpu_onnxruntime_next_to_the_gpu_one() -> None:
    for variant in INSTALLABLE:
        names = pinned_names(lockfile_text(variant))
        if variant.gpu_vendor == "nvidia":
            assert "onnxruntime-gpu" in names and "onnxruntime" not in names, variant.id
        else:
            assert "onnxruntime" in names and "onnxruntime-gpu" not in names, variant.id


def test_the_inference_stack_is_the_same_across_variants() -> None:
    stacks = {
        v.id: {n: pins(lockfile_text(v))[n] for n in ("ctranslate2", "faster-whisper")} for v in INSTALLABLE
    }
    assert len({str(s) for s in stacks.values()}) == 1


def test_cuda_variants_pin_cuda12_libraries_and_split_at_blackwell() -> None:
    cuda12 = pins(lockfile_text(get_variant("cuda12")))
    blackwell = pins(lockfile_text(get_variant("cuda12-blackwell")))
    for locked in (cuda12, blackwell):
        assert Version("9") <= locked["nvidia-cudnn-cu12"][0] < Version("10")
        assert all(v >= Version("1.21") for v in locked["onnxruntime-gpu"])
        assert all(v < Version("1.27") for v in locked["onnxruntime-gpu"])  # 1.27 is built for CUDA 13
    assert cuda12["nvidia-cublas-cu12"][0] < Version("12.8") <= blackwell["nvidia-cublas-cu12"][0]


def test_python_310_gets_onnxruntime_builds_that_exist_for_it() -> None:
    # onnxruntime 1.24 declares Python >= 3.10 but publishes no cp310 wheel
    for variant in INSTALLABLE:
        name = "onnxruntime-gpu" if variant.gpu_vendor == "nvidia" else "onnxruntime"
        marked = re.search(rf"^{name}==(\S+) ; python_full_version < '3\.11'", lockfile_text(variant), re.M)
        assert marked and Version(marked.group(1)) < Version("1.24"), variant.id


# --- the script that produces them ------------------------------------------------------------


def test_compile_command_is_universal_hashed_and_keeps_the_cpu_onnxruntime_out_of_gpu_locks() -> None:
    script = _load_script("compile_lockfiles")
    cpu = script.compile_command("cpu", "cpu.txt", "onnxruntime>=1.18\n")
    gpu = script.compile_command("cuda12", "cuda12.txt", "onnxruntime-gpu[cuda]>=1.21\n")
    for command in (cpu, gpu):
        assert command[:4] == ["uv", "pip", "compile", command[3]]
        assert "--universal" in command and "--generate-hashes" in command
    assert "--no-emit-package" not in cpu
    assert gpu[gpu.index("--no-emit-package") + 1] == "onnxruntime"
    assert gpu[gpu.index("-o") + 1] == "src/hibiki_asr/provision/lockfiles/cuda12.txt"


def test_compile_script_rejects_a_variant_without_a_lockfile(capsys) -> None:
    script = _load_script("compile_lockfiles")
    assert script.main(["rocm-linux"]) == 2
    assert "rocm-linux" in capsys.readouterr().err
