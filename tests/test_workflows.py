"""The GitHub Actions workflows: they are valid, and they build what the variant table says to publish.

The workflows cannot run here, so these tests pin the parts that must stay in step with the rest of the repository.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from hibiki_asr.diagnostics.findings import IMAGE
from hibiki_asr.provision.variants import get_variant, load_variants

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
NAMES = ["ci", "release"]
# docker tag -> the variant that image installs (see docker/Dockerfile.*)
IMAGE_VARIANTS = {"cpu": "cpu", "cuda": "cuda12", "rocm": "rocm-linux"}


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))


def steps(job: dict) -> list[dict]:
    return job["steps"]


def commands(job: dict) -> str:
    return "\n".join(s["run"] for s in steps(job) if "run" in s)


@pytest.mark.parametrize("name", NAMES)
def test_every_step_runs_something_and_every_action_is_pinned_to_a_major_version(name: str) -> None:
    workflow = load(name)
    assert workflow["permissions"]  # never the broad default token
    for job_name, job in workflow["jobs"].items():
        for step in steps(job):
            assert ("uses" in step) != ("run" in step), f"{name}/{job_name}: {step}"
            if "uses" in step:
                assert re.fullmatch(r"[\w./-]+@v\d+(\.\d+)*", step["uses"]), step["uses"]


def test_ci_runs_lint_types_and_tests_on_linux_and_windows_for_the_oldest_and_a_current_python() -> None:
    job = load("ci")["jobs"]["test"]
    matrix = job["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "windows-latest"]
    assert matrix["python"] == ["3.10", "3.12"]
    assert job["strategy"]["fail-fast"] is False  # one failing platform must not hide the other
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"].startswith(">=3.10")  # the oldest supported Python is tested
    text = commands(job)
    assert '".[runtime,dev]"' in text
    for tool in ("ruff check .", "mypy", "pytest"):
        assert re.search(rf"^uv run --no-sync {re.escape(tool)}", text, re.MULTILINE), tool
    mypy = next(s for s in steps(job) if s.get("name") == "mypy")
    assert mypy["if"] == "matrix.python == '3.12'"  # 3.10 gets older numpy stubs, see the comment in ci.yml


def test_ci_fails_when_openapi_json_is_stale() -> None:
    text = commands(load("ci")["jobs"]["openapi"])
    assert text.index("scripts/export_openapi.py") < text.index("git diff --exit-code -- openapi.json")
    assert (ROOT / "scripts" / "export_openapi.py").is_file()


def test_ci_builds_the_images_it_can_and_checks_that_they_start() -> None:
    job = load("ci")["jobs"]["docker"]
    tags = {entry["tag"]: entry["dockerfile"] for entry in job["strategy"]["matrix"]["include"]}
    assert tags == {"cpu": "docker/Dockerfile.cpu", "cuda": "docker/Dockerfile.cuda"}  # rocm cannot be built
    text = commands(job)
    assert "HIBIKI_ASR_TOKEN=ci" in text and "/v1/healthz" in text and "hibiki-asr doctor" in text
    for dockerfile in tags.values():
        assert (ROOT / dockerfile).is_file()


def test_ci_checks_the_install_script_and_the_compose_file() -> None:
    text = commands(load("ci")["jobs"]["scripts"])
    assert "shellcheck install.sh" in text
    assert "for profile in cpu cuda rocm" in text and "docker/compose.example.yml" in text


# --- release ------------------------------------------------------------------------------------------


def test_release_runs_on_version_tags_only() -> None:
    workflow = load("release")
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as True
    assert triggers == {"push": {"tags": ["v*"]}}


def test_release_attaches_the_sdist_and_wheel_to_a_github_release() -> None:
    jobs = load("release")["jobs"]
    assert "uv build" in commands(jobs["dist"])
    publish = jobs["github-release"]
    assert publish["needs"] == "dist"
    (release,) = [s for s in steps(publish) if s.get("uses", "").startswith("softprops/action-gh-release")]
    assert release["with"]["files"] == "dist/*"


def test_release_builds_exactly_the_images_the_variant_table_publishes() -> None:
    job = load("release")["jobs"]["docker"]
    entries = {e["tag"]: e for e in job["strategy"]["matrix"]["include"]}
    assert set(entries) == {v.docker_tag for v in load_variants().values() if v.docker}
    for tag, entry in entries.items():
        variant = get_variant(IMAGE_VARIANTS[tag])
        assert variant.docker_tag == tag
        assert entry["experimental"] is variant.experimental  # the experimental image may fail the release
        assert (ROOT / entry["dockerfile"]).is_file()
    assert job["continue-on-error"] == "${{ matrix.experimental }}"


def test_release_pushes_versioned_and_latest_tags_of_the_image_the_findings_name() -> None:
    workflow = load("release")
    assert (
        workflow["env"]["IMAGE"] == IMAGE == "ghcr.io/sakura-byte/hibiki-asr"
    )  # lower case, as registries need
    meta = next(s for s in steps(workflow["jobs"]["docker"]) if s.get("id") == "meta")
    assert meta["with"]["images"] == "${{ env.IMAGE }}"
    assert "type=semver,pattern={{version}},suffix=-${{ matrix.tag }}" in meta["with"]["tags"]
    assert "type=raw,value=latest-${{ matrix.tag }}" in meta["with"]["tags"]
    assert "!contains(github.ref_name, '-')" in meta["with"]["tags"]  # a pre-release never becomes latest
    assert "packages" in workflow["permissions"] and workflow["permissions"]["packages"] == "write"


def test_the_rocm_image_is_built_only_when_a_wheel_was_configured_and_gets_the_arguments_it_needs() -> None:
    job = load("release")["jobs"]["docker"]
    rocm = next(e for e in job["strategy"]["matrix"]["include"] if e["tag"] == "rocm")
    dockerfile = (ROOT / rocm["dockerfile"]).read_text(encoding="utf-8")
    passed = dict(line.split("=", 1) for line in rocm["build_args"].splitlines() if line)
    assert set(passed) == {"CT2_ROCM_WHEEL", "CT2_ROCM_SHA256", "ROCM_IMAGE"}
    for name in passed:
        assert f"ARG {name}" in dockerfile
    build = next(s for s in steps(job) if s.get("name") == "Build and push")
    assert build["if"] == "${{ !matrix.experimental || vars.CT2_ROCM_WHEEL != '' }}"
    assert build["with"]["build-args"] == "${{ matrix.build_args }}"
    default_image = re.search(r"ARG ROCM_IMAGE=(\S+)", dockerfile)
    assert (
        default_image and default_image.group(1) in passed["ROCM_IMAGE"]
    )  # the same fallback in both places
