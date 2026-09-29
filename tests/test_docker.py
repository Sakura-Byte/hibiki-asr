"""The Docker files and the compose example, checked statically (no Docker daemon is needed or used)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from hibiki_asr.diagnostics.findings import IMAGE
from hibiki_asr.provision.variants import get_variant, load_variants
from hibiki_asr.settings import DEFAULT_PORT

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
# image tag suffix -> (Dockerfile, the variant it installs by default)
IMAGES = {
    "cpu": ("Dockerfile.cpu", "cpu"),
    "cuda": ("Dockerfile.cuda", "cuda12"),
    "rocm": ("Dockerfile.rocm", "rocm-linux"),
}


def dockerfile(tag: str) -> str:
    return (DOCKER / IMAGES[tag][0]).read_text(encoding="utf-8")


def instructions(text: str) -> str:
    """The Dockerfile without comments, with continued lines joined."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"\\\n\s*", " ", "\n".join(lines))


def test_there_is_a_dockerfile_for_every_published_image() -> None:
    published = {v.docker_tag for v in load_variants().values() if v.docker}
    assert published == set(IMAGES)
    assert {p.name for p in DOCKER.glob("Dockerfile.*")} == {name for name, _ in IMAGES.values()}
    for tag, (_, variant_id) in IMAGES.items():
        assert get_variant(variant_id).docker_tag == tag


@pytest.mark.parametrize("tag", ["cpu", "cuda"])
def test_images_run_the_same_setup_as_a_bare_metal_install(tag: str) -> None:
    text = instructions(dockerfile(tag))
    variant = get_variant(IMAGES[tag][1])
    assert f"ARG HIBIKI_ASR_VARIANT={variant.id}" in text
    assert variant.installable and not variant.experimental  # a default that `setup` accepts without a flag
    assert 'hibiki-asr setup --variant "${HIBIKI_ASR_VARIANT}" --yes ${HIBIKI_ASR_SETUP_ARGS}' in text
    assert 'HIBIKI_ASR_VARIANT="${HIBIKI_ASR_VARIANT}"' in text  # what the engine reads at run time
    assert ".[runtime]" in text  # the API process needs numpy to start; setup pins the rest


def test_the_cuda_image_is_for_an_nvidia_variant_and_the_cpu_image_is_not() -> None:
    assert get_variant(IMAGES["cuda"][1]).gpu_vendor == "nvidia"
    assert get_variant(IMAGES["cpu"][1]).gpu_vendor == "none"
    cuda = instructions(dockerfile("cuda"))
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in cuda and "NVIDIA_VISIBLE_DEVICES=all" in cuda
    assert "NVIDIA_" not in instructions(dockerfile("cpu"))


def test_the_cuda_image_needs_no_cuda_base_image() -> None:
    assert re.search(r"^FROM (\$\{PYTHON_IMAGE\}|python:)", dockerfile("cuda"), re.MULTILINE)
    assert "nvidia/cuda" not in instructions(dockerfile("cuda"))  # it is explained in a comment, not used


@pytest.mark.parametrize("tag", list(IMAGES))
def test_every_image_is_a_non_root_engine_with_a_data_volume_and_a_health_check(tag: str) -> None:
    text = instructions(dockerfile(tag))
    assert "USER 10001:10001" in text
    assert "HIBIKI_ASR_DATA_DIR=/data" in text and "VOLUME /data" in text
    assert "HIBIKI_ASR_HOST=0.0.0.0" in text
    assert f"EXPOSE {DEFAULT_PORT}" in text
    assert "/v1/healthz" in text and "HEALTHCHECK" in text
    assert 'ENTRYPOINT ["hibiki-asr"]' in text and 'CMD ["serve"]' in text
    assert "chown 10001:10001 /data" in text  # a named volume inherits this ownership


def test_the_health_check_calls_an_endpoint_the_api_has() -> None:
    # that it needs no token is covered by test_api.py::test_healthz_needs_no_token
    spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
    assert "get" in spec["paths"]["/v1/healthz"]


def test_the_rocm_image_says_it_is_experimental_and_unverified_and_what_it_needs() -> None:
    header = dockerfile("rocm").split("ARG ROCM_IMAGE", 1)[0]
    assert "EXPERIMENTAL" in header and "NOT VERIFIED" in header
    assert "CT2_ROCM_WHEEL" in header and "CT2_ROCM_SHA256" in header
    text = instructions(dockerfile("rocm"))
    assert "sha256sum -c" in text  # the wheel is checked before it is installed
    assert "hibiki-asr setup --variant cpu --yes" in text  # the verified path; only ctranslate2 is replaced
    assert "HIBIKI_ASR_VARIANT=rocm-linux" in text
    assert not get_variant("rocm-linux").installable  # which is why the image cannot use `setup` for it


def test_dockerignore_keeps_what_the_dockerfiles_copy() -> None:
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for tag in IMAGES:
        for line in instructions(dockerfile(tag)).splitlines():
            if line.startswith("COPY ") and "--from" not in line:
                *sources, _target = line.split()[1:]
                for source in sources:
                    assert (ROOT / source).exists(), f"{IMAGES[tag][0]} copies {source}, which does not exist"
                    assert source.rstrip("/") not in ignored, f"{source} is in .dockerignore"


def test_the_hint_for_building_a_blackwell_image_names_arguments_the_dockerfile_has() -> None:
    text = instructions(dockerfile("cuda"))
    assert "ARG HIBIKI_ASR_VARIANT=" in text and "ARG HIBIKI_ASR_SETUP_ARGS=" in text


# --- compose ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((DOCKER / "compose.example.yml").read_text(encoding="utf-8"))


def test_compose_has_one_service_per_profile_on_the_published_images(compose: dict) -> None:
    services = compose["services"]
    assert {name: s["profiles"] for name, s in services.items()} == {
        "hibiki-asr-cpu": ["cpu"],
        "hibiki-asr-cuda": ["cuda"],
        "hibiki-asr-rocm": ["rocm"],
    }
    for tag in IMAGES:
        assert services[f"hibiki-asr-{tag}"]["image"] == f"{IMAGE}:latest-{tag}"


def test_compose_requires_a_token_publishes_8001_and_keeps_data_in_a_named_volume(compose: dict) -> None:
    for service in compose["services"].values():
        assert service["environment"]["HIBIKI_ASR_TOKEN"].startswith("${HIBIKI_ASR_TOKEN:?")  # fails if unset
        assert service["ports"] == [f"{DEFAULT_PORT}:{DEFAULT_PORT}"]
        assert "hibiki-asr-data:/data" in service["volumes"]
    assert "hibiki-asr-data" in compose["volumes"]


def test_compose_passes_the_gpu_in_the_way_the_findings_tell_users_to(compose: dict) -> None:
    cuda = compose["services"]["hibiki-asr-cuda"]
    (device,) = cuda["deploy"]["resources"]["reservations"]["devices"]
    assert device["driver"] == "nvidia" and device["count"] == "all" and device["capabilities"] == ["gpu"]

    rocm = compose["services"]["hibiki-asr-rocm"]
    assert {"/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"} <= set(rocm["devices"])
    assert set(rocm["group_add"]) == {"video", "render"}
    assert rocm["security_opt"] == ["seccomp=unconfined"]

    cpu = compose["services"]["hibiki-asr-cpu"]
    assert not {"deploy", "devices", "gpus", "group_add"} & set(cpu)


def test_compose_says_the_rocm_profile_is_experimental() -> None:
    text = (DOCKER / "compose.example.yml").read_text(encoding="utf-8")
    section = text.split("# --- AMD ROCm", 1)[1].split("hibiki-asr-rocm:", 1)[0]
    assert "EXPERIMENTAL" in section and "not verified" in section
