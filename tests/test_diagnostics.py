"""Diagnostics: hardware probe, device selection, findings and variant recommendation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from hibiki_asr.diagnostics.findings import evaluate_findings
from hibiki_asr.diagnostics.probe import probe_hardware
from hibiki_asr.diagnostics.schema import GpuInfo, HardwareProbe, RuntimeFacts, Severity
from hibiki_asr.diagnostics.selection import normalize_device, pick_compute_type, select_runtime
from hibiki_asr.diagnostics.system import SystemAccess
from hibiki_asr.provision.variants import load_variants, recommend_variant

CPU_TYPES = ["float32", "int16", "int8", "int8_float32"]
GPU_TYPES = ["bfloat16", "float16", "float32", "int8", "int8_bfloat16", "int8_float16"]


def hw(*gpus: GpuInfo, os: str = "linux", container: bool = False, env: Mapping[str, str] | None = None,
       smi: bool | None = None, kfd: bool | None = None, avx2: bool | None = True) -> HardwareProbe:
    return HardwareProbe(
        os=os,
        arch="x86_64",
        in_container=container,
        cpu_model="Test CPU",
        cpu_cores=8,
        cpu_avx2=avx2,
        gpus=list(gpus),
        nvidia_smi_found=smi if smi is not None else any(g.vendor == "nvidia" and g.driver for g in gpus),
        kfd_present=kfd,
        env=dict(env or {}),
    )


def rt(gpu_count: int = 0, **kwargs) -> RuntimeFacts:
    types = {"cpu": CPU_TYPES}
    if gpu_count:
        types["cuda"] = GPU_TYPES
    base = dict(
        python="3.11.0",
        ctranslate2_version="4.8.2",
        faster_whisper_version="1.2.1",
        onnxruntime_version="1.30.0",
        onnxruntime_providers=["CPUExecutionProvider"],
        cuda_device_count=gpu_count,
        compute_types=types,
    )
    base.update(kwargs)
    return RuntimeFacts(**base)


RTX4090 = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4090", driver="555.42.06", vram_mb=24_564, compute_capability="8.9")
RTX5090 = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 5090", driver="575.51.03", vram_mb=32_000, compute_capability="12.0")
GTX1050 = GpuInfo(vendor="nvidia", name="NVIDIA GeForce GTX 1050", driver="536.23", vram_mb=2_048, compute_capability="6.1")
RX7900 = GpuInfo(vendor="amd", name="AMD Radeon RX 7900 XTX", gfx="gfx1100", integrated=False)
RX6800 = GpuInfo(vendor="amd", name="AMD Radeon RX 6800", gfx="gfx1030", integrated=False)
IGPU = GpuInfo(vendor="amd", name="AMD Radeon 890M", gfx="gfx1150", integrated=True)


def codes(findings) -> list[str]:
    return [f.code for f in findings]


def run(probe: HardwareProbe, runtime: RuntimeFacts, *, device: str = "auto", compute: str = "auto",
        variant: str | None = None):
    selection = select_runtime(probe, runtime, device, compute)
    return selection, evaluate_findings(probe, runtime, selection, variant=variant, requested_compute=compute)


# --- selection --------------------------------------------------------------------------------


def test_normalize_device_aliases() -> None:
    assert [normalize_device(v) for v in ("AMD", "rocm", "hip", "cuda", "cpu", None, " auto ")] == [
        "cuda", "cuda", "cuda", "cuda", "cpu", "auto", "auto"]
    with pytest.raises(ValueError):
        normalize_device("tpu")


def test_compute_type_choice() -> None:
    assert pick_compute_type("cuda", GPU_TYPES, "auto", 24_000) == "bfloat16"
    assert pick_compute_type("cuda", ["float16", "float32"], "auto", 24_000) == "float16"
    assert pick_compute_type("cuda", GPU_TYPES, "auto", 4_096) == "int8_float16"  # low VRAM
    assert pick_compute_type("cpu", CPU_TYPES, "auto", None) == "int8"
    assert pick_compute_type("cpu", CPU_TYPES, "float32", None) == "float32"  # explicit and supported
    assert pick_compute_type("cpu", CPU_TYPES, "float16", None) == "int8"  # explicit but unsupported
    assert pick_compute_type("cuda", [], "auto", None) == "float16"  # runtime reported nothing
    assert pick_compute_type("cpu", [], "auto", None) == "int8"


# --- findings: table of machines --------------------------------------------------------------


def test_cpu_only_machine_is_calm() -> None:
    selection, findings = run(hw(), rt())
    assert (selection.device, selection.degraded) == ("cpu", False)
    assert codes(findings) == ["CPU_ONLY"]
    assert findings[0].severity is Severity.info


def test_healthy_nvidia_machine_has_no_findings() -> None:
    selection, findings = run(hw(RTX4090), rt(1, onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]))
    assert (selection.device, selection.compute_type, selection.degraded, selection.vad_device) == (
        "cuda", "bfloat16", False, "cuda")
    assert findings == []


def test_nvidia_in_container_without_gpu_access_explains_how_to_fix_it() -> None:
    selection, findings = run(hw(RTX4090, container=True), rt(0), variant="cpu")
    assert selection.degraded and selection.device == "cpu"
    assert "CT2_NO_GPU_SUPPORT" in codes(findings) and codes(findings)[-1] == "DEGRADED_TO_CPU"
    hint = next(f.hint for f in findings if f.code == "CT2_NO_GPU_SUPPORT")
    assert "gpus: all" in hint and "cuda12" in hint


def test_nvidia_bare_metal_with_cpu_install_points_at_setup() -> None:
    _, findings = run(hw(RTX4090), rt(0), variant="cpu")
    hint = next(f.hint for f in findings if f.code == "CT2_NO_GPU_SUPPORT")
    assert "hibiki-asr setup --variant cuda12" in hint


@pytest.mark.parametrize("container,expected", [(True, "NVIDIA Container Toolkit"), (False, "nvidia.com/Download")])
def test_missing_nvidia_driver(container: bool, expected: str) -> None:
    card = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 3060")
    selection, findings = run(hw(card, container=container, smi=False), rt(0))
    assert selection.degraded
    missing = next(f for f in findings if f.code == "NVIDIA_DRIVER_MISSING")
    assert missing.severity is Severity.error and expected in (missing.hint or "")


def test_gpu_hidden_by_environment() -> None:
    selection, findings = run(hw(RTX4090, env={"CUDA_VISIBLE_DEVICES": "-1"}), rt(0), variant="cuda12")
    assert selection.degraded
    assert "CUDA_DISABLED_BY_ENV" in codes(findings)
    assert "CT2_NO_GPU_SUPPORT" not in codes(findings)  # the env var already explains it
    assert "Unset CUDA_VISIBLE_DEVICES" in next(f.hint for f in findings if f.code == "CUDA_DISABLED_BY_ENV")


def test_explicit_cpu_is_not_degraded_even_with_a_gpu() -> None:
    selection, findings = run(hw(RTX4090), rt(1), device="cpu")
    assert (selection.device, selection.degraded) == ("cpu", False)
    assert "DEGRADED_TO_CPU" not in codes(findings) and "CUDA_DISABLED_BY_ENV" not in codes(findings)


def test_explicit_cuda_without_a_gpu_is_degraded() -> None:
    selection, findings = run(hw(), rt(0), device="cuda")
    assert selection.degraded
    assert codes(findings) == ["DEGRADED_TO_CPU"]  # nothing more specific is known, but it is never silent


def test_driver_too_old_for_installed_runtime() -> None:
    old = GpuInfo(vendor="nvidia", name="NVIDIA GeForce GTX 1080", driver="470.57.02", vram_mb=8_192, compute_capability="6.1")
    _, findings = run(hw(old), rt(0), variant="cuda12")
    finding = next(f for f in findings if f.code == "NVIDIA_DRIVER_TOO_OLD")
    assert finding.severity is Severity.error
    assert "525" in finding.message and "--variant cuda11" in (finding.hint or "")


def test_blackwell_needs_the_cuda128_build() -> None:
    _, findings = run(hw(RTX5090), rt(1), variant="cuda12")
    assert "BLACKWELL_NEEDS_CU128" in codes(findings)
    _, findings = run(hw(RTX5090), rt(1), variant="cuda12-blackwell")
    assert "BLACKWELL_NEEDS_CU128" not in codes(findings)


def test_ctranslate2_import_failure_mentions_missing_libraries() -> None:
    broken = rt(0, ctranslate2_version=None, ctranslate2_error="OSError: libcudnn_ops.so.9: cannot open shared object file")
    _, findings = run(hw(RTX4090), broken, variant="cuda12")
    finding = next(f for f in findings if f.code == "CT2_IMPORT_FAILED")
    assert finding.severity is Severity.error
    assert "libcudnn_ops" in finding.message and "GPU runtime library is missing" in (finding.hint or "")


def test_amd_linux_without_kfd_in_docker() -> None:
    _, findings = run(hw(RX7900, container=True, kfd=False), rt(0))
    finding = next(f for f in findings if f.code == "AMD_DEVICE_NOT_VISIBLE")
    assert "--device=/dev/kfd" in (finding.hint or "") and "seccomp=unconfined" in (finding.hint or "")


def test_amd_linux_with_kfd_but_no_hip_build() -> None:
    _, findings = run(hw(RX7900, kfd=True), rt(0), variant="cpu")
    finding = next(f for f in findings if f.code == "AMD_HIP_WHEEL_MISSING")
    assert "--variant rocm-linux" in (finding.hint or "")


def test_amd_windows_recommends_the_matching_family_and_warns_about_rdna2() -> None:
    probe = hw(RX6800, os="win32")
    assert recommend_variant(probe, "win32").id == "rocm-win-gfx103x"
    _, findings = run(probe, rt(0), variant="cpu")
    assert "AMD_HIP_WHEEL_MISSING" in codes(findings)
    assert "--variant rocm-win-gfx103x" in next(f.hint for f in findings if f.code == "AMD_HIP_WHEEL_MISSING")
    assert "AMD_RDNA2_RUNTIME_CRASH" in codes(findings)


def test_amd_integrated_gpu_is_not_treated_as_a_missing_gpu() -> None:
    selection, findings = run(hw(IGPU, os="win32"), rt(0))
    assert not selection.degraded  # an iGPU is never "expected" under 'auto'
    assert codes(findings) == ["AMD_IGPU_UNSUPPORTED", "CPU_ONLY"]
    assert all(f.severity is Severity.info for f in findings)  # useful to know, nothing to fix

    # ... but asking for a GPU explicitly makes it a warning, and the degrade is reported.
    selection, findings = run(hw(IGPU, os="win32"), rt(0), device="cuda")
    assert selection.degraded
    assert next(f.severity for f in findings if f.code == "AMD_IGPU_UNSUPPORTED") is Severity.warning


def test_low_vram_and_vad_on_cpu_are_informational() -> None:
    selection, findings = run(hw(GTX1050), rt(1), variant="cuda12")
    assert selection.compute_type == "int8_float16" and selection.vad_device == "cpu"
    assert {"VRAM_LOW", "VAD_ON_CPU"} <= set(codes(findings))
    assert all(f.severity is Severity.info for f in findings)


def test_unsupported_explicit_compute_type_is_reported() -> None:
    selection, findings = run(hw(), rt(0), compute="float16")
    assert selection.compute_type == "int8"
    assert "COMPUTE_TYPE_DOWNGRADED" in codes(findings)


def test_failed_runtime_probe_is_an_error() -> None:
    failed = RuntimeFacts(probe_ok=False, probe_error="runtime probe exited with code -11")
    _, findings = run(hw(), failed)
    assert findings[0].code == "RUNTIME_PROBE_FAILED" and findings[0].severity is Severity.error
    assert "-11" in findings[0].message


def test_cpu_without_avx2_warns() -> None:
    _, findings = run(hw(avx2=False), rt())
    assert "CPU_NO_AVX2" in codes(findings)


def test_findings_are_sorted_most_severe_first() -> None:
    _, findings = run(hw(RTX4090, container=True, smi=False), rt(0), variant="cpu")
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s.value])


# --- variants ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gpu,platform,expected",
    [
        (RTX4090, "linux", "cuda12"),
        (RTX5090, "win32", "cuda12-blackwell"),
        (GPU_OLD := GpuInfo(vendor="nvidia", name="old", driver="470.1", compute_capability="6.1"), "linux", "cuda11"),
        (GpuInfo(vendor="nvidia", name="ancient", driver="390.1", compute_capability="6.1"), "linux", "cpu"),
        (RX7900, "linux", "rocm-linux"),
        (RX7900, "win32", "rocm-win-gfx110x"),
        (IGPU, "win32", "cpu"),
        (GpuInfo(vendor="amd", name="AMD Radeon Mystery"), "win32", "cpu"),
        (RTX4090, "darwin", "cpu"),
    ],
)
def test_variant_recommendation(gpu: GpuInfo, platform: str, expected: str) -> None:
    assert recommend_variant(hw(gpu, os=platform), platform).id == expected


def test_no_gpu_recommends_cpu() -> None:
    assert recommend_variant(hw(), "linux").id == "cpu"


def test_every_variant_row_is_wellformed() -> None:
    variants = load_variants()
    assert {"cpu", "cuda11", "cuda12", "cuda12-blackwell", "rocm-linux"} <= set(variants)
    for variant in variants.values():
        assert variant.lockfile.endswith(".txt")
        if variant.gpu_vendor == "amd":
            assert variant.gfx
        if variant.gpu_vendor == "nvidia":
            assert variant.min_driver and variant.min_cc


# --- probe with a fake system -----------------------------------------------------------------


def fake_system(*, platform: str = "linux", commands: Mapping[str, tuple[int, str]] | None = None,
                files: Mapping[str, str] | None = None, env: Mapping[str, str] | None = None) -> SystemAccess:
    commands = commands or {}
    files = files or {}

    def run(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        return commands.get(argv[0], (127, ""))

    def listdir(path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return sorted({f[len(prefix):].split("/")[0] for f in files if f.startswith(prefix)})

    return SystemAccess(
        platform=platform, machine="x86_64", cpu_count=lambda: 16, env=dict(env or {}), run=run,
        read_text=lambda p: files.get(p), exists=lambda p: p in files, listdir=listdir, which=lambda name: None,
    )


def test_probe_reads_nvidia_smi_and_container_markers() -> None:
    smi = "NVIDIA GeForce RTX 4090, 555.42.06, 24564, 8.9\n"
    probe = probe_hardware(fake_system(
        commands={"nvidia-smi": (0, smi)},
        files={"/.dockerenv": "", "/dev/kfd": "", "/proc/cpuinfo": "model name\t: Test CPU\nflags\t: fpu avx avx2\n"},
        env={"CUDA_VISIBLE_DEVICES": "0", "HOME": "/root"},
    ))
    assert probe.in_container and probe.nvidia_smi_found and probe.cpu_avx2 is True and probe.cpu_model == "Test CPU"
    assert probe.env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert probe.gpus[0] == GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4090", driver="555.42.06",
                                    vram_mb=24_564, compute_capability="8.9")


def test_probe_retries_nvidia_smi_without_compute_cap_on_old_drivers() -> None:
    calls: list[str] = []

    def run(argv, timeout):
        if argv[0] != "nvidia-smi":
            return 127, ""
        query = next(a for a in argv if a.startswith("--query-gpu="))
        calls.append(query)
        if "compute_cap" in query:
            return 2, ""
        return 0, "Tesla K80, 470.1, 11441\n"

    system = fake_system()
    system.run = run
    probe = probe_hardware(system)
    assert len(calls) == 2 and probe.gpus[0].compute_capability is None and probe.gpus[0].vram_mb == 11_441


def test_probe_finds_nvidia_card_when_driver_is_missing() -> None:
    probe = probe_hardware(fake_system(commands={"lspci": (0, "01:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060]\n")}))
    assert not probe.nvidia_smi_found
    assert [(g.vendor, g.driver) for g in probe.gpus] == [("nvidia", None)]


def test_probe_amd_from_kernel_topology_without_rocm_tools() -> None:
    files = {
        "/dev/kfd": "",
        "/sys/class/kfd/kfd/topology/nodes/0/properties": "cpu_cores_count 16\ngfx_target_version 0\n",
        "/sys/class/kfd/kfd/topology/nodes/1/properties": "simd_count 96\ngfx_target_version 110000\n",
    }
    probe = probe_hardware(fake_system(files=files, commands={"lspci": (0, "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]\n")}))
    assert probe.kfd_present is True
    assert [(g.vendor, g.gfx) for g in probe.gpus] == [("amd", "gfx1100")]
    assert "7900 XTX" in probe.gpus[0].name


def test_probe_amd_via_rocm_agent_enumerator() -> None:
    probe = probe_hardware(fake_system(commands={"rocm_agent_enumerator": (0, "gfx000\ngfx1030\n")}))
    assert [g.gfx for g in probe.gpus] == ["gfx1030"]


def test_probe_windows_maps_adapter_names() -> None:
    names = "AMD Radeon RX 7800 XT\nIntel(R) UHD Graphics\n"
    probe = probe_hardware(fake_system(platform="win32", commands={"powershell": (0, names)}))
    assert [(g.vendor, g.gfx) for g in probe.gpus] == [("amd", "gfx1101"), ("intel", None)]
    assert probe.kfd_present is None


def test_probe_windows_falls_back_to_wmic_and_skips_duplicate_nvidia_names() -> None:
    smi = "NVIDIA GeForce RTX 4090, 555.42.06, 24564, 8.9\n"
    wmic = "Name\nNVIDIA GeForce RTX 4090\nAMD Radeon(TM) Graphics\n"
    probe = probe_hardware(fake_system(platform="win32", commands={"nvidia-smi": (0, smi), "wmic": (0, wmic)}))
    assert [g.vendor for g in probe.gpus] == ["nvidia", "amd"]
    assert probe.gpus[1].integrated is True


def test_probe_with_no_tools_reports_an_empty_machine() -> None:
    probe = probe_hardware(fake_system())
    assert probe.gpus == [] and probe.nvidia_smi_found is False and probe.in_container is False
