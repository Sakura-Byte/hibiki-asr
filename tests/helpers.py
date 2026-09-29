"""Builders for hardware and runtime facts, shared by the diagnostics and worker tests."""

from __future__ import annotations

from collections.abc import Mapping

from hibiki_asr import cli
from hibiki_asr.diagnostics.schema import GpuInfo, HardwareProbe, RuntimeFacts

CPU_TYPES = ["float32", "int16", "int8", "int8_float32"]
GPU_TYPES = ["bfloat16", "float16", "float32", "int8", "int8_bfloat16", "int8_float16"]


def hw(
    *gpus: GpuInfo,
    os: str = "linux",
    container: bool = False,
    env: Mapping[str, str] | None = None,
    smi: bool | None = None,
    kfd: bool | None = None,
    avx2: bool | None = True,
) -> HardwareProbe:
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


RTX4090 = GpuInfo(
    vendor="nvidia",
    name="NVIDIA GeForce RTX 4090",
    driver="555.42.06",
    vram_mb=24_564,
    compute_capability="8.9",
)
RTX5090 = GpuInfo(
    vendor="nvidia",
    name="NVIDIA GeForce RTX 5090",
    driver="575.51.03",
    vram_mb=32_000,
    compute_capability="12.0",
)
GTX1050 = GpuInfo(
    vendor="nvidia", name="NVIDIA GeForce GTX 1050", driver="536.23", vram_mb=2_048, compute_capability="6.1"
)
RX7900 = GpuInfo(vendor="amd", name="AMD Radeon RX 7900 XTX", gfx="gfx1100", integrated=False)
RX6800 = GpuInfo(vendor="amd", name="AMD Radeon RX 6800", gfx="gfx1030", integrated=False)
IGPU = GpuInfo(vendor="amd", name="AMD Radeon 890M", gfx="gfx1150", integrated=True)


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    """Run ``hibiki-asr <argv>`` in this process; returns (exit code, stdout, stderr)."""
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err
