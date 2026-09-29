"""Typed diagnostics. These models are part of the public API contract (see openapi.json)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class Finding(BaseModel):
    """One observation about the environment, with a fix the user can act on."""

    code: str = Field(description="Stable machine readable identifier, e.g. CT2_NO_GPU_SUPPORT.")
    severity: Severity
    message: str = Field(description="What was detected.")
    hint: str | None = Field(default=None, description="What to do about it. Commands are copy-pasteable.")


class GpuInfo(BaseModel):
    vendor: Literal["nvidia", "amd", "intel", "other"]
    name: str
    vram_mb: int | None = None
    driver: str | None = None
    compute_capability: str | None = Field(default=None, description="NVIDIA only, e.g. 8.9.")
    gfx: str | None = Field(default=None, description="AMD only, e.g. gfx1100.")
    integrated: bool | None = None


class HardwareProbe(BaseModel):
    """What the machine has. Collected with the standard library only."""

    os: str
    arch: str
    in_container: bool
    cpu_model: str | None = None
    cpu_cores: int | None = None
    cpu_avx2: bool | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    nvidia_smi_found: bool = False
    kfd_present: bool | None = Field(default=None, description="Linux only: /dev/kfd exists (ROCm compute device).")
    env: dict[str, str] = Field(default_factory=dict, description="GPU related environment variables that are set.")


class RuntimeFacts(BaseModel):
    """What the installed inference stack can do. Collected in a subprocess so a broken install cannot crash the API."""

    probe_ok: bool = True
    probe_error: str | None = None
    python: str | None = None
    ctranslate2_version: str | None = None
    ctranslate2_error: str | None = None
    cuda_device_count: int = 0
    compute_types: dict[str, list[str]] = Field(default_factory=dict)
    onnxruntime_version: str | None = None
    onnxruntime_providers: list[str] = Field(default_factory=list)
    onnxruntime_error: str | None = None
    faster_whisper_version: str | None = None
    faster_whisper_error: str | None = None


class Selection(BaseModel):
    """The device and precision the engine will use, and whether that is a step down from what was wanted."""

    requested_device: str
    device: Literal["cuda", "cpu"]
    compute_type: str
    degraded: bool = Field(description="A GPU was expected (requested, or present under 'auto') but the CPU is used.")
    vad_device: Literal["cuda", "cpu"]


class Diagnostics(BaseModel):
    api_version: int
    engine_version: str
    variant: str | None = Field(default=None, description="Variant installed by `hibiki-asr setup`, if known.")
    hardware: HardwareProbe
    runtime: RuntimeFacts
    selection: Selection
    findings: list[Finding]
    settings: dict[str, object] = Field(description="Effective engine settings (read-only).")
