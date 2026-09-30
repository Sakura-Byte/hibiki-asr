"""Messages between the API process and the inference worker. Plain, picklable data only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..diagnostics.schema import Finding


@dataclass(frozen=True)
class RunRequest:
    job_id: str
    audio_path: str
    model_ref: str  # "id@version", for reporting
    model_dir: str
    components: dict[str, str]  # component id -> directory
    language: str
    task: str  # "transcribe" | "translate"
    device: str  # requested: auto | cpu | cuda | amd
    compute_type: str
    allow_cpu_fallback: bool
    vad: dict[str, Any]
    merge: dict[str, Any]
    generation: dict[str, Any]
    chunk_target_s: float
    vad_threads: int
    cpu_threads: int
    variant: str | None = None


@dataclass(frozen=True)
class Shutdown:
    """Ask the worker to exit cleanly."""


# -- worker -> parent -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Progress:
    stage: str  # loading | decoding | vad | transcribing
    fraction: float  # 0..1 over the whole job
    message: str


@dataclass(frozen=True)
class RunResult:
    segments: list[tuple[int, int, str]]  # (start_ms, end_ms, text)
    duration_s: float
    speech_s: float
    device: str
    compute_type: str
    vad_device: str
    degraded: bool
    model_ref: str
    findings: list[Finding] = field(default_factory=list)


@dataclass(frozen=True)
class RunFailed:
    code: str
    message: str
    hint: str | None = None
    findings: list[Finding] = field(default_factory=list)


@dataclass(frozen=True)
class RunCancelled:
    """The job noticed the cancel flag and stopped."""
