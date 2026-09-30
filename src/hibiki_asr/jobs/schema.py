"""Job types. Part of the public API contract (see openapi.json)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ..diagnostics.schema import Finding


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    cancelling = "cancelling"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_STATES = frozenset({JobState.succeeded, JobState.failed, JobState.cancelled})


class JobParams(BaseModel):
    """The ``params`` part of ``POST /v1/jobs``."""

    model_id: str = Field(
        description="A model id (its active version) or 'id@version'. See GET /v1/models.",
        examples=["chickenrice", "whisper-ja@1.5b"],
    )
    task: Literal["transcribe", "translate"] | None = Field(
        default=None, description="Defaults to the model's own task. A different one is rejected."
    )
    language: str | None = Field(
        default=None,
        description="Spoken language of the audio, e.g. 'ja'. Defaults to the model's first source language.",
    )


class Segment(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class RuntimeInfo(BaseModel):
    device: Literal["cuda", "cpu"] = Field(
        description="What actually ran the model ('cuda' also means ROCm/HIP)."
    )
    compute_type: str
    vad_device: Literal["cuda", "cpu"]
    degraded: bool = Field(description="A GPU was expected but the CPU was used. See findings for why.")
    findings: list[Finding] = Field(default_factory=list)


class ModelUsed(BaseModel):
    id: str
    version: str


class JobResult(BaseModel):
    segments: list[Segment]
    vtt: str = Field(description="The same segments as WebVTT.")
    duration_ms: int = Field(description="Length of the audio.")
    speech_ms: int = Field(description="How much of it was speech, according to the VAD.")
    model: ModelUsed
    runtime: RuntimeInfo


class JobError(BaseModel):
    code: str = Field(
        description="Stable identifier, e.g. MODEL_LOAD_FAILED, WORKER_CRASHED, AUDIO_DECODE_FAILED."
    )
    message: str
    hint: str | None = None
    findings: list[Finding] = Field(default_factory=list)


class JobStatus(BaseModel):
    id: str
    state: JobState
    stage: str = Field(description="queued | loading | decoding | vad | transcribing | done")
    progress: float = Field(ge=0.0, le=1.0)
    message: str = ""
    queue_position: int | None = Field(default=None, description="1 = next to run. Null once it started.")
    created_at: float
    updated_at: float
    result: JobResult | None = None
    error: JobError | None = None
