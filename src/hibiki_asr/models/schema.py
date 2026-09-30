"""Model management types. Part of the public API contract (see openapi.json)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VersionStatus(str, Enum):
    not_installed = "not_installed"
    downloading = "downloading"
    installed = "installed"
    corrupt = "corrupt"


class RequirementInfo(BaseModel):
    id: str
    version: str
    display_name: str
    installed: bool


class ModelVersionInfo(BaseModel):
    version: str
    revision: str = Field(description="Pinned Hugging Face commit the files are downloaded from.")
    status: VersionStatus
    active: bool = Field(description="This is the version used when a job names the model without a version.")
    size_bytes: int | None = Field(default=None, description="Download size of the model's own files.")
    notes: str = ""


class ModelInfo(BaseModel):
    id: str
    display_name: str
    task: Literal["transcribe", "translate"]
    source_languages: list[str] = Field(description="Languages the model understands, e.g. ['ja'].")
    output_languages: list[str] = Field(
        description="Languages of the produced text, e.g. ['zh'] for a translate model."
    )
    license_note: str = ""
    requires: list[RequirementInfo] = Field(
        default_factory=list, description="Shared components downloaded with the model."
    )
    active_version: str | None = None
    latest_version: str
    update_available: bool = Field(
        description="A newer version than the installed ones exists in the catalog."
    )
    versions: list[ModelVersionInfo]


class DownloadState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


def _clean_endpoint(value: object) -> object:
    """An endpoint is an http(s) base URL; anything else is rejected before it reaches a request."""
    if value is None:
        return value
    if not isinstance(value, str):
        raise ValueError("endpoint must be a string")
    text = value.strip().rstrip("/")
    if not (text.startswith("https://") or text.startswith("http://")) or len(text.split("://", 1)[1]) == 0:
        raise ValueError("endpoint must be an http(s) URL such as https://hf-mirror.com")
    return text


class DownloadRequest(BaseModel):
    """Optional per-download overrides. Everything left out uses the engine settings."""

    endpoint: str | None = Field(
        default=None,
        description="Download from this Hugging Face compatible endpoint first, e.g. https://hf-mirror.com.",
    )
    fallback: bool = Field(
        default=True,
        description="Keep trying the configured endpoints and mirrors when `endpoint` fails. "
        "False downloads from `endpoint` only.",
    )
    threads: int | None = Field(default=None, ge=1, le=16, description="Parallel connections per large file.")

    _validate_endpoint = field_validator("endpoint", mode="before")(_clean_endpoint)


class DownloadStatus(BaseModel):
    id: str
    model_id: str
    version: str
    state: DownloadState
    stage: str = Field(description="queued | downloading | verifying | done")
    source: str | None = Field(default=None, description="Endpoint the data is currently coming from.")
    file: str | None = None
    bytes_done: int = 0
    bytes_total: int | None = None
    bytes_per_second: float = 0.0
    error: str | None = None


class SetActiveRequest(BaseModel):
    version: str


class VerifyResult(BaseModel):
    ok: bool
    checked_files: int
    problems: list[str] = Field(default_factory=list, description="One line per damaged or missing file.")


class CatalogRefreshResult(BaseModel):
    refreshed: bool
    models: int
    message: str


class SourceKind(str, Enum):
    official = "official"
    configured = "configured"
    mirror = "mirror"
    custom = "custom"


class SourceStatus(BaseModel):
    endpoint: str
    kind: SourceKind
    reachable: bool = Field(description="A small pinned file was fetched and its sha256 matched the catalog.")
    latency_ms: int | None = None
    error: str | None = None


class SourcesResponse(BaseModel):
    huggingface_reachable: bool = Field(
        description="The official https://huggingface.co can be reached directly."
    )
    recommended_endpoint: str | None = Field(
        default=None, description="The fastest source that works, or null when none does."
    )
    default_endpoint: str = Field(description="What downloads use first when a request names no endpoint.")
    sources: list[SourceStatus]
    checked_at: float = Field(description="Unix time of the probe.")


class ProbeSourcesRequest(BaseModel):
    endpoints: list[str] = Field(
        default_factory=list, description="Extra endpoints to test on top of the configured ones."
    )

    @field_validator("endpoints", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        if isinstance(value, list):
            return [_clean_endpoint(v) for v in value]
        return value
