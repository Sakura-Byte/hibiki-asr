"""Small API types that do not belong to models, jobs or diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"]
    api_version: int


class VersionInfo(BaseModel):
    api_version: int = Field(description="Bumped only for breaking API changes. Clients require a minimum.")
    engine_version: str
    variant: str | None = Field(default=None, description="Installed runtime variant, if known.")
    capabilities: list[str] = Field(
        description="Optional features this engine offers. Clients gate features on these."
    )


class ErrorBody(BaseModel):
    code: str = Field(description="Stable identifier, e.g. MODEL_NOT_INSTALLED.")
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
