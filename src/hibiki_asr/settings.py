"""Engine settings: defaults < config file < HIBIKI_ASR_* environment < command line.

Hardware and performance policy lives here, not in the client. The machine that
runs the engine is the one that knows its GPU, so it decides the device.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

ENV_PREFIX = "HIBIKI_ASR_"
DEFAULT_PORT = 8001
DEFAULT_CATALOG_URL = "https://raw.githubusercontent.com/Sakura-Byte/hibiki-asr/main/src/hibiki_asr/models/catalog.json"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def default_config_dir(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if platform.startswith("win"):
        return Path(env.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "hibiki-asr"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "hibiki-asr"
    return Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "hibiki-asr"


def default_data_dir(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if platform.startswith("win"):
        return Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "hibiki-asr"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "hibiki-asr"
    return Path(env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "hibiki-asr"


class VadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Higher misses quiet speech; lower lets noise through and can cause hallucinated text.
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(300, ge=0)
    min_silence_duration_ms: int = Field(100, ge=0)
    speech_pad_ms: int = Field(200, ge=0)
    threads: int = Field(0, ge=0, description="ONNX Runtime threads for VAD on CPU; 0 = half of the cores.")


class MergeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_gap_ms: int = Field(2000, ge=0)
    max_duration_ms: int = Field(20000, ge=1)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(DEFAULT_PORT, ge=1, le=65535)
    token: str | None = Field(None, description="Bearer token. Required when listening on a non-loopback address.")

    data_dir: Path = Field(default_factory=default_data_dir)
    models_dir: Path | None = Field(None, description="Defaults to <data_dir>/models.")

    device: Literal["auto", "cpu", "cuda", "amd"] = "auto"
    compute_type: str = "auto"
    allow_cpu_fallback: bool = True

    idle_unload_seconds: int = Field(600, ge=0, description="Free the model (and VRAM) after this long without work; 0 = never.")
    cancel_grace_seconds: float = Field(10.0, ge=0.0, description="How long a job may take to notice a cancel before the worker is killed.")
    job_ttl_seconds: int = Field(3600, ge=60, description="How long finished jobs stay queryable.")
    max_upload_mb: int = Field(8192, ge=1)

    chunk_target_s: float = Field(30.0, gt=0.0, le=30.0)
    vad: VadSettings = Field(default_factory=VadSettings)
    merge: MergeSettings = Field(default_factory=MergeSettings)
    generation: dict[str, Any] = Field(
        default_factory=dict, description="Extra faster-whisper transcribe() arguments, merged over the defaults."
    )

    # Where models are downloaded from. `hf_endpoint` is tried first, then each of `hf_mirrors`.
    # In regions where Hugging Face is slow or blocked, set `hf_endpoint` to a mirror such as https://hf-mirror.com.
    # The standard HF_ENDPOINT and HF_TOKEN environment variables are honoured too.
    hf_endpoint: str = "https://huggingface.co"
    hf_mirrors: list[str] = Field(default_factory=lambda: ["https://hf-mirror.com"])
    hf_token: str | None = Field(None, description="Access token for private or gated repositories.")
    download_threads: int = Field(4, ge=1, le=16, description="Parallel connections per large file; 1 disables splitting.")
    catalog_url: str = DEFAULT_CATALOG_URL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("device", mode="before")
    @classmethod
    def _device_case(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _level_case(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("hf_mirrors", mode="before")
    @classmethod
    def _split_mirrors(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def resolved_models_dir(self) -> Path:
        return self.models_dir or self.data_dir / "models"

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def validate_for_serving(self) -> None:
        """Refuse an unauthenticated engine that anyone on the network could use."""
        if not self.is_loopback and not self.token:
            raise ValueError(
                f"refusing to listen on {self.host} without a token: set HIBIKI_ASR_TOKEN "
                "(or `token` in the config file), or bind to 127.0.0.1"
            )


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc


# Standard Hugging Face variables. Their HIBIKI_ASR_* equivalents win when both are set.
_HF_ENV_ALIASES = {"HF_ENDPOINT": "hf_endpoint", "HF_TOKEN": "hf_token"}


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """HIBIKI_ASR_DEVICE=cpu, HIBIKI_ASR_VAD__THRESHOLD=0.4 (double underscore nests)."""
    out: dict[str, Any] = {name: env[var] for var, name in _HF_ENV_ALIASES.items() if env.get(var)}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX) or key in {ENV_PREFIX + "CONFIG"}:
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("__")
        target = out
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return out


def _deep_merge(base: dict[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_file_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env[ENV_PREFIX + "CONFIG"]) if env.get(ENV_PREFIX + "CONFIG") else default_config_dir(env) / "hibiki-asr.toml"


def load_settings(
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
) -> Settings:
    """Layer the sources. ``overrides`` are command line values and win over everything."""
    env = os.environ if env is None else env
    merged = _read_config_file(config_path or config_file_path(env))
    merged = _deep_merge(merged, _env_overrides(env))
    merged = _deep_merge(merged, {k: v for k, v in (overrides or {}).items() if v is not None})
    return Settings.model_validate(merged)


def _parse_free_form(raw: str) -> Any:
    """Values of the free-form ``generation`` table: ``5`` -> 5, ``true`` -> True, ``[0,0.2]`` -> list, else text."""
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_config_value(path: Path, dotted_key: str, raw_value: str) -> Settings:
    """Set ``key`` or ``section.key`` in the config file after validating the whole result."""
    current = _read_config_file(path)
    section, _, leaf = dotted_key.rpartition(".")
    target = current
    if section:
        target = current.setdefault(section, {})
        if not isinstance(target, dict):
            raise ValueError(f"{section} is not a table")
    target[leaf] = _parse_free_form(raw_value) if section == "generation" else raw_value
    validated = Settings.model_validate(current)  # rejects unknown keys and bad values before anything is written

    # Re-read the typed value so the file holds numbers and booleans, not strings.
    typed = validated.model_dump(mode="json")
    for part in section.split(".") if section else []:
        typed = typed[part]
    target[leaf] = typed[leaf]

    lines: list[str] = []
    tables: dict[str, dict[str, Any]] = {}
    for key, value in current.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        lines.extend(["", f"[{name}]", *(f"{k} = {_toml_value(v)}" for k, v in table.items())])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validated
