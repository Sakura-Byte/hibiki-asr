"""The committed openapi.json is the contract clients build against; it must match the code."""

from __future__ import annotations

import json
from pathlib import Path

from hibiki_asr import API_VERSION
from hibiki_asr.api.openapi import render_openapi

SPEC = Path(__file__).resolve().parents[1] / "openapi.json"


def test_the_committed_spec_is_current() -> None:
    assert SPEC.read_text(encoding="utf-8") == render_openapi(), (
        "openapi.json is stale: run `python scripts/export_openapi.py` and commit it"
    )


def test_the_spec_declares_the_api_version() -> None:
    assert json.loads(SPEC.read_text(encoding="utf-8"))["info"]["x-api-version"] == API_VERSION


def test_every_operation_documents_its_error_shape_and_has_a_summary_or_name() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            assert operation["operationId"], (method, path)
            responses = operation["responses"]
            if path != "/v1/healthz":
                assert {"401", "403"} <= set(responses), f"{method} {path} does not document auth errors"


def test_the_contract_clients_rely_on() -> None:
    """The paths, methods and fields Hibiki's client uses. Removing or renaming one is a breaking change."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    ops = {(m.upper(), p) for p, methods in spec["paths"].items() for m in methods}
    assert {
        ("GET", "/v1/healthz"),
        ("GET", "/v1/version"),
        ("GET", "/v1/diagnostics"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/sources"),
        ("POST", "/v1/models/sources/probe"),
        ("POST", "/v1/models/refresh"),
        ("POST", "/v1/models/{model_id}/versions/{version}/download"),
        ("GET", "/v1/models/downloads/{download_id}"),
        ("DELETE", "/v1/models/downloads/{download_id}"),
        ("PUT", "/v1/models/{model_id}/active"),
        ("POST", "/v1/models/{model_id}/versions/{version}/verify"),
        ("DELETE", "/v1/models/{model_id}/versions/{version}"),
        ("POST", "/v1/jobs"),
        ("GET", "/v1/jobs/{job_id}"),
        ("DELETE", "/v1/jobs/{job_id}"),
    } <= ops

    schemas = spec["components"]["schemas"]
    assert {"queued", "running", "cancelling", "succeeded", "failed", "cancelled"} == set(
        schemas["JobState"]["enum"]
    )
    assert {"id", "state", "stage", "progress", "queue_position", "result", "error"} <= set(
        schemas["JobStatus"]["properties"]
    )
    assert {"segments", "vtt", "duration_ms", "model", "runtime"} <= set(schemas["JobResult"]["properties"])
    assert {"endpoint", "fallback", "threads"} == set(schemas["DownloadRequest"]["properties"])
    assert {"huggingface_reachable", "recommended_endpoint", "sources"} <= set(
        schemas["SourcesResponse"]["properties"]
    )
    assert {"code", "severity", "message", "hint"} == set(schemas["Finding"]["properties"])
