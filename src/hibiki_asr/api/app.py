"""The HTTP API (``/v1``). Thin: every route delegates to the engine's managers."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import API_VERSION
from ..diagnostics.schema import Diagnostics
from ..engine import Engine, engine_version
from ..jobs.schema import JobParams, JobStatus
from ..models.catalog import Ref
from ..models.manager import ModelsError, Unprocessable
from ..models.schema import (
    CatalogRefreshResult,
    DownloadRequest,
    DownloadStatus,
    ModelInfo,
    ProbeSourcesRequest,
    SetActiveRequest,
    SourcesResponse,
    VerifyResult,
)
from .schema import ErrorResponse, Health, VersionInfo

CAPABILITIES = [
    "jobs",
    "job-cancel",
    "job-upload",
    "diagnostics",
    "models",
    "model-versions",
    "model-download",
    "model-download-sources",
    "model-catalog-refresh",
]

_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or malformed Authorization header."},
    403: {"model": ErrorResponse, "description": "Wrong token."},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
_CHUNK = 1024 * 1024


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(engine: Engine) -> FastAPI:
    settings = engine.settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Probing can take a few seconds; do it off the request path and log the outcome for the operator.
        threading.Thread(target=engine.log_startup_report, name="startup-report", daemon=True).start()
        yield
        engine.shutdown()

    app = FastAPI(
        title="hibiki-asr",
        version=engine_version(),
        summary="Local Whisper speech recognition with a cancellable job API.",
        description=(
            "Submit audio, poll the job, cancel it if needed. Models are installed and versioned through the "
            "`/v1/models` endpoints; `/v1/diagnostics` explains which device is used and why."
        ),
        lifespan=lifespan,
        responses=_ERRORS,
    )

    # FastAPI has no constructor option for extra `info` fields, so stamp the API version into the spec here.
    generate = app.openapi

    def openapi_with_api_version() -> dict[str, Any]:
        schema = generate()
        schema["info"]["x-api-version"] = API_VERSION
        return schema

    app.openapi = openapi_with_api_version  # type: ignore[method-assign]

    # -- errors -----------------------------------------------------------------------------------

    @app.exception_handler(ModelsError)
    async def _models_error(_request: Request, exc: ModelsError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return _error(422, "INVALID_REQUEST", problems)

    @app.exception_handler(HTTPException)
    async def _http(_request: Request, exc: HTTPException) -> JSONResponse:
        code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 413: "PAYLOAD_TOO_LARGE"}.get(exc.status_code, "ERROR")
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        response = _error(exc.status_code, code, str(exc.detail))
        if headers:
            response.headers.update(headers)
        return response

    # -- auth -------------------------------------------------------------------------------------

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not settings.token:
            return
        if not authorization:
            raise HTTPException(401, "missing Authorization header; send `Authorization: Bearer <token>`")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "malformed Authorization header; expected `Bearer <token>`")
        if not secrets.compare_digest(token.strip().encode(), settings.token.encode()):
            raise HTTPException(403, "invalid token")

    protected = [Depends(require_token)]

    # -- meta -------------------------------------------------------------------------------------

    @app.get(
        "/v1/healthz", tags=["meta"], response_model=Health, summary="Is the engine up? (no token needed)"
    )
    def healthz() -> Health:
        return Health(status="ok", api_version=API_VERSION)

    @app.get("/v1/version", tags=["meta"], response_model=VersionInfo, dependencies=protected)
    def version() -> VersionInfo:
        return VersionInfo(
            api_version=API_VERSION,
            engine_version=engine_version(),
            variant=engine.variant,
            capabilities=CAPABILITIES,
        )

    @app.get(
        "/v1/diagnostics",
        tags=["meta"],
        response_model=Diagnostics,
        dependencies=protected,
        summary="Hardware, the device in use, and findings that explain a CPU fallback",
    )
    def diagnostics(refresh: bool = False) -> Diagnostics:
        return engine.diagnostics(refresh=refresh)

    # -- models -----------------------------------------------------------------------------------

    models = engine.models

    @app.get("/v1/models", tags=["models"], response_model=list[ModelInfo], dependencies=protected)
    def list_models() -> list[ModelInfo]:
        return models.list_models()

    @app.get(
        "/v1/models/sources",
        tags=["models"],
        response_model=SourcesResponse,
        dependencies=protected,
        summary="Can Hugging Face be reached directly, and which mirror is best?",
    )
    def sources(refresh: bool = False) -> SourcesResponse:
        return models.sources(refresh=refresh)

    @app.post(
        "/v1/models/sources/probe",
        tags=["models"],
        response_model=SourcesResponse,
        dependencies=protected,
        summary="Test extra endpoints, e.g. a mirror the user typed in",
    )
    def probe_sources(request: ProbeSourcesRequest) -> SourcesResponse:
        return models.sources(refresh=True, extra=request.endpoints)

    @app.post(
        "/v1/models/refresh",
        tags=["models"],
        response_model=CatalogRefreshResult,
        dependencies=protected,
        summary="Fetch a newer model catalog",
    )
    def refresh_catalog() -> CatalogRefreshResult:
        return models.refresh_catalog()

    @app.get(
        "/v1/models/downloads", tags=["models"], response_model=list[DownloadStatus], dependencies=protected
    )
    def list_downloads() -> list[DownloadStatus]:
        return models.list_downloads()

    @app.get(
        "/v1/models/downloads/{download_id}",
        tags=["models"],
        response_model=DownloadStatus,
        dependencies=protected,
    )
    def get_download(download_id: str) -> DownloadStatus:
        return models.get_download(download_id)

    @app.delete(
        "/v1/models/downloads/{download_id}",
        tags=["models"],
        response_model=DownloadStatus,
        status_code=202,
        dependencies=protected,
        summary="Cancel a download (idempotent; partial data is kept for resuming)",
    )
    def cancel_download(download_id: str) -> DownloadStatus:
        return models.cancel_download(download_id)

    @app.get("/v1/models/{model_id}", tags=["models"], response_model=ModelInfo, dependencies=protected)
    def get_model(model_id: str) -> ModelInfo:
        return models.get_model(model_id)

    @app.post(
        "/v1/models/{model_id}/versions/{version}/download",
        tags=["models"],
        response_model=DownloadStatus,
        status_code=202,
        dependencies=protected,
        summary="Download a model version together with the components it needs",
    )
    def download(
        model_id: str, version: str, request: Annotated[DownloadRequest | None, Body()] = None
    ) -> DownloadStatus:
        return models.start_download(Ref(model_id, version), request)

    @app.put(
        "/v1/models/{model_id}/active",
        tags=["models"],
        response_model=ModelInfo,
        dependencies=protected,
        summary="Choose which installed version jobs use by default",
    )
    def set_active(model_id: str, request: SetActiveRequest) -> ModelInfo:
        return models.set_active(model_id, request.version)

    @app.post(
        "/v1/models/{model_id}/versions/{version}/verify",
        tags=["models"],
        response_model=VerifyResult,
        dependencies=protected,
        summary="Re-hash the installed files",
    )
    def verify(model_id: str, version: str) -> VerifyResult:
        return models.verify(Ref(model_id, version))

    @app.delete(
        "/v1/models/{model_id}/versions/{version}", tags=["models"], status_code=204, dependencies=protected
    )
    def delete_version(model_id: str, version: str) -> Response:
        models.delete(Ref(model_id, version))
        return Response(status_code=204)

    # -- jobs -------------------------------------------------------------------------------------

    def _save_upload(upload: UploadFile, job_id: str) -> Path:
        suffix = Path(upload.filename or "").suffix.lower()
        suffix = suffix if _SAFE_SUFFIX.match(suffix) else ""
        directory = settings.data_dir / "jobs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"audio{suffix}"
        limit = settings.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with open(target, "wb") as handle:
                while chunk := upload.file.read(_CHUNK):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            413, f"the audio is larger than max_upload_mb ({settings.max_upload_mb} MB)"
                        )
                    handle.write(chunk)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        if written == 0:
            shutil.rmtree(directory, ignore_errors=True)
            raise ModelsError("EMPTY_AUDIO", "the uploaded audio file is empty")
        return target

    @app.post(
        "/v1/jobs",
        tags=["jobs"],
        response_model=JobStatus,
        status_code=202,
        dependencies=protected,
        summary="Submit audio for transcription (multipart: `params` JSON + `audio` file)",
    )
    def create_job(
        params: Annotated[str, Form(description="JSON of JobParams")],
        audio: Annotated[UploadFile, File(description="The audio (or video) file")],
    ) -> JobStatus:
        try:
            parsed = JobParams.model_validate(json.loads(params))
        except (ValueError, ValidationError) as exc:
            raise ModelsError("INVALID_REQUEST", f"params is not valid JobParams JSON: {exc}") from exc

        resolved = models.resolve(parsed.model_id)  # NotFound / ModelNotInstalled before anything is stored
        entry = resolved.entry
        task = parsed.task or entry.task or "transcribe"
        if task != entry.task:
            raise Unprocessable(
                "TASK_NOT_SUPPORTED", f"{entry.id} does its own task ({entry.task}); it cannot {task}"
            )
        language = parsed.language or entry.source_languages[0]
        if language not in entry.source_languages:
            raise Unprocessable(
                "LANGUAGE_NOT_SUPPORTED",
                f"{entry.id} understands {', '.join(entry.source_languages)}, not {language!r}",
            )

        job_id = uuid.uuid4().hex
        path = _save_upload(audio, job_id)
        return engine.jobs.submit(resolved, task, language, path, variant=engine.variant, job_id=job_id)

    @app.get("/v1/jobs/{job_id}", tags=["jobs"], response_model=JobStatus, dependencies=protected)
    def get_job(job_id: str) -> JobStatus:
        return engine.jobs.get(job_id)

    @app.delete(
        "/v1/jobs/{job_id}",
        tags=["jobs"],
        response_model=JobStatus,
        status_code=202,
        dependencies=protected,
        summary="Cancel a job (idempotent)",
    )
    def cancel_job(job_id: str) -> JobStatus:
        return engine.jobs.cancel(job_id)

    return app
