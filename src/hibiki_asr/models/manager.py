"""Model management: what is installed, downloads, versions, dependencies, catalog refresh."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..settings import Settings, default_config_dir
from .adopt import find_problems, stage_files
from .catalog import (
    Catalog,
    CatalogError,
    Entry,
    Ref,
    VersionSpec,
    load_bundled,
    merge_entries,
    parse_catalog,
    parse_local_toml,
)
from .download import DownloadCancelled, Downloader, DownloadError, DownloadProgress, sha256_of
from .schema import (
    CatalogRefreshResult,
    DownloadRequest,
    DownloadState,
    DownloadStatus,
    ModelInfo,
    ModelVersionInfo,
    RequirementInfo,
    SourcesResponse,
    VerifyResult,
    VersionStatus,
)
from .sources import probe_sources
from .store import ModelStore

logger = logging.getLogger(__name__)

SOURCES_CACHE_SECONDS = 60.0
MAX_REMEMBERED_DOWNLOADS = 50
MAX_CATALOG_BYTES = 1024 * 1024


class ModelsError(Exception):
    """A model operation was refused. ``code`` is stable, ``message`` is safe to show to a user."""

    status = 400

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NotFound(ModelsError):
    status = 404


class Conflict(ModelsError):
    status = 409


class Unprocessable(ModelsError):
    """The request is well formed but cannot be done, e.g. a language the model does not speak."""

    status = 422


class ModelNotInstalled(Conflict):
    """The requested model (or one of its components) is not installed."""


@dataclass(frozen=True)
class ImportResult:
    ref: Ref
    files: int
    size_bytes: int
    moved: bool


@dataclass(frozen=True)
class ResolvedModel:
    ref: Ref
    entry: Entry
    version: VersionSpec
    model_dir: Path
    components: dict[str, Path]  # component id -> installed directory


@dataclass
class _Job:
    id: str
    target: Ref
    cancel: threading.Event = field(default_factory=threading.Event)
    status: DownloadStatus | None = None
    future: Future[None] | None = None

    def snapshot(self) -> DownloadStatus:
        assert self.status is not None
        return self.status.model_copy()


def load_catalog(settings: Settings, config_dir: Path | None = None) -> tuple[Catalog, list[str]]:
    """Bundled catalog, then the cached refresh, then the user's ``catalog.local.toml``. Returns (catalog, warnings)."""
    warnings: list[str] = []
    base = load_bundled()
    layers = []

    cache = settings.data_dir / "catalog.cache.json"
    if cache.is_file():
        try:
            layers.append(parse_catalog(json.loads(cache.read_text(encoding="utf-8")), require_hashes=True))
        except (OSError, ValueError, CatalogError) as exc:
            warnings.append(f"ignoring the cached catalog {cache}: {exc}")

    local = (config_dir or default_config_dir()) / "catalog.local.toml"
    if local.is_file():
        try:
            layers.append(parse_local_toml(local.read_text(encoding="utf-8")))
        except (OSError, CatalogError) as exc:
            warnings.append(f"ignoring {local}: {exc}")

    # A layer that breaks the requirement graph must not take the whole catalog down with it.
    merged = merge_entries(base)
    for layer in layers:
        try:
            merged = merge_entries(merged.entries.values(), layer)
        except CatalogError as exc:
            warnings.append(f"ignoring a catalog source: {exc}")
    return merged, warnings


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        store: ModelStore,
        *,
        client_factory: Callable[[], httpx.Client] | None = None,
        warnings: list[str] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._store = store
        # A blocked host usually hangs on connect: fail that fast, but allow a slow read on a big file.
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0, read=60.0))
        )
        self._clock = clock
        self._sleep = sleep
        self.catalog_warnings = list(warnings or [])

        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-download")
        self._sources: tuple[float, SourcesResponse] | None = None

    # -- read side --------------------------------------------------------------------------------

    @property
    def catalog(self) -> Catalog:
        return self._catalog

    def _busy(self, ref: Ref) -> bool:
        with self._lock:
            return any(
                j.target == ref
                and j.status is not None
                and j.status.state in (DownloadState.queued, DownloadState.running)
                for j in self._jobs.values()
            )

    def _status_of(self, ref: Ref) -> VersionStatus:
        if self._busy(ref):
            return VersionStatus.downloading
        if not self._store.is_installed(ref):
            return VersionStatus.not_installed
        return VersionStatus.installed if self._store.is_intact(ref) else VersionStatus.corrupt

    def _effective_active(self, entry: Entry) -> str | None:
        """The pinned active version if it is installed, else the newest installed one."""
        pinned = self._store.active(entry.id)
        installed = set(self._store.installed_versions(entry.id))
        if pinned in installed:
            return pinned
        for spec in reversed(entry.versions):  # the catalog lists versions oldest to newest
            if spec.version in installed:
                return spec.version
        return None

    def _info(self, entry: Entry) -> ModelInfo:
        active = self._effective_active(entry)
        installed = self._store.installed_versions(entry.id)
        return ModelInfo(
            id=entry.id,
            display_name=entry.display_name,
            task=entry.task,  # the catalog parser guarantees a model task
            source_languages=list(entry.source_languages),
            output_languages=list(entry.output_languages),
            license_note=entry.license_note,
            requires=[
                RequirementInfo(
                    id=r.id,
                    version=r.version,
                    display_name=(self._catalog.get(r.id).display_name if self._catalog.get(r.id) else r.id),  # type: ignore[union-attr]
                    installed=self._store.is_installed(r),
                )
                for r in entry.requires
            ],
            active_version=active,
            latest_version=entry.latest.version,
            update_available=bool(installed) and entry.latest.version not in installed,
            versions=[
                ModelVersionInfo(
                    version=v.version,
                    revision=v.revision,
                    status=self._status_of(Ref(entry.id, v.version)),
                    active=v.version == active,
                    size_bytes=v.total_size,
                    notes=v.notes,
                )
                for v in entry.versions
            ],
        )

    def list_models(self) -> list[ModelInfo]:
        return [self._info(e) for e in self._catalog.models()]

    def get_model(self, model_id: str) -> ModelInfo:
        entry = self._catalog.get(model_id)
        if entry is None or entry.kind != "model":
            raise NotFound("MODEL_NOT_FOUND", f"unknown model {model_id!r}")
        return self._info(entry)

    def resolve(self, model_ref: str) -> ResolvedModel:
        """``"id"`` (its active version) or ``"id@version"`` -> the installed files, or ModelNotInstalled."""
        ident, _, requested = model_ref.partition("@")
        entry = self._catalog.get(ident)
        if entry is None or entry.kind != "model":
            raise NotFound("MODEL_NOT_FOUND", f"unknown model {ident!r}")

        version = requested or self._effective_active(entry)
        if not version:
            raise ModelNotInstalled(
                "MODEL_NOT_INSTALLED",
                f"model {entry.id!r} is not installed. Download it first (Hibiki: Admin > Local engine > Models).",
            )
        ref = Ref(entry.id, version)
        try:
            _, spec = self._catalog.resolve(ref)
        except KeyError:
            raise NotFound("VERSION_NOT_FOUND", f"model {entry.id!r} has no version {version!r}") from None
        if not self._store.is_installed(ref):
            raise ModelNotInstalled("MODEL_NOT_INSTALLED", f"{ref} is not installed. Download it first.")
        if not self._store.is_intact(ref):
            raise ModelNotInstalled(
                "MODEL_CORRUPT", f"{ref} is damaged (a file is missing). Delete it and download it again."
            )

        components: dict[str, Path] = {}
        for requirement in entry.requires:
            if not self._store.is_intact(requirement):
                raise ModelNotInstalled(
                    "MODEL_NOT_INSTALLED",
                    f"{ref} needs {requirement}, which is not installed. Download {entry.id} again to fetch it.",
                )
            components[requirement.id] = self._store.version_dir(requirement)
        return ResolvedModel(ref, entry, spec, self._store.version_dir(ref), components)

    # -- versions -----------------------------------------------------------------------------------

    def set_active(self, model_id: str, version: str) -> ModelInfo:
        entry = self._catalog.get(model_id)
        if entry is None or entry.kind != "model":
            raise NotFound("MODEL_NOT_FOUND", f"unknown model {model_id!r}")
        if entry.version(version) is None:
            raise NotFound("VERSION_NOT_FOUND", f"model {model_id!r} has no version {version!r}")
        if not self._store.is_installed(Ref(model_id, version)):
            raise Conflict("MODEL_NOT_INSTALLED", f"{model_id}@{version} is not installed; download it first")
        self._store.set_active(model_id, version)
        return self._info(entry)

    def delete(self, ref: Ref) -> None:
        try:
            entry, _ = self._catalog.resolve(ref)
        except KeyError:
            raise NotFound("VERSION_NOT_FOUND", f"unknown model version {ref}") from None
        if self._busy(ref):
            raise Conflict("DOWNLOAD_IN_PROGRESS", f"{ref} is being downloaded; cancel the download first")

        if entry.kind == "component":
            users = [d for d in self._catalog.dependents_of(ref) if self._store.is_installed(d)]
            if users:
                raise Conflict(
                    "IN_USE", f"{ref} is needed by {', '.join(str(u) for u in users)}; delete those first"
                )
        else:
            installed = self._store.installed_versions(entry.id)
            if self._effective_active(entry) == ref.version and len(installed) > 1:
                raise Conflict(
                    "ACTIVE_VERSION",
                    f"{ref} is the active version. Make another installed version active before deleting it.",
                )

        self._store.delete(ref)
        if self._store.active(entry.id) == ref.version:
            self._store.set_active(entry.id, None)

    def verify(self, ref: Ref) -> VerifyResult:
        """Hash every file of an installed version against the checksums recorded when it was installed."""
        if not self._store.is_installed(ref):
            raise Conflict("MODEL_NOT_INSTALLED", f"{ref} is not installed")
        marker = self._store.read_marker(ref) or {}
        root = self._store.version_dir(ref)
        problems: list[str] = []
        files = marker.get("files", [])
        for f in files:
            target = root / f["path"]
            if not target.is_file():
                problems.append(f"{f['path']}: missing")
            elif f.get("size") is not None and target.stat().st_size != f["size"]:
                problems.append(f"{f['path']}: expected {f['size']} bytes, found {target.stat().st_size}")
            elif f.get("sha256") and sha256_of(target) != f["sha256"]:
                problems.append(f"{f['path']}: checksum mismatch")
        return VerifyResult(ok=not problems, checked_files=len(files), problems=problems)

    # -- downloads --------------------------------------------------------------------------------------

    def _endpoints(self, request: DownloadRequest) -> list[str]:
        configured = [self._settings.hf_endpoint, *self._settings.hf_mirrors]
        order = (
            [request.endpoint, *configured]
            if request.endpoint and request.fallback
            else ([request.endpoint] if request.endpoint else configured)
        )
        return list(dict.fromkeys(e.rstrip("/") for e in order if e))  # de-duplicated, order kept

    def start_download(self, ref: Ref, request: DownloadRequest | None = None) -> DownloadStatus:
        request = request or DownloadRequest()
        try:
            entry, spec = self._catalog.resolve(ref)
        except KeyError:
            raise NotFound("VERSION_NOT_FOUND", f"unknown model version {ref}") from None
        if entry.kind != "model":
            raise NotFound(
                "MODEL_NOT_FOUND", f"{ref.id!r} is a component; download a model, it brings its components"
            )

        with self._lock:
            for job in self._jobs.values():
                if (
                    job.target == ref
                    and job.status
                    and job.status.state in (DownloadState.queued, DownloadState.running)
                ):
                    return job.snapshot()  # already on its way: one download per version

            if self._status_of(ref) is VersionStatus.corrupt:
                self._store.delete(ref)

            job = _Job(id=uuid.uuid4().hex, target=ref)
            job.status = DownloadStatus(
                id=job.id,
                model_id=ref.id,
                version=ref.version,
                state=DownloadState.queued,
                stage="queued",
                bytes_total=self._plan_total(entry, spec),
            )
            self._jobs[job.id] = job
            self._forget_old_jobs()
            job.future = self._pool.submit(self._run_download, job, entry, spec, request)
            return job.snapshot()

    def _missing(self, entry: Entry, spec: VersionSpec) -> list[tuple[Ref, VersionSpec]]:
        items: list[tuple[Ref, VersionSpec]] = []
        for requirement in entry.requires:
            _, component_spec = self._catalog.resolve(requirement)
            if not self._store.is_intact(requirement):
                items.append((requirement, component_spec))
        target = Ref(entry.id, spec.version)
        if not self._store.is_intact(target):
            items.append((target, spec))
        return items

    def _plan_total(self, entry: Entry, spec: VersionSpec) -> int | None:
        sizes = [s.total_size for _, s in self._missing(entry, spec)]
        return None if any(s is None for s in sizes) else sum(s for s in sizes if s is not None)

    def _run_download(self, job: _Job, entry: Entry, spec: VersionSpec, request: DownloadRequest) -> None:
        assert job.status is not None
        status = job.status
        status.state = DownloadState.running
        status.stage = "downloading"

        try:
            plan = self._missing(entry, spec)
            finished_bytes = 0
            with self._client_factory() as client:
                downloader = Downloader(
                    client,
                    self._endpoints(request),
                    threads=request.threads or self._settings.download_threads,
                    token=self._settings.hf_token,
                    sleep=self._sleep,
                )
                for ref, item in plan:
                    base = finished_bytes

                    def on_progress(p: DownloadProgress, base: int = base, ref: Ref = ref) -> None:
                        status.stage = p.stage
                        status.file = f"{ref.id}/{p.file}"
                        status.bytes_done = base + p.bytes_done
                        status.bytes_per_second = p.bytes_per_second
                        status.source = p.endpoint or status.source

                    self._store.root.mkdir(parents=True, exist_ok=True)
                    downloader.fetch(
                        item,
                        self._store.staging_dir(ref),
                        self._store.partial_root,
                        cancelled=job.cancel.is_set,
                        on_progress=on_progress,
                    )
                    self._store.commit(ref, item)
                    finished_bytes += item.total_size or 0

            self._activate_if_first(entry, spec)
            status.state, status.stage, status.bytes_per_second = DownloadState.succeeded, "done", 0.0
            if status.bytes_total is not None:
                status.bytes_done = status.bytes_total
        except DownloadCancelled:
            status.state, status.stage, status.bytes_per_second = DownloadState.cancelled, "cancelled", 0.0
        except DownloadError as exc:
            status.state, status.error, status.bytes_per_second = DownloadState.failed, str(exc), 0.0
        except Exception as exc:
            logger.exception("download of %s failed unexpectedly", job.target)
            status.state, status.error, status.bytes_per_second = (
                DownloadState.failed,
                f"{type(exc).__name__}: {exc}",
                0.0,
            )

    def _activate_if_first(self, entry: Entry, spec: VersionSpec) -> None:
        """The first installed version of a model becomes the active one."""
        if entry.kind == "model" and self._store.active(entry.id) is None:
            self._store.set_active(entry.id, spec.version)

    # -- import ---------------------------------------------------------------------------------------------

    def import_version(
        self,
        ref: Ref,
        source: Path,
        *,
        move: bool = False,
        on_file: Callable[[str], None] | None = None,
    ) -> ImportResult:
        """Adopt files that are already on disk (say, another tool's model folder) instead of downloading them.

        Every file the catalog lists for ``ref`` must be in ``source`` with the pinned size and sha256. All of
        them are checked first: one wrong file rejects the import and nothing is installed. Then the files go
        through the same staging directory and atomic rename as a download, and the first installed version of
        a model becomes the active one. The files are copied, or with ``move`` taken out of ``source``.
        """
        try:
            entry, spec = self._catalog.resolve(ref)
        except KeyError:
            raise NotFound("VERSION_NOT_FOUND", f"unknown model version {ref}") from None
        if self._busy(ref):
            raise Conflict("DOWNLOAD_IN_PROGRESS", f"{ref} is being downloaded; cancel the download first")
        if self._status_of(ref) is VersionStatus.installed:
            raise Conflict("ALREADY_INSTALLED", f"{ref} is already installed")
        if not source.is_dir():
            raise Unprocessable("IMPORT_SOURCE_NOT_FOUND", f"{source} is not a directory")
        resolved = source.resolve()
        if self._store.root.resolve() in (resolved, *resolved.parents):
            raise Unprocessable(
                "IMPORT_SOURCE_IN_STORE", f"{source} is inside the model store; import from elsewhere"
            )

        problems = find_problems(spec, source, on_file)
        if problems:
            raise Unprocessable(
                "IMPORT_REJECTED",
                f"{source} does not match {ref}, nothing was installed:\n  " + "\n  ".join(problems),
            )

        staging = self._store.staging_dir(ref)
        shutil.rmtree(
            staging, ignore_errors=True
        )  # what an interrupted download left; the import replaces it
        staging.mkdir(parents=True)
        staged = None
        try:
            staged = stage_files(spec, source, staging, move=move)
            self._store.commit(ref, spec)
        except BaseException:
            if staged is not None:
                staged.undo()  # give the user's files back before the half-finished staging directory goes
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staged.finish()
        self._activate_if_first(entry, spec)
        return ImportResult(ref, len(spec.files), sum(f.size or 0 for f in spec.files), move)

    def _forget_old_jobs(self) -> None:
        finished = [
            j
            for j in self._jobs.values()
            if j.status and j.status.state not in (DownloadState.queued, DownloadState.running)
        ]
        for job in finished[: max(0, len(self._jobs) - MAX_REMEMBERED_DOWNLOADS)]:
            del self._jobs[job.id]

    def get_download(self, download_id: str) -> DownloadStatus:
        with self._lock:
            job = self._jobs.get(download_id)
        if job is None:
            raise NotFound("DOWNLOAD_NOT_FOUND", f"unknown download {download_id!r}")
        return job.snapshot()

    def cancel_download(self, download_id: str) -> DownloadStatus:
        with self._lock:
            job = self._jobs.get(download_id)
        if job is None:
            raise NotFound("DOWNLOAD_NOT_FOUND", f"unknown download {download_id!r}")
        job.cancel.set()  # idempotent; the worker notices between chunks
        return job.snapshot()

    def list_downloads(self) -> list[DownloadStatus]:
        with self._lock:
            return [j.snapshot() for j in self._jobs.values()]

    # -- sources --------------------------------------------------------------------------------------------

    def sources(self, *, refresh: bool = False, extra: list[str] | None = None) -> SourcesResponse:
        """Probe the official endpoint, the configured one and the mirrors (cached for a minute)."""
        with self._lock:
            cached = self._sources
        if cached and not refresh and not extra and self._clock() - cached[0] < SOURCES_CACHE_SECONDS:
            return cached[1]
        with self._client_factory() as client:
            result = probe_sources(
                client,
                self._catalog,
                configured=self._settings.hf_endpoint,
                mirrors=self._settings.hf_mirrors,
                extra=extra or (),
            )
        if not extra:
            with self._lock:
                self._sources = (self._clock(), result)
        return result

    # -- catalog ----------------------------------------------------------------------------------------------

    def refresh_catalog(self) -> CatalogRefreshResult:
        """Fetch a newer catalog from ``catalog_url``. A failure keeps the current catalog and says why."""
        url = self._settings.catalog_url
        if not url.startswith("https://"):
            return CatalogRefreshResult(
                refreshed=False, models=len(self._catalog.models()), message="catalog_url must be https"
            )
        try:
            with self._client_factory() as client:
                response = client.get(url, follow_redirects=True, timeout=20.0)
                response.raise_for_status()
            if len(response.content) > MAX_CATALOG_BYTES:
                raise CatalogError("the catalog is unreasonably large")
            data = response.json()
            fetched = parse_catalog(data, require_hashes=True)
            merged = merge_entries(self._catalog.entries.values(), fetched)  # keeps the user's local entries
        except (httpx.HTTPError, ValueError, CatalogError) as exc:
            return CatalogRefreshResult(
                refreshed=False,
                models=len(self._catalog.models()),
                message=f"kept the current catalog: {exc}",
            )

        cache = self._settings.data_dir / "catalog.cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self._catalog = merged
        return CatalogRefreshResult(refreshed=True, models=len(merged.models()), message="catalog updated")

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
