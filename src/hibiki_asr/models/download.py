"""Downloading a pinned model version.

* Large files are fetched as fixed-size blocks by several threads at once (HTTP Range), which is what
  makes multi-GB models tolerable on a slow or throttled link. A small side file records the finished
  blocks, so an interrupted download resumes where it stopped.
* Every request goes to the first endpoint and falls back through the mirrors, so a blocked or slow
  Hugging Face does not stop the download.
* Every file is verified against the size and sha256 pinned in the catalog before it is installed.
* It can be cancelled at any point; partial data is kept for the next attempt.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from .catalog import FileSpec, VersionSpec

logger = logging.getLogger(__name__)

CHUNK = 1024 * 1024
BLOCK_SIZE = 16 * 1024 * 1024
PARALLEL_MIN_SIZE = 64 * 1024 * 1024  # smaller files are not worth splitting
DISK_MARGIN_BYTES = 256 * 1024 * 1024
ATTEMPTS_PER_ENDPOINT = 3


class DownloadError(RuntimeError):
    """The download cannot complete; the message says why and is safe to show to a user."""


class DownloadCancelled(Exception):  # noqa: N818
    """The download was cancelled by the user. Partial files are kept so it can resume."""


class InsufficientDiskSpace(DownloadError):
    pass


class ChecksumMismatch(DownloadError):
    pass


class _NoRangeSupport(Exception):  # noqa: N818
    """The server ignored a Range request, so the file cannot be fetched in parallel."""


@dataclass(frozen=True)
class DownloadProgress:
    stage: str  # "downloading" | "verifying"
    bytes_done: int
    bytes_total: int | None
    file: str
    bytes_per_second: float
    endpoint: str | None = None  # the source the data is currently coming from


ProgressFn = Callable[[DownloadProgress], None]


def sha256_of(path: Path, on_chunk: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
            if on_chunk:
                on_chunk(len(chunk))
    return digest.hexdigest()


class _Rate:
    """Bytes per second over roughly the last few seconds."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._samples: list[tuple[float, int]] = []

    def update(self, total_bytes: int) -> float:
        now = self._clock()
        self._samples.append((now, total_bytes))
        self._samples = [s for s in self._samples if now - s[0] <= 5.0] or self._samples[-1:]
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0


class Downloader:
    def __init__(
        self,
        client: httpx.Client,
        endpoints: Sequence[str],
        *,
        threads: int = 4,
        token: str | None = None,
        block_size: int = BLOCK_SIZE,
        parallel_min_size: int = PARALLEL_MIN_SIZE,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint is required")
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self._client = client
        self._endpoints = [e.rstrip("/") for e in endpoints]
        self._threads = threads
        self._token = token
        self._block = block_size
        self._parallel_min = parallel_min_size
        self._sleep = sleep
        self._clock = clock
        self._active_endpoint: str | None = None
        self._dead: set[str] = set()  # endpoints that exhausted their retries; tried last from then on
        self._dead_lock = threading.Lock()

    # -- public ---------------------------------------------------------------------------------

    def fetch(
        self,
        version: VersionSpec,
        dest: Path,
        partial_root: Path,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        on_progress: ProgressFn = lambda _p: None,
    ) -> None:
        """Bring every file of ``version`` into ``dest`` (files already there with the right size are kept)."""
        dest.mkdir(parents=True, exist_ok=True)
        partial_dir = partial_root / f"{version.repo.replace('/', '__')}@{version.revision}"
        partial_dir.mkdir(parents=True, exist_ok=True)

        pending = [f for f in version.files if not self._is_present(dest / f.path, f)]
        self._require_disk_space(pending, partial_dir, dest)

        total = version.total_size
        done_before = sum(f.size or 0 for f in version.files if f not in pending)
        rate = _Rate(self._clock)
        lock = threading.Lock()
        state = {"done": done_before}

        for spec in pending:
            part = partial_dir / (spec.path.replace("/", "__") + ".part")
            base = state["done"]

            def on_bytes(file_bytes: int, name: str = spec.path, base: int = base) -> None:
                with lock:
                    state["done"] = base + file_bytes
                    on_progress(
                        DownloadProgress(
                            "downloading", state["done"], total, name, rate.update(state["done"]), self._active_endpoint
                        )
                    )

            self._download_file(version, spec, part, cancelled, on_bytes)
            if cancelled():
                raise DownloadCancelled

            # Hashing must not move the bar backwards: the whole file is already counted as downloaded.
            self._verify(
                spec,
                part,
                lambda _n, name=spec.path: on_progress(
                    DownloadProgress("verifying", state["done"], total, name, 0.0, self._active_endpoint)
                ),
            )
            target = dest / spec.path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, target)

    # -- planning -------------------------------------------------------------------------------

    @staticmethod
    def _is_present(target: Path, spec: FileSpec) -> bool:
        return target.is_file() and (spec.size is None or target.stat().st_size == spec.size)

    @staticmethod
    def _require_disk_space(pending: Sequence[FileSpec], partial_dir: Path, dest: Path) -> None:
        if any(f.size is None for f in pending):
            return
        already = 0
        for f in pending:
            for suffix in (".part", ".blocks"):
                candidate = partial_dir / (f.path.replace("/", "__") + suffix)
                if candidate.exists():
                    already += candidate.stat().st_size
        needed = sum(f.size or 0 for f in pending) - already + DISK_MARGIN_BYTES
        free = shutil.disk_usage(dest).free
        if needed > free:
            raise InsufficientDiskSpace(
                f"not enough disk space in {dest}: need about {needed // 2**20} MB, {free // 2**20} MB free. "
                "Free some space or point `models_dir` at a larger disk."
            )

    # -- one file -------------------------------------------------------------------------------

    def _download_file(
        self,
        version: VersionSpec,
        spec: FileSpec,
        part: Path,
        cancelled: Callable[[], bool],
        on_bytes: Callable[[int], None],
    ) -> None:
        if spec.size is not None and part.exists() and part.stat().st_size == spec.size:
            on_bytes(spec.size)
            return  # a complete .part from an interrupted run; verification follows

        if self._threads > 1 and spec.size is not None and spec.size >= self._parallel_min:
            try:
                self._download_blocks(version, spec, part, cancelled, on_bytes)
                return
            except _NoRangeSupport:
                logger.info("%s: server does not support Range; falling back to a single stream", spec.path)
                for leftover in (part.with_name(part.name.removesuffix(".part") + ".blocks"),):
                    leftover.unlink(missing_ok=True)
                    Path(str(leftover) + ".done").unlink(missing_ok=True)

        def attempt(url: str, endpoint: str) -> None:
            self._stream_to(url, endpoint, spec, part, cancelled, on_bytes)

        self._with_endpoints(version, spec, cancelled, attempt)

    def _ordered_endpoints(self) -> list[str]:
        """Endpoints that have not failed yet come first, so a dead one is not retried for every file and block."""
        with self._dead_lock:
            return [e for e in self._endpoints if e not in self._dead] + [e for e in self._endpoints if e in self._dead]

    def _demote(self, endpoint: str) -> None:
        with self._dead_lock:
            if endpoint not in self._dead:
                logger.warning("%s failed; trying the other sources first from now on", endpoint)
            self._dead.add(endpoint)

    def _url(self, version: VersionSpec, endpoint: str, path: str) -> str:
        return f"{endpoint}/{version.repo}/resolve/{version.revision}/{quote(path)}"

    def _headers(self, endpoint: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self._token and endpoint == self._endpoints[0]:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _with_endpoints(
        self,
        version: VersionSpec,
        spec: FileSpec,
        cancelled: Callable[[], bool],
        attempt: Callable[[str, str], None],
    ) -> None:
        """Run ``attempt(url, endpoint)`` on the first endpoint, then the mirrors, retrying transient failures."""
        errors: list[str] = []
        for endpoint in self._ordered_endpoints():
            url = self._url(version, endpoint, spec.path)
            for n in range(ATTEMPTS_PER_ENDPOINT):
                if cancelled():
                    raise DownloadCancelled
                try:
                    attempt(url, endpoint)
                    return
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    errors.append(f"{endpoint}: {_describe(exc)}")
                    logger.warning("download of %s from %s failed (attempt %d): %s", spec.path, endpoint, n + 1, exc)
                    if _not_worth_retrying(exc):
                        break  # retrying the same endpoint will not help; try the next mirror
                    self._sleep(min(2**n, 8))
            self._demote(endpoint)
        raise DownloadError(
            f"could not download {spec.path} ({version.repo}@{version.revision[:8]}): {errors[-1] if errors else 'no endpoint'}. "
            "Check the network connection or choose another download source (`hf_endpoint` / `hf_mirrors`; HTTPS_PROXY is honoured)."
        )

    # -- single stream (small files, or servers without Range) ------------------------------------

    def _stream_to(
        self,
        url: str,
        endpoint: str,
        spec: FileSpec,
        part: Path,
        cancelled: Callable[[], bool],
        on_bytes: Callable[[int], None],
    ) -> None:
        offset = part.stat().st_size if part.exists() else 0
        if spec.size is not None and offset > spec.size:
            part.unlink()
            offset = 0
        if spec.size is not None and offset == spec.size:
            on_bytes(offset)
            return  # a complete .part from an interrupted run; verification follows

        headers = self._headers(endpoint, {"Range": f"bytes={offset}-"} if offset else None)
        with self._client.stream("GET", url, headers=headers, follow_redirects=True) as response:
            if response.status_code == 416:  # our offset is past the end: start over
                part.unlink(missing_ok=True)
                raise httpx.TransportError("server rejected the resume offset")
            response.raise_for_status()
            self._active_endpoint = endpoint
            resumed = offset > 0 and response.status_code == 206
            if offset and not resumed:
                offset = 0  # the server ignored Range and is sending the whole file
            written = offset
            on_bytes(written)
            with open(part, "ab" if resumed else "wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    if cancelled():
                        raise DownloadCancelled
                    handle.write(chunk)
                    written += len(chunk)
                    on_bytes(written)

    # -- parallel blocks (large files) ------------------------------------------------------------

    def _download_blocks(
        self,
        version: VersionSpec,
        spec: FileSpec,
        part: Path,
        cancelled: Callable[[], bool],
        on_bytes: Callable[[int], None],
    ) -> None:
        size = spec.size
        assert size is not None
        blocks_file = part.with_name(part.name.removesuffix(".part") + ".blocks")
        done_file = Path(str(blocks_file) + ".done")

        count = -(-size // self._block)
        finished = self._read_finished(done_file, count)
        if not blocks_file.exists() or blocks_file.stat().st_size != size:
            finished.clear()
            done_file.unlink(missing_ok=True)
            with open(blocks_file, "wb") as handle:
                handle.truncate(size)

        def block_len(index: int) -> int:
            return min(self._block, size - index * self._block)

        lock = threading.Lock()
        completed_bytes = sum(block_len(i) for i in finished)
        inflight: dict[int, int] = {}
        stop = threading.Event()
        on_bytes(completed_bytes)

        def report() -> None:
            on_bytes(completed_bytes + sum(inflight.values()))

        def fetch_block(index: int) -> None:
            nonlocal completed_bytes
            start = index * self._block
            length = block_len(index)

            def attempt(url: str, endpoint: str) -> None:
                if stop.is_set() or cancelled():
                    raise DownloadCancelled
                headers = self._headers(endpoint, {"Range": f"bytes={start}-{start + length - 1}"})
                with self._client.stream("GET", url, headers=headers, follow_redirects=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise _NoRangeSupport
                    self._active_endpoint = endpoint
                    received = 0
                    with open(blocks_file, "r+b") as handle:
                        handle.seek(start)
                        for chunk in response.iter_bytes(CHUNK):
                            if stop.is_set() or cancelled():
                                raise DownloadCancelled
                            handle.write(chunk[: length - received])
                            received += len(chunk)
                            with lock:
                                inflight[index] = min(received, length)
                                report()
                            if received >= length:
                                break
                if received < length:
                    raise httpx.TransportError(f"block {index} ended after {received} of {length} bytes")

            self._with_endpoints(version, spec, lambda: stop.is_set() or cancelled(), attempt)
            with lock:
                inflight.pop(index, None)
                completed_bytes += length
                finished.add(index)
                with open(done_file, "a", encoding="ascii") as log:
                    log.write(f"{index}\n")
                report()

        todo = [i for i in range(count) if i not in finished]
        with ThreadPoolExecutor(max_workers=min(self._threads, max(1, len(todo))), thread_name_prefix="dl") as pool:
            futures = [pool.submit(fetch_block, i) for i in todo]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        if cancelled():
            raise DownloadCancelled

        os.replace(blocks_file, part)
        done_file.unlink(missing_ok=True)

    @staticmethod
    def _read_finished(done_file: Path, count: int) -> set[int]:
        try:
            lines = done_file.read_text(encoding="ascii").split()
        except (OSError, ValueError):
            return set()
        return {int(x) for x in lines if x.isdigit() and int(x) < count}

    # -- verification -----------------------------------------------------------------------------

    @staticmethod
    def _verify(spec: FileSpec, part: Path, on_bytes: Callable[[int], None]) -> None:
        actual = part.stat().st_size
        if spec.size is not None and actual != spec.size:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{spec.path}: expected {spec.size} bytes, got {actual}; the download was discarded, retry it")
        if spec.sha256 is None:
            return
        done = 0

        def tick(n: int) -> None:
            nonlocal done
            done += n
            on_bytes(done)

        if sha256_of(part, tick) != spec.sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{spec.path}: sha256 does not match the catalog; the download was discarded, retry it")


def _not_worth_retrying(exc: Exception) -> bool:
    """A missing/forbidden file or an unreachable host will not fix itself within a few seconds."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (401, 403, 404)
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"
