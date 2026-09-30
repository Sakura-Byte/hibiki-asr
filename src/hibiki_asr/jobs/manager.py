"""Runs jobs one at a time in the inference worker, and makes cancelling reliable.

Cancelling is cooperative first: the worker checks a shared flag between VAD windows and between decoded
segments. If a job still has not stopped ``cancel_grace_seconds`` after the request (a very long decode, a hung
GPU call), the worker process is killed and a fresh one is started for the next job. That is slower for the next
job, since the model has to be loaded again, but a cancel is never ignored.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..models.manager import NotFound, ResolvedModel
from ..pipeline.types import Segment as PipelineSegment
from ..pipeline.writers import to_vtt
from ..settings import Settings
from ..worker.process import WorkerFactory, WorkerHandle
from ..worker.protocol import Progress, RunCancelled, RunFailed, RunRequest, RunResult
from ..worker.session import vad_settings_to_dict
from .schema import (
    TERMINAL_STATES,
    JobError,
    JobResult,
    JobState,
    JobStatus,
    ModelUsed,
    RuntimeInfo,
    Segment,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.1
HOUSEKEEPING_SECONDS = 30.0


@dataclass
class _Job:
    id: str
    model: ResolvedModel
    task: str
    language: str
    audio_path: Path
    variant: str | None
    created_at: float
    state: JobState = JobState.queued
    stage: str = "queued"
    progress: float = 0.0
    message: str = "queued"
    updated_at: float = 0.0
    finished_at: float | None = None
    cancel_requested: bool = False
    result: JobResult | None = None
    error: JobError | None = None
    done: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(
        self,
        settings: Settings,
        worker_factory: WorkerFactory,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._factory = worker_factory
        self._clock = clock
        self._wall = wall_clock

        self._cond = threading.Condition()
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[str] = deque()
        self._stopping = False
        self._worker: WorkerHandle | None = None
        self._last_activity = clock()

        self._thread = threading.Thread(target=self._loop, name="job-dispatcher", daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------------------------------

    def submit(
        self,
        model: ResolvedModel,
        task: str,
        language: str,
        audio_path: Path,
        *,
        variant: str | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        """Queue a job. ``job_id`` lets the caller name the upload directory after the job before it exists."""
        now = self._wall()
        job = _Job(
            id=job_id or uuid.uuid4().hex,
            model=model,
            task=task,
            language=language,
            audio_path=audio_path,
            variant=variant,
            created_at=now,
            updated_at=now,
        )
        with self._cond:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._cond.notify_all()
            return self._snapshot(job)

    def get(self, job_id: str) -> JobStatus:
        with self._cond:
            return self._snapshot(self._find(job_id))

    def cancel(self, job_id: str) -> JobStatus:
        """Idempotent. A queued job is cancelled at once; a running one is asked to stop."""
        with self._cond:
            job = self._find(job_id)
            if job.state in TERMINAL_STATES:
                return self._snapshot(job)
            job.cancel_requested = True
            if job.state is JobState.queued:
                self._queue.remove(job.id)
                self._finish(job, JobState.cancelled, message="cancelled before it started")
                self._discard_audio(job)
            else:
                job.state = JobState.cancelling
                job.message = "cancelling"
                job.updated_at = self._wall()
            self._cond.notify_all()
            return self._snapshot(job)

    def wait(self, job_id: str, timeout: float | None = None) -> JobStatus:
        """Block until the job reaches a final state (used by tests and the CLI)."""
        with self._cond:
            job = self._find(job_id)
        job.done.wait(timeout)
        return self.get(job_id)

    def worker_running(self) -> bool:
        with self._cond:
            return self._worker is not None and self._worker.is_alive()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            for job in list(self._jobs.values()):
                if job.state in TERMINAL_STATES:
                    continue
                job.cancel_requested = True
                if job.state is JobState.queued:
                    self._queue.remove(job.id)
                    self._finish(job, JobState.cancelled, message="the engine is shutting down")
                    self._discard_audio(job)
            self._cond.notify_all()
        self._thread.join(timeout=15)
        self._drop_worker(graceful=True)

    # -- state helpers (callers hold the lock unless noted) -----------------------------------------------------

    def _find(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFound(
                "JOB_NOT_FOUND",
                f"unknown job {job_id!r} (jobs are forgotten after a while, and when the engine restarts)",
            )
        return job

    def _snapshot(self, job: _Job) -> JobStatus:
        position = (
            self._queue.index(job.id) + 1 if job.state is JobState.queued and job.id in self._queue else None
        )
        return JobStatus(
            id=job.id,
            state=job.state,
            stage=job.stage,
            progress=job.progress,
            message=job.message,
            queue_position=position,
            created_at=job.created_at,
            updated_at=job.updated_at,
            result=job.result,
            error=job.error,
        )

    def _finish(
        self,
        job: _Job,
        state: JobState,
        *,
        message: str,
        result: JobResult | None = None,
        error: JobError | None = None,
    ) -> None:
        job.state, job.message, job.result, job.error = state, message, result, error
        job.stage = "done" if state is JobState.succeeded else job.stage
        job.progress = 1.0 if state is JobState.succeeded else job.progress
        job.updated_at = job.finished_at = self._wall()
        job.done.set()

    @staticmethod
    def _discard_audio(job: _Job) -> None:
        try:
            job.audio_path.unlink(missing_ok=True)
            parent = job.audio_path.parent
            if parent.name == job.id:
                shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            logger.warning("could not remove the uploaded audio of job %s", job.id)

    # -- dispatcher --------------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._cond:
                job = self._next_job()
                if job is None:
                    if self._stopping:
                        return
                    continue
            self._execute(job)

    def _next_job(self) -> _Job | None:
        """Wait for work; while idle, unload the model and forget old jobs. Called with the lock held."""
        if not self._queue and not self._stopping:
            timeout = HOUSEKEEPING_SECONDS
            idle_limit = self._settings.idle_unload_seconds
            if idle_limit and self._worker is not None:
                timeout = max(0.05, min(timeout, idle_limit - (self._clock() - self._last_activity)))
            self._cond.wait(timeout)
            self._housekeeping()
        if self._stopping:
            return None
        if not self._queue:
            return None
        job = self._jobs[self._queue.popleft()]
        job.state, job.stage, job.message = JobState.running, "loading", "starting"
        job.updated_at = self._wall()
        return job

    def _housekeeping(self) -> None:
        idle_limit = self._settings.idle_unload_seconds
        if (
            idle_limit
            and self._worker is not None
            and not self._queue
            and self._clock() - self._last_activity >= idle_limit
        ):
            logger.info("no work for %ds: unloading the model to free memory", idle_limit)
            worker, self._worker = self._worker, None
            threading.Thread(target=lambda: worker.shutdown(10.0), daemon=True).start()

        cutoff = self._wall() - self._settings.job_ttl_seconds
        for job_id in [
            j.id for j in self._jobs.values() if j.finished_at is not None and j.finished_at < cutoff
        ]:
            del self._jobs[job_id]

    def _ensure_worker(self) -> WorkerHandle:
        with self._cond:
            if self._worker is not None and not self._worker.is_alive():
                self._worker = None
            if self._worker is None:
                self._worker = self._factory()
            return self._worker

    def _drop_worker(self, *, graceful: bool) -> None:
        with self._cond:
            worker, self._worker = self._worker, None
        if worker is None:
            return
        if graceful:
            worker.shutdown(5.0)
        else:
            worker.kill()

    def _request_for(self, job: _Job) -> RunRequest:
        s = self._settings
        return RunRequest(
            job_id=job.id,
            audio_path=str(job.audio_path),
            model_ref=str(job.model.ref),
            model_dir=str(job.model.model_dir),
            components={k: str(v) for k, v in job.model.components.items()},
            language=job.language,
            task=job.task,
            device=s.device,
            compute_type=s.compute_type,
            allow_cpu_fallback=s.allow_cpu_fallback,
            vad=vad_settings_to_dict(s.vad),
            merge=s.merge.model_dump(),
            generation=dict(s.generation),
            chunk_target_s=s.chunk_target_s,
            vad_threads=s.vad.threads,
            cpu_threads=s.cpu_threads,
            variant=job.variant,
        )

    def _execute(self, job: _Job) -> None:
        try:
            self._run_on_worker(job)
        except Exception as exc:
            logger.exception("job %s failed unexpectedly", job.id)
            with self._cond:
                if job.state not in TERMINAL_STATES:
                    self._finish(
                        job,
                        JobState.failed,
                        message=str(exc),
                        error=JobError(code="INTERNAL_ERROR", message=str(exc)),
                    )
        finally:
            self._discard_audio(job)
            with self._cond:
                self._last_activity = self._clock()
                if job.state not in TERMINAL_STATES:  # defensive: never leave a job dangling
                    self._finish(
                        job,
                        JobState.failed,
                        message="stopped",
                        error=JobError(code="INTERNAL_ERROR", message="job stopped unexpectedly"),
                    )

    def _run_on_worker(self, job: _Job) -> None:
        try:
            worker = self._ensure_worker()
        except Exception as exc:
            with self._cond:
                self._finish(
                    job,
                    JobState.failed,
                    message="cannot start the inference worker",
                    error=JobError(
                        code="WORKER_START_FAILED",
                        message=f"Could not start the inference worker: {type(exc).__name__}: {exc}",
                        hint="Run `hibiki-asr doctor`; the inference runtime may not be installed (`hibiki-asr setup`).",
                    ),
                )
            return

        worker.clear_cancel()
        worker.send(self._request_for(job))
        cancel_deadline: float | None = None

        while True:
            with self._cond:
                cancelling = job.cancel_requested
            if cancelling and cancel_deadline is None:
                worker.request_cancel()
                cancel_deadline = self._clock() + self._settings.cancel_grace_seconds

            message = worker.poll(POLL_SECONDS)

            if message is None:
                if not worker.is_alive():
                    self._on_crash(job, worker)
                    return
                if cancel_deadline is not None and self._clock() > cancel_deadline:
                    logger.warning(
                        "job %s ignored the cancel request for %.0fs: killing the worker",
                        job.id,
                        self._settings.cancel_grace_seconds,
                    )
                    self._drop_worker(graceful=False)
                    with self._cond:
                        self._finish(
                            job, JobState.cancelled, message="cancelled (the worker was restarted to stop it)"
                        )
                    return
                continue

            if isinstance(message, Progress):
                with self._cond:
                    job.stage, job.message = message.stage, message.message
                    job.progress = max(job.progress, min(message.fraction, 0.99))
                    job.updated_at = self._wall()
                continue

            with self._cond:
                if isinstance(message, RunResult):
                    if job.cancel_requested:
                        self._finish(
                            job, JobState.cancelled, message="cancelled"
                        )  # the client asked to stop: drop the result
                    else:
                        self._finish(
                            job, JobState.succeeded, message="completed", result=self._to_result(message)
                        )
                elif isinstance(message, RunCancelled):
                    self._finish(job, JobState.cancelled, message="cancelled")
                elif isinstance(message, RunFailed):
                    self._finish(
                        job,
                        JobState.failed,
                        message=message.message,
                        error=JobError(
                            code=message.code,
                            message=message.message,
                            hint=message.hint,
                            findings=message.findings,
                        ),
                    )
                else:  # pragma: no cover - protocol violation
                    continue
            return

    def _on_crash(self, job: _Job, worker: WorkerHandle) -> None:
        code = worker.exit_code()
        how = f"signal {-code}" if code is not None and code < 0 else f"exit code {code}"
        with self._cond:
            self._worker = None
            if job.cancel_requested:
                self._finish(job, JobState.cancelled, message="cancelled")
                return
            self._finish(
                job,
                JobState.failed,
                message=f"the inference worker crashed ({how})",
                error=JobError(
                    code="WORKER_CRASHED",
                    message=f"The inference worker crashed ({how}) while processing the audio.",
                    hint=(
                        "This is usually a GPU driver or runtime problem, or running out of memory. Run `hibiki-asr doctor`. "
                        "As a workaround set HIBIKI_ASR_DEVICE=cpu. The next job starts a fresh worker."
                    ),
                ),
            )

    @staticmethod
    def _to_result(r: RunResult) -> JobResult:
        segments = [Segment(start_ms=s, end_ms=e, text=t) for s, e, t in r.segments]
        model_id, _, version = r.model_ref.partition("@")
        return JobResult(
            segments=segments,
            vtt=to_vtt([PipelineSegment(s.start_ms, s.end_ms, s.text) for s in segments]),
            duration_ms=round(r.duration_s * 1000),
            speech_ms=round(r.speech_s * 1000),
            model=ModelUsed(id=model_id, version=version),
            runtime=RuntimeInfo(
                device=r.device,
                compute_type=r.compute_type,
                vad_device=r.vad_device,
                degraded=r.degraded,
                findings=r.findings,
            ),
        )
