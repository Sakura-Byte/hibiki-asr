"""Job manager: queueing, progress, cooperative cancel, kill-after-grace, crashes, idle unload."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hibiki_asr.diagnostics.schema import Finding, Severity
from hibiki_asr.jobs.manager import JobManager
from hibiki_asr.jobs.schema import JobState
from hibiki_asr.models.catalog import Entry, Ref, VersionSpec
from hibiki_asr.models.manager import NotFound, ResolvedModel
from hibiki_asr.settings import Settings
from hibiki_asr.worker.protocol import Progress, RunCancelled, RunFailed, RunRequest, RunResult, Shutdown

Behavior = Callable[["FakeWorker", RunRequest], None]


class FakeWorker:
    """A worker that runs in a thread and is scripted by a ``behavior`` function."""

    def __init__(self, behavior: Behavior) -> None:
        self.behavior = behavior
        self.inbox: queue.Queue[object] = queue.Queue()
        self.cancel = threading.Event()
        self.alive = True
        self.killed = False
        self.shut_down = False
        self.code: int | None = None
        self.requests: list[RunRequest] = []

    # WorkerHandle
    def send(self, message: object) -> None:
        if isinstance(message, Shutdown):
            self.alive = False
            self.shut_down = True
        elif isinstance(message, RunRequest):
            self.requests.append(message)
            threading.Thread(target=self.behavior, args=(self, message), daemon=True).start()

    def poll(self, timeout: float) -> object | None:
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        return self.alive

    def exit_code(self) -> int | None:
        return self.code

    def request_cancel(self) -> None:
        self.cancel.set()

    def clear_cancel(self) -> None:
        self.cancel.clear()

    def kill(self) -> None:
        self.killed, self.alive, self.code = True, False, -9

    def shutdown(self, timeout: float) -> None:
        self.alive = False
        self.shut_down = True

    # helpers for behaviors
    def emit(self, message: object) -> None:
        self.inbox.put(message)

    def crash(self, code: int = -11) -> None:
        self.alive, self.code = False, code

    def wait_for_cancel(self, seconds: float) -> bool:
        return self.cancel.wait(seconds)


def result(req: RunRequest, *, degraded: bool = False) -> RunResult:
    return RunResult(
        segments=[(0, 1500, "こんにちは"), (2000, 3000, "さようなら")],
        duration_s=12.5,
        speech_s=2.5,
        device="cpu" if degraded else "cuda",
        compute_type="int8" if degraded else "float16",
        vad_device="cpu",
        degraded=degraded,
        model_ref=req.model_ref,
        findings=[Finding(code="CPU_ONLY", severity=Severity.info, message="Running on the CPU.")]
        if degraded
        else [],
    )


def ok(worker: FakeWorker, req: RunRequest) -> None:
    worker.emit(Progress("loading", 0.0, "loading the model"))
    worker.emit(Progress("transcribing", 0.5, "half way"))
    worker.emit(result(req))


def cooperative(worker: FakeWorker, req: RunRequest) -> None:
    worker.emit(Progress("transcribing", 0.2, "working"))
    if worker.wait_for_cancel(5):
        worker.emit(RunCancelled())
    else:
        worker.emit(result(req))


def stubborn(worker: FakeWorker, req: RunRequest) -> None:
    worker.emit(Progress("transcribing", 0.2, "working"))
    time.sleep(30)  # never answers; only killing the worker stops it


class Harness:
    def __init__(self, tmp_path: Path, behaviors: list[Behavior], **settings) -> None:
        self.tmp = tmp_path
        self.workers: list[FakeWorker] = []
        self._behaviors = iter(behaviors)
        self._last: Behavior = behaviors[-1]
        self.settings = Settings(data_dir=tmp_path / "data", **settings)
        self.jobs = JobManager(self.settings, self._make)

    def _make(self) -> FakeWorker:
        worker = FakeWorker(next(self._behaviors, self._last))
        self.workers.append(worker)
        return worker

    def submit(self, name: str = "a") -> str:
        job_dir = self.tmp / f"job-{name}"
        job_dir.mkdir(exist_ok=True)
        audio = job_dir / "audio.wav"
        audio.write_bytes(b"RIFF")
        model = ResolvedModel(
            ref=Ref("tiny", "v1"),
            entry=Entry(
                "tiny",
                "model",
                "Tiny",
                (),
                task="translate",
                source_languages=("ja",),
                output_languages=("zh",),
            ),
            version=VersionSpec("v1", "acme/tiny", "a" * 40, ()),
            model_dir=self.tmp / "model",
            components={"vad-asr": self.tmp / "vad", "whisper-base-fe": self.tmp / "fe"},
        )
        self.audio = audio
        return self.jobs.submit(model, "translate", "ja", audio, variant="cpu").id

    def wait(self, job_id: str, timeout: float = 5.0):
        status = self.jobs.wait(job_id, timeout)
        assert status.state in (JobState.succeeded, JobState.failed, JobState.cancelled), (
            f"still {status.state}"
        )
        return status

    def wait_until(self, condition: Callable[[], bool], timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.01)
        raise AssertionError("condition not met in time")


@pytest.fixture
def harness(tmp_path: Path):
    made: list[Harness] = []

    def build(*behaviors: Behavior, **settings) -> Harness:
        h = Harness(tmp_path, list(behaviors), **settings)
        made.append(h)
        return h

    yield build
    for h in made:
        h.jobs.stop()


# --- success ----------------------------------------------------------------------------------------


def test_a_job_runs_and_reports_its_result(harness) -> None:
    h = harness(ok)
    job_id = h.submit()
    status = h.wait(job_id)

    assert status.state is JobState.succeeded and status.progress == 1.0 and status.stage == "done"
    assert status.queue_position is None and status.error is None
    first = status.result.segments[0]
    assert (first.start_ms, first.end_ms, first.text) == (0, 1500, "こんにちは")
    assert status.result.vtt.startswith("WEBVTT\n\n") and "00:00:00.000 --> 00:00:01.500" in status.result.vtt
    assert (status.result.duration_ms, status.result.speech_ms) == (12_500, 2_500)
    assert (status.result.model.id, status.result.model.version) == ("tiny", "v1")
    assert (status.result.runtime.device, status.result.runtime.degraded) == ("cuda", False)
    assert not h.audio.exists()  # the uploaded audio is deleted as soon as the job ends


def test_the_worker_is_asked_with_the_engine_settings(harness) -> None:
    h = harness(ok, device="cpu", compute_type="int8", allow_cpu_fallback=False, chunk_target_s=20)
    h.wait(h.submit())
    req = h.workers[0].requests[0]
    assert (req.device, req.compute_type, req.allow_cpu_fallback, req.chunk_target_s) == (
        "cpu",
        "int8",
        False,
        20,
    )
    assert (req.task, req.language, req.model_ref, req.variant) == ("translate", "ja", "tiny@v1", "cpu")
    assert "threads" not in req.vad and req.vad["threshold"] == 0.5 and req.vad_threads == 0


def test_a_degraded_result_carries_its_findings(harness) -> None:
    h = harness(lambda w, r: w.emit(result(r, degraded=True)))
    status = h.wait(h.submit())
    assert status.result.runtime.degraded and status.result.runtime.findings[0].code == "CPU_ONLY"


def test_progress_is_monotonic_and_visible_while_running(harness) -> None:
    release = threading.Event()

    def slow(worker: FakeWorker, req: RunRequest) -> None:
        worker.emit(Progress("transcribing", 0.6, "chunk 3/5"))
        worker.emit(Progress("transcribing", 0.4, "late duplicate"))
        release.wait(5)
        worker.emit(result(req))

    h = harness(slow)
    job_id = h.submit()
    h.wait_until(lambda: h.jobs.get(job_id).progress >= 0.6)
    status = h.jobs.get(job_id)
    assert (
        status.state is JobState.running
        and status.progress == pytest.approx(0.6)
        and status.stage == "transcribing"
    )
    release.set()
    h.wait(job_id)


def test_jobs_run_in_order_and_report_their_queue_position(harness) -> None:
    release = threading.Event()

    def gated(worker: FakeWorker, req: RunRequest) -> None:
        release.wait(5)
        worker.emit(result(req))

    h = harness(gated)
    first, second, third = h.submit("1"), h.submit("2"), h.submit("3")
    h.wait_until(lambda: h.jobs.get(first).state is JobState.running)
    assert [h.jobs.get(j).queue_position for j in (first, second, third)] == [None, 1, 2]
    release.set()
    assert [h.wait(j).state for j in (first, second, third)] == [JobState.succeeded] * 3
    assert len(h.workers) == 1  # the model stayed loaded between jobs


# --- cancel -------------------------------------------------------------------------------------------


def test_cancelling_a_queued_job_never_starts_it(harness) -> None:
    release = threading.Event()
    h = harness(lambda w, r: (release.wait(5), w.emit(result(r))))
    running, queued = h.submit("1"), h.submit("2")
    h.wait_until(lambda: h.jobs.get(running).state is JobState.running)

    status = h.jobs.cancel(queued)
    assert status.state is JobState.cancelled and status.queue_position is None
    assert h.jobs.cancel(queued).state is JobState.cancelled  # idempotent
    release.set()
    h.wait(running)
    assert len(h.workers[0].requests) == 1  # the cancelled job was never sent to the worker


def test_a_cooperative_cancel_stops_the_job_and_keeps_the_worker(harness) -> None:
    h = harness(cooperative, ok)
    job_id = h.submit("1")
    h.wait_until(lambda: h.jobs.get(job_id).progress > 0)

    assert h.jobs.cancel(job_id).state is JobState.cancelling
    status = h.wait(job_id)
    assert status.state is JobState.cancelled and status.result is None

    # The same worker (with its loaded model) serves the next job.
    h.workers[0].behavior = ok
    assert h.wait(h.submit("2")).state is JobState.succeeded
    assert len(h.workers) == 1 and not h.workers[0].killed


def test_a_worker_that_ignores_cancel_is_killed_after_the_grace_period(harness) -> None:
    h = harness(stubborn, ok, cancel_grace_seconds=0.3)
    job_id = h.submit("1")
    h.wait_until(lambda: h.jobs.get(job_id).progress > 0)

    started = time.monotonic()
    h.jobs.cancel(job_id)
    status = h.wait(job_id)

    assert status.state is JobState.cancelled and "restarted" in status.message
    assert 0.25 < time.monotonic() - started < 3
    assert h.workers[0].killed

    assert h.wait(h.submit("2")).state is JobState.succeeded  # a fresh worker takes over
    assert len(h.workers) == 2 and not h.workers[1].killed


def test_a_result_that_arrives_after_the_cancel_request_is_dropped(harness) -> None:
    proceed = threading.Event()

    def late(worker: FakeWorker, req: RunRequest) -> None:
        worker.emit(Progress("transcribing", 0.5, "almost"))
        proceed.wait(5)
        worker.emit(result(req))  # finishes without ever looking at the cancel flag

    h = harness(late)
    job_id = h.submit()
    h.wait_until(lambda: h.jobs.get(job_id).progress > 0)
    h.jobs.cancel(job_id)
    proceed.set()
    status = h.wait(job_id)
    assert status.state is JobState.cancelled and status.result is None


def test_cancelling_a_finished_job_changes_nothing(harness) -> None:
    h = harness(ok)
    job_id = h.submit()
    h.wait(job_id)
    assert h.jobs.cancel(job_id).state is JobState.succeeded


# --- failures ---------------------------------------------------------------------------------------------


def test_a_reported_failure_keeps_its_code_hint_and_findings(harness) -> None:
    finding = Finding(code="CT2_IMPORT_FAILED", severity=Severity.error, message="boom", hint="reinstall")

    def fail(worker: FakeWorker, req: RunRequest) -> None:
        worker.emit(RunFailed("MODEL_LOAD_FAILED", "Could not load tiny@v1", "verify the files", [finding]))

    h = harness(fail)
    status = h.wait(h.submit())
    assert status.state is JobState.failed and status.result is None
    assert (status.error.code, status.error.hint) == ("MODEL_LOAD_FAILED", "verify the files")
    assert status.error.findings[0].code == "CT2_IMPORT_FAILED"


def test_a_crashing_worker_fails_the_job_and_the_next_job_gets_a_new_worker(harness) -> None:
    def crash(worker: FakeWorker, req: RunRequest) -> None:
        worker.emit(Progress("transcribing", 0.1, "working"))
        time.sleep(0.1)
        worker.crash(-11)

    h = harness(crash, ok)
    status = h.wait(h.submit("1"))
    assert status.state is JobState.failed and status.error.code == "WORKER_CRASHED"
    assert "signal 11" in status.error.message and "HIBIKI_ASR_DEVICE=cpu" in status.error.hint

    assert h.wait(h.submit("2")).state is JobState.succeeded
    assert len(h.workers) == 2


def test_a_crash_during_cancellation_counts_as_cancelled(harness) -> None:
    def dies_when_asked(worker: FakeWorker, req: RunRequest) -> None:
        worker.emit(Progress("transcribing", 0.1, "working"))
        worker.wait_for_cancel(5)
        worker.crash(-9)

    h = harness(dies_when_asked)
    job_id = h.submit()
    h.wait_until(lambda: h.jobs.get(job_id).progress > 0)
    h.jobs.cancel(job_id)
    assert h.wait(job_id).state is JobState.cancelled


def test_a_worker_that_cannot_start_fails_the_job_clearly(tmp_path: Path) -> None:
    def broken() -> FakeWorker:
        raise OSError("spawn failed")

    jobs = JobManager(Settings(data_dir=tmp_path / "d"), broken)
    try:
        h = Harness.__new__(Harness)
        h.tmp, h.jobs = tmp_path, jobs
        job_id = Harness.submit(h)
        status = jobs.wait(job_id, 5)
        assert status.state is JobState.failed and status.error.code == "WORKER_START_FAILED"
        assert "hibiki-asr setup" in status.error.hint
    finally:
        jobs.stop()


# --- lifecycle ------------------------------------------------------------------------------------------------


def test_the_model_is_unloaded_after_the_idle_period(harness) -> None:
    h = harness(ok, ok, idle_unload_seconds=1)
    h.wait(h.submit("1"))
    first = h.workers[0]
    h.wait_until(lambda: first.shut_down, timeout=6)
    assert not h.jobs.worker_running()

    assert h.wait(h.submit("2")).state is JobState.succeeded
    assert len(h.workers) == 2  # the next job reloads


def test_idle_unload_can_be_disabled(harness) -> None:
    h = harness(ok, idle_unload_seconds=0)
    h.wait(h.submit())
    time.sleep(0.4)
    assert h.jobs.worker_running() and not h.workers[0].shut_down


def test_finished_jobs_are_forgotten_after_the_ttl(tmp_path: Path) -> None:
    clock = {"now": 1_000.0}
    jobs = JobManager(
        Settings(data_dir=tmp_path / "d", job_ttl_seconds=60),
        lambda: FakeWorker(ok),
        wall_clock=lambda: clock["now"],
    )
    try:
        h = Harness.__new__(Harness)
        h.tmp, h.jobs = tmp_path, jobs
        job_id = Harness.submit(h)
        assert jobs.wait(job_id, 5).state is JobState.succeeded

        clock["now"] += 30
        with jobs._cond:
            jobs._housekeeping()
        assert jobs.get(job_id).state is JobState.succeeded  # still within the TTL

        clock["now"] += 61
        with jobs._cond:
            jobs._housekeeping()
        with pytest.raises(NotFound) as error:
            jobs.get(job_id)
        assert error.value.code == "JOB_NOT_FOUND"
    finally:
        jobs.stop()


def test_stop_cancels_queued_jobs_and_shuts_the_worker_down(harness) -> None:
    h = harness(cooperative)
    running, queued = h.submit("1"), h.submit("2")
    h.wait_until(lambda: h.jobs.get(running).progress > 0)

    h.jobs.stop()
    assert h.jobs.get(queued).state is JobState.cancelled
    assert h.jobs.get(running).state is JobState.cancelled
    assert h.workers[0].shut_down or h.workers[0].killed


def test_unknown_job(harness) -> None:
    h = harness(ok)
    with pytest.raises(NotFound):
        h.jobs.get("nope")
    with pytest.raises(NotFound):
        h.jobs.cancel("nope")
