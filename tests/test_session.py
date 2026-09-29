"""The worker's per-job logic, with fake loaders: device choice, the one CPU retry, error mapping, cancel."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import RTX4090, hw, rt
from hibiki_asr.diagnostics.schema import Severity
from hibiki_asr.pipeline.vad import VadInitError
from hibiki_asr.worker.protocol import Progress, RunCancelled, RunFailed, RunRequest, RunResult
from hibiki_asr.worker.session import WorkerSession

SR = 16_000


class FakeVad:
    def __init__(self, device: str = "cpu", speech: bool = True) -> None:
        self.device = device
        self.speech = speech

    def speech_chunks(self, audio, options, *, check=lambda: None, on_window=None):
        check()
        if on_window:
            on_window(1, 1)
        return [{"start": 0, "end": len(audio)}] if self.speech and len(audio) else []


class FakeModel:
    def __init__(self, device: str, compute: str) -> None:
        self.device, self.compute = device, compute

    def transcribe(self, audio, **kwargs):
        return iter([SimpleNamespace(start=0.0, end=len(audio) / SR, text=" こんにちは ")]), SimpleNamespace()


class FakeLoaders:
    def __init__(
        self,
        probe,
        runtime,
        *,
        gpu_error: str | None = None,
        cpu_error: str | None = None,
        vad_error: str | None = None,
        decode_error: str | None = None,
        speech: bool = True,
    ) -> None:
        self._probe, self._runtime = probe, runtime
        self.gpu_error, self.cpu_error, self.vad_error, self.decode_error, self.speech = (
            gpu_error,
            cpu_error,
            vad_error,
            decode_error,
            speech,
        )
        self.model_loads: list[tuple[str, str, str]] = []
        self.vad_loads = 0

    def hardware(self):
        return self._probe

    def runtime(self):
        return self._runtime

    def load_model(self, model_dir, device, compute_type, cpu_threads):
        self.model_loads.append((str(model_dir), device, compute_type))
        error = self.gpu_error if device == "cuda" else self.cpu_error
        if error:
            raise RuntimeError(error)
        return FakeModel(device, compute_type)

    def load_vad(self, vad_dir, feature_dir, prefer_gpu, threads):
        self.vad_loads += 1
        if self.vad_error:
            raise VadInitError(self.vad_error)
        return FakeVad("cuda" if prefer_gpu else "cpu", self.speech)

    def decode(self, path):
        if self.decode_error:
            raise ValueError(self.decode_error)
        return np.zeros(3 * SR, dtype=np.float32)


def make_loaders(*args, **kwargs) -> FakeLoaders:
    return FakeLoaders(*args, **kwargs)


def request(**overrides) -> RunRequest:
    base = dict(
        job_id="j1",
        audio_path="/tmp/a.wav",
        model_ref="tiny@v1",
        model_dir="/models/tiny/v1",
        components={"vad-asr": "/models/vad-asr/1", "whisper-base-fe": "/models/whisper-base-fe/1"},
        language="ja",
        task="translate",
        device="auto",
        compute_type="auto",
        allow_cpu_fallback=True,
        vad={
            "threshold": 0.5,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 200,
        },
        merge={"enabled": True, "max_gap_ms": 2000, "max_duration_ms": 20000},
        generation={},
        chunk_target_s=30.0,
        vad_threads=0,
        cpu_threads=0,
        variant="cuda12",
    )
    base.update(overrides)
    return RunRequest(**base)


def run(session: WorkerSession, req: RunRequest | None = None, *, cancelled=lambda: False):
    progress: list[Progress] = []
    return session.run(req or request(), cancelled, progress.append), progress


GPU_OK = dict(gpu_count=1, onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_cpu_machine_transcribes_and_says_it_is_cpu_only() -> None:
    session = WorkerSession(make_loaders(hw(), rt(0)))
    result, progress = run(session, request(variant="cpu"))

    assert isinstance(result, RunResult)
    assert (result.device, result.compute_type, result.degraded) == ("cpu", "int8", False)
    assert result.segments == [(0, 3000, "こんにちは")]
    assert (result.duration_s, result.speech_s) == (3.0, 3.0)
    assert [f.code for f in result.findings] == ["CPU_ONLY"]
    assert [p.stage for p in progress][:2] == ["loading", "decoding"]
    fractions = [p.fraction for p in progress]
    assert fractions == sorted(fractions) and fractions[-1] <= 1.0


def test_a_working_gpu_is_used_and_the_model_stays_loaded_between_jobs() -> None:
    loaders = make_loaders(hw(RTX4090), rt(**GPU_OK))
    session = WorkerSession(loaders)

    first, _ = run(session)
    second, _ = run(session)

    assert isinstance(first, RunResult) and (first.device, first.compute_type, first.vad_device) == (
        "cuda",
        "bfloat16",
        "cuda",
    )
    assert not first.degraded and first.findings == []
    assert isinstance(second, RunResult)
    assert len(loaders.model_loads) == 1 and loaders.vad_loads == 1  # nothing was reloaded


def test_a_gpu_load_failure_falls_back_to_the_cpu_exactly_once_and_explains_why() -> None:
    loaders = make_loaders(hw(RTX4090), rt(**GPU_OK), gpu_error="CUDA failed with error out of memory")
    session = WorkerSession(loaders)

    first, _ = run(session)
    assert isinstance(first, RunResult)
    assert (first.device, first.degraded) == ("cpu", True)
    codes = [f.code for f in first.findings]
    assert codes[0] == "GPU_LOAD_FAILED_FALLBACK_CPU" and "DEGRADED_TO_CPU" in codes
    fallback = first.findings[0]
    assert fallback.severity is Severity.warning and "out of memory" in fallback.message
    assert "ran out of memory" in fallback.hint and "int8_float16" in fallback.hint
    assert [d for _, d, _ in loaders.model_loads] == ["cuda", "cpu"]

    second, _ = run(session)  # the GPU is not attempted again in this worker
    assert isinstance(second, RunResult) and second.degraded
    assert second.findings[0].code == "GPU_LOAD_FAILED_FALLBACK_CPU"
    assert [d for _, d, _ in loaders.model_loads] == ["cuda", "cpu"]


def test_a_non_memory_gpu_failure_points_at_doctor() -> None:
    session = WorkerSession(make_loaders(hw(RTX4090), rt(**GPU_OK), gpu_error="libcudnn_ops.so.9 not found"))
    result, _ = run(session)
    assert isinstance(result, RunResult)
    assert (
        "hibiki-asr doctor" in result.findings[0].hint
        and "ALLOW_CPU_FALLBACK=false" in result.findings[0].hint
    )


def test_with_cpu_fallback_disabled_a_gpu_failure_fails_the_job_with_a_hint() -> None:
    loaders = make_loaders(hw(RTX4090), rt(**GPU_OK), gpu_error="CUDA driver version is insufficient")
    result, _ = run(WorkerSession(loaders), request(allow_cpu_fallback=False))

    assert isinstance(result, RunFailed) and result.code == "GPU_LOAD_FAILED"
    assert "insufficient" in result.message and "HIBIKI_ASR_ALLOW_CPU_FALLBACK=true" in result.hint
    assert [d for _, d, _ in loaders.model_loads] == ["cuda"]  # no silent CPU attempt


def test_asking_for_the_cpu_never_touches_the_gpu() -> None:
    loaders = make_loaders(hw(RTX4090), rt(**GPU_OK))
    result, _ = run(WorkerSession(loaders), request(device="cpu"))
    assert isinstance(result, RunResult) and (result.device, result.degraded) == ("cpu", False)
    assert [d for _, d, _ in loaders.model_loads] == ["cpu"]


def test_asking_for_cuda_on_a_machine_without_one_is_reported_as_degraded() -> None:
    result, _ = run(WorkerSession(make_loaders(hw(), rt(0))), request(device="cuda"))
    assert isinstance(result, RunResult) and result.degraded
    assert "DEGRADED_TO_CPU" in [f.code for f in result.findings]


def test_a_cpu_load_failure_is_a_model_error() -> None:
    result, _ = run(WorkerSession(make_loaders(hw(), rt(0), cpu_error="config.json not found")))
    assert isinstance(result, RunFailed) and result.code == "MODEL_LOAD_FAILED"
    assert "config.json" in result.message and "verify" in result.hint.lower()


def test_a_vad_failure_is_an_error_not_silence() -> None:
    result, _ = run(WorkerSession(make_loaders(hw(), rt(0), vad_error="cannot load VAD model model.onnx")))
    assert isinstance(result, RunFailed) and result.code == "VAD_INIT_FAILED"
    assert "model.onnx" in result.message


def test_an_undecodable_file_is_reported_with_advice() -> None:
    result, _ = run(WorkerSession(make_loaders(hw(), rt(0), decode_error="Invalid data found")))
    assert isinstance(result, RunFailed) and result.code == "AUDIO_DECODE_FAILED"
    assert "Invalid data" in result.message and "ffmpeg" in result.hint


def test_silence_yields_no_segments_and_says_so() -> None:
    result, _ = run(WorkerSession(make_loaders(hw(), rt(0), speech=False)))
    assert isinstance(result, RunResult) and result.segments == []
    assert "NO_SPEECH_DETECTED" in [f.code for f in result.findings]


@pytest.mark.parametrize("when", ["before", "during"])
def test_cancel_is_honoured(when: str) -> None:
    state = {"calls": 0}

    def cancelled() -> bool:
        state["calls"] += 1
        return when == "before" or state["calls"] > 3

    result, _ = run(WorkerSession(make_loaders(hw(), rt(0))), cancelled=cancelled)
    assert isinstance(result, RunCancelled)


def test_an_unexpected_error_still_produces_an_answer() -> None:
    class Exploding(FakeModel):
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("kaboom")

    loaders = make_loaders(hw(), rt(0))
    loaders.load_model = lambda *a, **k: Exploding("cpu", "int8")  # type: ignore[method-assign]
    result, _ = run(WorkerSession(loaders))
    assert isinstance(result, RunFailed) and result.code == "INTERNAL_ERROR" and "kaboom" in result.message


def test_only_one_model_stays_resident() -> None:
    loaders = make_loaders(hw(), rt(0))
    session = WorkerSession(loaders)
    run(session, request(model_dir="/models/a"))
    run(session, request(model_dir="/models/b"))
    run(session, request(model_dir="/models/a"))
    assert [m for m, _, _ in loaders.model_loads] == ["/models/a", "/models/b", "/models/a"]


def test_generation_overrides_reach_the_model() -> None:
    seen: dict = {}

    class Recording(FakeModel):
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)
            return super().transcribe(audio, **kwargs)

    loaders = make_loaders(hw(), rt(0))
    loaders.load_model = lambda *a, **k: Recording("cpu", "int8")  # type: ignore[method-assign]
    run(
        WorkerSession(loaders),
        request(generation={"beam_size": 5, "temperature": [0.0, 0.2]}, language="ja", task="translate"),
    )
    assert seen["beam_size"] == 5 and seen["temperature"] == [0.0, 0.2]
    assert seen["repetition_penalty"] == 1.1 and seen["language"] == "ja" and seen["task"] == "translate"
    assert Path(request().audio_path).name == "a.wav"
