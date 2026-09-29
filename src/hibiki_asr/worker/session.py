"""What the worker does for one job. Free of process plumbing so it can be tested with fake loaders."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..diagnostics.findings import evaluate_findings
from ..diagnostics.schema import Finding, HardwareProbe, RuntimeFacts, Selection, Severity
from ..diagnostics.selection import pick_compute_type, select_runtime
from ..pipeline.cancel import JobCancelled
from ..pipeline.merge import MergeOptions
from ..pipeline.transcribe import (
    DEFAULT_GENERATION,
    PipelineOptions,
    SpeechDetector,
    WhisperLike,
    run_pipeline,
)
from ..pipeline.vad import VadInitError, VadOptions
from .protocol import Progress, RunCancelled, RunFailed, RunRequest, RunResult

logger = logging.getLogger(__name__)


class Loaders(Protocol):
    """Everything that touches the real inference stack; replaced by fakes in tests."""

    def hardware(self) -> HardwareProbe: ...
    def runtime(self) -> RuntimeFacts: ...
    def load_model(
        self, model_dir: Path, device: str, compute_type: str, cpu_threads: int
    ) -> WhisperLike: ...
    def load_vad(
        self, vad_dir: Path, feature_dir: Path, prefer_gpu: bool, threads: int
    ) -> SpeechDetector: ...
    def decode(self, path: Path) -> np.ndarray: ...


@dataclass
class _Loaded:
    model: WhisperLike
    device: str
    compute_type: str


_OOM_MARKERS = ("out of memory", "cudaerrormemoryallocation", "hiperroroutofmemory")


def _looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


class WorkerSession:
    def __init__(self, loaders: Loaders) -> None:
        self._loaders = loaders
        self._probe: HardwareProbe | None = None
        self._runtime: RuntimeFacts | None = None
        self._loaded: dict[tuple[str, str, str], _Loaded] = {}
        self._vad: dict[tuple[str, str, bool], SpeechDetector] = {}
        self._gpu_broken: dict[str, str] = {}  # model dir -> why the GPU could not load it

    def _facts(self) -> tuple[HardwareProbe, RuntimeFacts]:
        if self._probe is None or self._runtime is None:
            self._probe, self._runtime = self._loaders.hardware(), self._loaders.runtime()
        return self._probe, self._runtime

    # -- loading ------------------------------------------------------------------------------------------

    def _acquire_model(
        self, req: RunRequest, selection: Selection, runtime: RuntimeFacts, dynamic: list[Finding]
    ) -> _Loaded:
        model_dir = Path(req.model_dir)
        device, compute = selection.device, selection.compute_type

        if device == "cuda" and req.model_dir in self._gpu_broken:
            # The GPU already failed for this model in this worker; do not pay for another failed attempt.
            device = "cpu"
            compute = pick_compute_type("cpu", runtime.compute_types.get("cpu", []), req.compute_type, None)
            dynamic.append(self._fallback_finding(self._gpu_broken[req.model_dir]))

        key = (req.model_dir, device, compute)
        if key in self._loaded:
            return self._loaded[key]

        try:
            model = self._loaders.load_model(model_dir, device, compute, req.cpu_threads)
        except Exception as exc:
            if device != "cuda":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            if not req.allow_cpu_fallback:
                raise _GpuLoadFailed(reason) from exc
            logger.warning("loading %s on the GPU failed (%s); retrying on the CPU", req.model_ref, reason)
            self._gpu_broken[req.model_dir] = reason
            dynamic.append(self._fallback_finding(reason))
            device = "cpu"
            compute = pick_compute_type("cpu", runtime.compute_types.get("cpu", []), req.compute_type, None)
            model = self._loaders.load_model(model_dir, device, compute, req.cpu_threads)

        # Only one model stays resident: keeping several multi-GB models around would exhaust memory.
        self._loaded = {(req.model_dir, device, compute): _Loaded(model, device, compute)}
        return self._loaded[(req.model_dir, device, compute)]

    @staticmethod
    def _fallback_finding(reason: str) -> Finding:
        oom = _looks_like_oom(reason)
        return Finding(
            code="GPU_LOAD_FAILED_FALLBACK_CPU",
            severity=Severity.warning,
            message=f"Loading the model on the GPU failed, so the CPU is used instead: {reason}",
            hint=(
                "The GPU ran out of memory. Close other programs that use the GPU, or use a smaller compute type "
                "(HIBIKI_ASR_COMPUTE_TYPE=int8_float16)."
                if oom
                else "Run `hibiki-asr doctor` for the likely cause. Set HIBIKI_ASR_ALLOW_CPU_FALLBACK=false to fail "
                "instead of falling back."
            ),
        )

    def _acquire_vad(self, req: RunRequest, prefer_gpu: bool) -> SpeechDetector:
        vad_dir, feature_dir = req.components["vad-asr"], req.components["whisper-base-fe"]
        key = (vad_dir, feature_dir, prefer_gpu)
        if key not in self._vad:
            self._vad = {
                key: self._loaders.load_vad(Path(vad_dir), Path(feature_dir), prefer_gpu, req.vad_threads)
            }
        return self._vad[key]

    # -- one job ----------------------------------------------------------------------------------------------

    def run(
        self, req: RunRequest, cancelled: Callable[[], bool], emit: Callable[[Progress], None]
    ) -> RunResult | RunFailed | RunCancelled:
        def check() -> None:
            if cancelled():
                raise JobCancelled

        try:
            return self._run(req, check, emit)
        except JobCancelled:
            return RunCancelled()
        except Exception as exc:
            logger.exception("job %s failed unexpectedly", req.job_id)
            return RunFailed(
                "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}", "See the engine log for the full traceback."
            )

    def _run(
        self, req: RunRequest, check: Callable[[], None], emit: Callable[[Progress], None]
    ) -> RunResult | RunFailed:
        probe, runtime = self._facts()
        selection = select_runtime(probe, runtime, req.device, req.compute_type, req.variant)
        dynamic: list[Finding] = []

        emit(Progress("loading", 0.0, "loading the model"))
        try:
            loaded = self._acquire_model(req, selection, runtime, dynamic)
        except _GpuLoadFailed as exc:
            return RunFailed(
                "GPU_LOAD_FAILED",
                f"Loading the model on the GPU failed and CPU fallback is disabled: {exc}",
                "Fix the GPU problem (run `hibiki-asr doctor`), or set HIBIKI_ASR_ALLOW_CPU_FALLBACK=true.",
                self._findings(probe, runtime, selection, req, dynamic),
            )
        except Exception as exc:
            return RunFailed(
                "MODEL_LOAD_FAILED",
                f"Could not load {req.model_ref}: {type(exc).__name__}: {exc}",
                "The model files may be damaged: verify them in Models, or delete and download the model again.",
                self._findings(probe, runtime, selection, req, dynamic),
            )
        check()

        final = selection.model_copy(
            update={
                "device": loaded.device,
                "compute_type": loaded.compute_type,
                "degraded": selection.degraded or loaded.device != selection.device,
            }
        )
        try:
            vad = self._acquire_vad(req, prefer_gpu=loaded.device == "cuda")
        except VadInitError as exc:
            return RunFailed(
                "VAD_INIT_FAILED",
                f"The voice activity detector could not be loaded: {exc}",
                "Verify or re-download the model in Models; its VAD component may be damaged.",
                self._findings(probe, runtime, final, req, dynamic),
            )
        final = final.model_copy(update={"vad_device": getattr(vad, "device", "cpu")})

        emit(Progress("decoding", 0.05, "decoding audio"))
        try:
            audio = self._loaders.decode(Path(req.audio_path))
        except Exception as exc:
            return RunFailed(
                "AUDIO_DECODE_FAILED",
                f"Could not decode the audio: {type(exc).__name__}: {exc}",
                "Anything ffmpeg can read works (mp3, wav, flac, m4a, ogg, mp4, ...). Check that the file is not corrupt.",
                self._findings(probe, runtime, final, req, dynamic),
            )
        check()

        options = PipelineOptions(
            language=req.language,
            task=req.task,
            vad=VadOptions(**req.vad),
            merge=MergeOptions(**req.merge),
            generation={**DEFAULT_GENERATION, **req.generation},
            chunk_target_s=req.chunk_target_s,
        )
        result = run_pipeline(
            model=loaded.model,
            vad=vad,
            audio=audio,
            options=options,
            check=check,
            on_progress=lambda stage, fraction, message: emit(Progress(stage, 0.1 + 0.9 * fraction, message)),
        )

        findings = self._findings(probe, runtime, final, req, dynamic)
        if not result.segments:
            findings.append(
                Finding(
                    code="NO_SPEECH_DETECTED",
                    severity=Severity.info,
                    message="No speech was detected in the audio, so there is nothing to transcribe.",
                    hint="Check that the file has audible speech. If it is very quiet, lower vad.threshold.",
                )
            )
        return RunResult(
            segments=[(s.start_ms, s.end_ms, s.text) for s in result.segments],
            duration_s=result.duration_s,
            speech_s=result.speech_s,
            device=final.device,
            compute_type=final.compute_type,
            vad_device=final.vad_device,
            degraded=final.degraded,
            model_ref=req.model_ref,
            findings=findings,
        )

    @staticmethod
    def _findings(
        probe: HardwareProbe,
        runtime: RuntimeFacts,
        selection: Selection,
        req: RunRequest,
        dynamic: list[Finding],
    ) -> list[Finding]:
        static = evaluate_findings(
            probe, runtime, selection, variant=req.variant, requested_compute=req.compute_type
        )
        known = {f.code for f in dynamic}
        merged = [*dynamic, *(f for f in static if f.code not in known)]
        order = {Severity.error: 0, Severity.warning: 1, Severity.info: 2}
        return sorted(merged, key=lambda f: order[f.severity])


class _GpuLoadFailed(Exception):
    pass


def vad_settings_to_dict(vad: Any) -> dict[str, Any]:
    """The VadSettings fields that VadOptions understands (threads is a loader option, not a VAD option)."""
    data = vad.model_dump()
    data.pop("threads", None)
    return data
