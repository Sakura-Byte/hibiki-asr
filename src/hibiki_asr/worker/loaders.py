"""The real inference stack. Imported lazily, and only inside the worker process."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..diagnostics.probe import probe_hardware
from ..diagnostics.runtime import collect_runtime_facts
from ..diagnostics.schema import HardwareProbe, RuntimeFacts
from ..pipeline.transcribe import SpeechDetector, WhisperLike
from ..pipeline.types import WHISPER_SAMPLING_RATE
from ..pipeline.vad import AsmrVad


class RealLoaders:
    def hardware(self) -> HardwareProbe:
        return probe_hardware()

    def runtime(self) -> RuntimeFacts:
        return collect_runtime_facts()

    def load_model(self, model_dir: Path, device: str, compute_type: str, cpu_threads: int) -> WhisperLike:
        from faster_whisper import WhisperModel

        return WhisperModel(str(model_dir), device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    def load_vad(self, vad_dir: Path, feature_dir: Path, prefer_gpu: bool, threads: int) -> SpeechDetector:
        return AsmrVad(
            vad_dir / "model.onnx",
            vad_dir / "model_metadata.json",
            feature_dir / "preprocessor_config.json",
            prefer_gpu=prefer_gpu,
            num_threads=threads,
        )

    def decode(self, path: Path) -> np.ndarray:
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=WHISPER_SAMPLING_RATE)
