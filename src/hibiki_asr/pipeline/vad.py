"""Voice activity detection with the ASMR-tuned Whisper encoder/decoder ONNX model.

Ported from Faster-Whisper-TransWithAI-ChickenRice (MIT), see NOTICE. Two things differ from upstream:

* A failing VAD is an error. Upstream logs it and silently returns "no speech",
  which produces empty subtitles.
* There is one long-lived instance per worker. Upstream constructs a new manager
  (and ONNX session) for every faster-whisper VAD call.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cancel import CancelCheck, never_cancelled
from .melspec import LogMelExtractor
from .types import WHISPER_SAMPLING_RATE

logger = logging.getLogger(__name__)


class VadInitError(RuntimeError):
    """The VAD model or its feature extractor could not be loaded."""


@dataclass(frozen=True)
class VadOptions:
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 300
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 200


def speech_chunks_from_probs(
    probs: Sequence[float] | np.ndarray,
    options: VadOptions,
    *,
    frame_duration_ms: int = 20,
    sampling_rate: int = WHISPER_SAMPLING_RATE,
) -> list[dict[str, int]]:
    """Turn frame-level speech probabilities into ``{"start", "end"}`` sample ranges.

    Silero-style hysteresis: speech starts when the probability reaches
    ``threshold`` and only ends after it stayed below ``neg_threshold`` for
    ``min_silence_duration_ms``. Neighbouring segments are then padded without
    overlapping each other.
    """
    frame_samples = int(sampling_rate * frame_duration_ms / 1000)
    min_speech_frames = int(options.min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(options.min_silence_duration_ms / frame_duration_ms)
    pad_frames = int(options.speech_pad_ms / frame_duration_ms)
    total = len(probs)
    max_speech_frames = (
        int(options.max_speech_duration_s * 1000 / frame_duration_ms)
        if math.isfinite(options.max_speech_duration_s)
        else total
    )
    neg_threshold = options.neg_threshold
    if neg_threshold is None:
        neg_threshold = max(options.threshold - 0.15, 0.01)

    triggered = False
    start = 0
    temp_end = 0
    spans: list[list[int]] = []

    for i, prob in enumerate(probs):
        if prob >= options.threshold and not triggered:
            triggered = True
            start = i
            continue

        if triggered and i - start > max_speech_frames:
            spans.append([start, start + max_speech_frames])
            triggered = False
            temp_end = 0
            continue

        if prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = i
            if i - temp_end >= min_silence_frames:
                if temp_end - start >= min_speech_frames:
                    spans.append([start, temp_end])
                triggered = False
                temp_end = 0
        elif prob >= options.threshold and temp_end:
            temp_end = 0

    if triggered and total - start >= min_speech_frames:
        spans.append([start, total])

    for index, span in enumerate(spans):
        span[0] = max(0, span[0] - pad_frames) if index == 0 else max(spans[index - 1][1], span[0] - pad_frames)
        span[1] = (
            min(spans[index + 1][0], span[1] + pad_frames)
            if index < len(spans) - 1
            else min(total, span[1] + pad_frames)
        )

    return [{"start": s * frame_samples, "end": e * frame_samples} for s, e in spans]


class AsmrVad:
    """The ONNX VAD model plus the feature extractor it was trained with."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        preprocessor_config_path: str | Path,
        *,
        prefer_gpu: bool = True,
        num_threads: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise VadInitError(f"onnxruntime is not installed: {exc}") from exc

        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.extractor = LogMelExtractor.from_preprocessor_config(preprocessor_config_path)
        except (OSError, ValueError) as exc:
            raise VadInitError(f"cannot read VAD metadata or feature extractor config: {exc}") from exc

        self.frame_duration_ms = int(metadata.get("frame_duration_ms", 20))
        self.window_ms = int(metadata.get("total_duration_ms", 30_000))
        self.sampling_rate = self.extractor.sampling_rate
        self.window_samples = self.window_ms * self.sampling_rate // 1000
        self.frame_samples = self.sampling_rate * self.frame_duration_ms // 1000

        options = ort.SessionOptions()
        threads = num_threads if num_threads > 0 else max(1, (os.cpu_count() or 2) // 2)
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        self.gpu_unavailable_reason: str | None = None
        if prefer_gpu:
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            else:
                self.gpu_unavailable_reason = "onnxruntime has no CUDAExecutionProvider"

        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        except Exception as exc:
            raise VadInitError(f"cannot load VAD model {model_path}: {exc}") from exc

        used = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in used else "cpu"
        if prefer_gpu and self.device == "cpu" and self.gpu_unavailable_reason is None:
            self.gpu_unavailable_reason = "CUDAExecutionProvider was requested but is not active"
        self._input = self.session.get_inputs()[0].name
        self._outputs = [out.name for out in self.session.get_outputs()]
        logger.info("VAD loaded model=%s device=%s providers=%s", model_path, self.device, used)

    def frame_probs(
        self,
        audio: np.ndarray,
        *,
        check: CancelCheck = never_cancelled,
        on_window: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Speech probability for every 20 ms frame of ``audio``."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        windows = -(-samples.size // self.window_samples)
        probs: list[np.ndarray] = []
        for index in range(windows):
            check()
            window = samples[index * self.window_samples : (index + 1) * self.window_samples]
            features = self.extractor(window)[None, :, :]
            logits = self.session.run(self._outputs, {self._input: features})[0][0]
            probs.append(1.0 / (1.0 + np.exp(-logits)))
            if on_window is not None:
                on_window(index + 1, windows)

        frames = -(-samples.size // self.frame_samples)
        return np.concatenate(probs)[:frames]

    def speech_chunks(
        self,
        audio: np.ndarray,
        options: VadOptions,
        *,
        check: CancelCheck = never_cancelled,
        on_window: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, int]]:
        probs = self.frame_probs(audio, check=check, on_window=on_window)
        return speech_chunks_from_probs(
            probs,
            options,
            frame_duration_ms=self.frame_duration_ms,
            sampling_rate=self.sampling_rate,
        )
