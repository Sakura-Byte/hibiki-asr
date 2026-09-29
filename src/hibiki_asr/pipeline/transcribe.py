"""The transcription pipeline: outer VAD -> smart chunks -> per-chunk VAD + Whisper -> merge.

Ported from Faster-Whisper-TransWithAI-ChickenRice v1.10 (MIT), see NOTICE. It is
written against two small protocols so it can be tested without a GPU or model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .cancel import CancelCheck, never_cancelled
from .chunking import create_contiguous_chunks, speech_spans_from_vad
from .merge import MergeOptions, enforce_timeline, merge_segments
from .speech_map import SpeechTimeMap, concat_speech
from .types import MAX_CHUNK_DURATION_S, WHISPER_SAMPLING_RATE, Segment
from .vad import VadOptions

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]

# Defaults carried over from upstream's generation_config.json5. Anything else
# keeps faster-whisper's own defaults. Users can override via settings.generation.
DEFAULT_GENERATION: dict[str, Any] = {
    "beam_size": 1,
    "condition_on_previous_text": False,
    "repetition_penalty": 1.1,
    "max_initial_timestamp": 30,
}

# Share of the progress bar taken by the whole-file VAD pass.
_VAD_SHARE = 0.15


class SpeechDetector(Protocol):
    device: str

    def speech_chunks(
        self,
        audio: np.ndarray,
        options: VadOptions,
        *,
        check: CancelCheck = ...,
        on_window: Callable[[int, int], None] | None = ...,
    ) -> list[dict[str, int]]: ...


class WhisperLike(Protocol):
    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


@dataclass(frozen=True)
class PipelineOptions:
    language: str
    task: str  # "transcribe" | "translate"
    vad: VadOptions = field(default_factory=VadOptions)
    merge: MergeOptions = field(default_factory=MergeOptions)
    generation: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_GENERATION))
    chunk_target_s: float = MAX_CHUNK_DURATION_S
    split_window_factor: float = 0.4


@dataclass(frozen=True)
class PipelineResult:
    segments: list[Segment]
    duration_s: float
    speech_s: float
    chunk_count: int


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _noop_progress(stage: str, fraction: float, message: str) -> None:
    return None


def run_pipeline(
    *,
    model: WhisperLike,
    vad: SpeechDetector,
    audio: np.ndarray,
    options: PipelineOptions,
    check: CancelCheck = never_cancelled,
    on_progress: ProgressFn = _noop_progress,
) -> PipelineResult:
    """Transcribe ``audio`` (float32 mono, 16 kHz) into cleaned-up segments.

    ``check`` is called between VAD windows and between decoded segments and
    raises ``JobCancelled`` when the job was cancelled.
    """
    if options.chunk_target_s <= 0:
        raise ValueError("chunk_target_s must be greater than 0")

    sr = WHISPER_SAMPLING_RATE
    duration = len(audio) / sr

    def vad_window(done: int, total: int) -> None:
        on_progress("vad", _VAD_SHARE * done / total, f"detecting speech {done}/{total}")

    outer = vad.speech_chunks(audio, options.vad, check=check, on_window=vad_window)
    speech_s = sum(c["end"] - c["start"] for c in outer) / sr
    if not outer:
        logger.info("no speech detected in %.1fs of audio", duration)
        return PipelineResult([], duration, 0.0, 0)

    chunks = create_contiguous_chunks(
        speech_spans_from_vad(outer, sr),
        min(options.chunk_target_s, MAX_CHUNK_DURATION_S),
        duration,
        options.split_window_factor,
    )
    logger.info("planned %d chunk(s) for %.1fs of audio (%.1fs speech)", len(chunks), duration, speech_s)

    segments: list[Segment] = []
    for chunk in chunks:
        check()
        start = max(0, min(len(audio), int(round(chunk.start * sr))))
        end = max(start, min(len(audio), int(round(chunk.end * sr))))
        if end <= start:
            continue

        segments.extend(_transcribe_chunk(model, vad, audio[start:end], chunk.start, chunk.end, options, check))
        fraction = _VAD_SHARE + (1 - _VAD_SHARE) * (chunk.end / duration if duration else 1.0)
        on_progress("transcribing", min(fraction, 1.0), f"transcribed chunk {chunk.index + 1}/{len(chunks)}")

    merged = merge_segments(segments, options.merge)
    final = enforce_timeline(merged, options.merge.max_duration_ms if options.merge.enabled else None)
    return PipelineResult(final, duration, speech_s, len(chunks))


def _transcribe_chunk(
    model: WhisperLike,
    vad: SpeechDetector,
    chunk_audio: np.ndarray,
    chunk_start_s: float,
    chunk_end_s: float,
    options: PipelineOptions,
    check: CancelCheck,
) -> list[Segment]:
    sr = WHISPER_SAMPLING_RATE
    inner = vad.speech_chunks(chunk_audio, options.vad, check=check)
    if not inner:
        return []

    speech_audio = concat_speech(chunk_audio, inner)
    time_map = SpeechTimeMap(inner, sr)

    kwargs = dict(options.generation)
    kwargs.update(language=options.language, task=options.task, vad_filter=False)
    raw_segments, _info = model.transcribe(speech_audio, **kwargs)

    offset_ms = _ms(chunk_start_s)
    end_limit_ms = _ms(chunk_end_s)
    produced: list[Segment] = []
    for raw in _iter_checked(raw_segments, check):
        start_ms = offset_ms + _ms(time_map.original_time(raw.start))
        if start_ms >= end_limit_ms:
            continue
        end_ms = min(offset_ms + _ms(time_map.original_time(raw.end, is_end=True)), end_limit_ms)
        text = raw.text.strip()
        if end_ms > start_ms and text:
            produced.append(Segment(start_ms, end_ms, text))
    return produced


def _iter_checked(items: Iterable[Any], check: CancelCheck) -> Iterable[Any]:
    """Iterate a lazy segment generator, checking for cancellation before each segment."""
    iterator = iter(items)
    while True:
        check()
        try:
            yield next(iterator)
        except StopIteration:
            return

