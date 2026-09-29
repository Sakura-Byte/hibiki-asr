"""Smart chunk planning.

Ported from Faster-Whisper-TransWithAI-ChickenRice (MIT), see NOTICE. Whisper
sees at most 30 s at a time, so a long recording is cut into contiguous chunks.
Instead of cutting at a fixed interval, the cut is placed in the middle of the
longest silence found in the last part of the window, which avoids splitting a
sentence in half.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .types import WHISPER_SAMPLING_RATE, AudioChunk, SpeechSpan

# A gap shorter than this is not considered a usable place to cut.
MIN_SPLIT_GAP_S = 0.1


def speech_spans_from_vad(
    vad_segments: Iterable[Mapping[str, float]], sampling_rate: int = WHISPER_SAMPLING_RATE
) -> list[SpeechSpan]:
    """Convert VAD output (sample indices) into speech spans in seconds."""
    spans: list[SpeechSpan] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            spans.append(SpeechSpan(start=start, end=end))
    return spans


def create_contiguous_chunks(
    spans: Iterable[SpeechSpan],
    max_duration: float,
    total_duration: float,
    split_window_factor: float = 0.4,
) -> list[AudioChunk]:
    """Cover ``[0, total_duration]`` with contiguous chunks of at most ``max_duration`` seconds.

    ``split_window_factor`` is the fraction of the window (counted from its end)
    in which a silence is accepted as a cut point.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be greater than 0")
    if total_duration <= 0:
        return []
    if total_duration <= max_duration:
        return [AudioChunk(0, 0.0, total_duration)]

    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: s.start)
    chunks: list[AudioChunk] = []
    current_start = 0.0

    while current_start < total_duration:
        potential_end = current_start + max_duration
        if potential_end >= total_duration:
            chunks.append(AudioChunk(len(chunks), current_start, total_duration))
            break

        decision_zone_start = current_start + max_duration * (1 - split_window_factor)
        best_split: float | None = None
        best_gap = 0.0

        for previous, current in zip(ordered, ordered[1:], strict=False):
            gap_start, gap_end = previous.end, current.start
            if decision_zone_start <= gap_start and gap_end <= potential_end:
                gap = gap_end - gap_start
                if gap > MIN_SPLIT_GAP_S and gap > best_gap:
                    best_gap = gap
                    best_split = gap_start + gap / 2

        split_point = best_split if best_split is not None else potential_end
        split_point = max(current_start, min(split_point, potential_end, total_duration))
        chunks.append(AudioChunk(len(chunks), current_start, split_point))
        current_start = split_point

    return chunks
