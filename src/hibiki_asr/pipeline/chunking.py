"""Smart chunk planning.

Ported from Faster-Whisper-TransWithAI-ChickenRice (MIT), see NOTICE. Whisper
sees at most 30 s at a time, so a long recording is cut into contiguous chunks.
Instead of cutting at a fixed interval, the cut is placed in the middle of the
longest silence found in the last part of the window, which avoids splitting a
sentence in half.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import pairwise

from .types import WHISPER_SAMPLING_RATE, AudioChunk, SpeechSpan

# Require this much silence inside the search window, even when the full gap
# extends beyond it. VAD padding has already been applied to these spans.
MIN_SPLIT_GAP_S = 0.1

# A short recording still stays in one chunk. For longer recordings, reserve
# enough context for the last chunk instead of leaving a sub-second remainder.
MIN_TAIL_DURATION_S = 5.0


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

    Prefer silences in the last ``split_window_factor`` of each window. If
    none is available, expand that search by half before resorting to a hard
    cut. A gap may cross a search-window edge; only its usable part is cut.

    Plan the final pair together when a silence lets both fit, and reserve a
    minimum tail. If there is no silence and a hard cut would leave a tiny
    tail, divide the remaining audio evenly between the last two chunks.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be greater than 0")
    if not 0 <= split_window_factor <= 1:
        raise ValueError("split_window_factor must be between 0 and 1")
    if total_duration <= 0:
        return []
    if total_duration <= max_duration:
        return [AudioChunk(0, 0.0, total_duration)]

    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: s.start)
    gaps = [(previous.end, current.start) for previous, current in pairwise(ordered)]
    preferred_offset = max_duration * (1 - split_window_factor)
    fallback_offset = max_duration * max(0.0, 1 - 1.5 * split_window_factor)
    min_tail = min(MIN_TAIL_DURATION_S, max_duration / 2)
    chunks: list[AudioChunk] = []
    current_start = 0.0

    while current_start < total_duration:
        potential_end = current_start + max_duration
        if potential_end >= total_duration:
            chunks.append(AudioChunk(len(chunks), current_start, total_duration))
            break

        latest_end = min(potential_end, total_duration - min_tail)
        best_split: float | None = None

        if total_duration - current_start <= 2 * max_duration:
            # Search the feasible boundary of the final pair, including an
            # earlier pause that would otherwise fall outside the preferred zone.
            earliest_end = max(current_start + fallback_offset, total_duration - max_duration)
            best_split = _split_in_gap(gaps, earliest_end, latest_end)

        if best_split is None:
            best_split = _split_in_gap(gaps, current_start + preferred_offset, latest_end)
        if best_split is None and fallback_offset < preferred_offset:
            best_split = _split_in_gap(gaps, current_start + fallback_offset, latest_end)

        if best_split is not None:
            split_point = best_split
        elif total_duration - potential_end < min_tail:
            split_point = (current_start + total_duration) / 2
        else:
            split_point = potential_end
        chunks.append(AudioChunk(len(chunks), current_start, split_point))
        current_start = split_point

    return chunks


def _split_in_gap(gaps: list[tuple[float, float]], start: float, end: float) -> float | None:
    """Choose the longest VAD gap with a usable intersection with ``[start, end]``."""
    best_split: float | None = None
    best_gap = 0.0
    for gap_start, gap_end in gaps:
        usable_start = max(start, gap_start)
        usable_end = min(end, gap_end)
        gap_duration = gap_end - gap_start
        if usable_end - usable_start > MIN_SPLIT_GAP_S and gap_duration > best_gap:
            best_gap = gap_duration
            best_split = (usable_start + usable_end) / 2
    return best_split
