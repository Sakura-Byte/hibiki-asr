"""Subtitle post-processing: merge repeated/overlapping segments, enforce a sane timeline.

Ported from Faster-Whisper-TransWithAI-ChickenRice (MIT), see NOTICE.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Segment


@dataclass(frozen=True)
class MergeOptions:
    enabled: bool = True
    # Only merge when the gap between neighbours is at most this long, so two
    # utterances separated by a long silence are never fused.
    max_gap_ms: int = 2_000
    # Stop merging once the fused segment would be longer than this.
    max_duration_ms: int = 20_000


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def merge_segments(segments: list[Segment], options: MergeOptions | None = None) -> list[Segment]:
    """Fold a segment into its predecessor when one repeats or extends the other."""
    options = options or MergeOptions()

    ordered = sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start_ms, s.end_ms))
    if not options.enabled:
        return ordered

    merged: list[Segment] = []
    for seg in ordered:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        if seg.start_ms - last.end_ms > options.max_gap_ms:
            merged.append(seg)
            continue
        if seg.end_ms - last.start_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_text = _normalize(last.text)
        seg_text = _normalize(seg.text)
        end_ms = max(last.end_ms, seg.end_ms)

        if seg_text.startswith(last_text):
            # The new segment extends the previous one: keep the longer text.
            merged[-1] = Segment(last.start_ms, end_ms, seg.text)
        elif last_text.startswith(seg_text) or last_text.endswith(seg_text):
            # The new segment repeats part of the previous one: keep the previous text.
            merged[-1] = Segment(last.start_ms, end_ms, last.text)
        elif seg_text.endswith(last_text):
            merged[-1] = Segment(last.start_ms, end_ms, seg.text)
        else:
            merged.append(seg)

    return merged


def enforce_timeline(segments: list[Segment], max_duration_ms: int | None = None) -> list[Segment]:
    """Drop empty text, force start times to be monotonic, and cap each segment's length."""
    normalized: list[Segment] = []

    for segment in sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start_ms, s.end_ms)):
        start = max(segment.start_ms, normalized[-1].end_ms if normalized else segment.start_ms)
        end = segment.end_ms
        if max_duration_ms is not None and max_duration_ms > 0:
            end = min(end, start + max_duration_ms)
        if end <= start:
            continue
        normalized.append(Segment(start, end, segment.text))

    return normalized
