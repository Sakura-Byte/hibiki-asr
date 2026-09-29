"""Subtitle serialisation. Pure functions that return text; no file IO."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .types import Segment

FORMATS = ("vtt", "srt", "lrc", "txt")


def _clock(ms: int, delimiter: str) -> str:
    ms = max(0, int(ms))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{delimiter}{ms:03d}"


def _cue_text(text: str) -> str:
    """A cue payload must not contain a blank line or the ``-->`` arrow."""
    lines = [line.strip() for line in text.strip().splitlines()]
    return "\n".join(line.replace("-->", "->") for line in lines if line)


def to_vtt(segments: Sequence[Segment]) -> str:
    parts = ["WEBVTT\n\n"]
    for index, seg in enumerate(segments, start=1):
        parts.append(f"{index}\n{_clock(seg.start_ms, '.')} --> {_clock(seg.end_ms, '.')}\n{_cue_text(seg.text)}\n\n")
    return "".join(parts)


def to_srt(segments: Sequence[Segment]) -> str:
    parts: list[str] = []
    for index, seg in enumerate(segments, start=1):
        parts.append(f"{index}\n{_clock(seg.start_ms, ',')} --> {_clock(seg.end_ms, ',')}\n{_cue_text(seg.text)}\n\n")
    return "".join(parts)


def _lrc_clock(ms: int) -> str:
    ms = max(0, int(ms))
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{ms // 10:02d}"


def to_lrc(segments: Sequence[Segment]) -> str:
    lines: list[str] = []
    for index, seg in enumerate(segments):
        lines.append(f"[{_lrc_clock(seg.start_ms)}]{seg.text}\n")
        end = _lrc_clock(seg.end_ms)
        # An end marker is redundant when the next line starts at the same instant.
        if index + 1 < len(segments) and end == _lrc_clock(segments[index + 1].start_ms):
            continue
        lines.append(f"[{end}]\n")
    return "".join(lines)


def to_txt(segments: Sequence[Segment]) -> str:
    return "".join(f"{seg.text}\n" for seg in segments)


_WRITERS: dict[str, Callable[[Sequence[Segment]], str]] = {
    "vtt": to_vtt,
    "srt": to_srt,
    "lrc": to_lrc,
    "txt": to_txt,
}


def render(segments: Sequence[Segment], fmt: str) -> str:
    try:
        return _WRITERS[fmt](segments)
    except KeyError:
        raise ValueError(f"unsupported subtitle format {fmt!r}; expected one of {', '.join(FORMATS)}") from None
