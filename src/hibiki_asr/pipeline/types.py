"""Small value types shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

WHISPER_SAMPLING_RATE = 16_000
MAX_CHUNK_DURATION_S = 30.0


@dataclass(frozen=True)
class Segment:
    """A subtitle segment. Times are integer milliseconds from the start of the audio."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SpeechSpan:
    """A span of detected speech, in seconds."""

    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioChunk:
    """A contiguous slice of the source audio handed to Whisper, in seconds."""

    index: int
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)
