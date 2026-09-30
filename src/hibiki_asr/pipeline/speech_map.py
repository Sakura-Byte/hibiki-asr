"""Concatenate speech-only audio and map timestamps back to the original timeline.

Whisper hallucinates on long silences, so each chunk is reduced to just its
speech before decoding. Decoding then reports times on the shortened audio;
``SpeechTimeMap`` translates them back. This mirrors what faster-whisper does
internally for ``vad_filter=True`` (SYSTRAN/faster-whisper, MIT, see NOTICE), but
it lives here so the pipeline does not depend on faster-whisper's private helpers.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence

import numpy as np


def concat_speech(audio: np.ndarray, chunks: Sequence[Mapping[str, int]]) -> np.ndarray:
    """Join the ``[start, end)`` sample ranges of ``chunks`` into one array."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[c["start"] : c["end"]] for c in chunks])


class SpeechTimeMap:
    """Translate times on the concatenated speech audio to times on the original audio."""

    def __init__(self, chunks: Sequence[Mapping[str, int]], sampling_rate: int, precision: int = 2) -> None:
        self.sampling_rate = sampling_rate
        self.precision = precision
        self._chunk_end_sample: list[int] = []
        self._silence_before_s: list[float] = []

        previous_end = 0
        silent_samples = 0
        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]
            self._chunk_end_sample.append(chunk["end"] - silent_samples)
            self._silence_before_s.append(silent_samples / sampling_rate)

    def _chunk_index(self, time: float, is_end: bool) -> int:
        sample = int(time * self.sampling_rate)
        if is_end and sample in self._chunk_end_sample:
            return self._chunk_end_sample.index(sample)
        return min(bisect.bisect(self._chunk_end_sample, sample), len(self._chunk_end_sample) - 1)

    def original_time(self, time: float, *, is_end: bool = False) -> float:
        index = self._chunk_index(time, is_end)
        return round(self._silence_before_s[index] + time, self.precision)
