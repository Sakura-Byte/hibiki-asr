"""Whisper log-mel spectrogram in plain numpy.

The ASMR VAD model consumes exactly what ``WhisperFeatureExtractor`` produces for
``openai/whisper-base``. Depending on ``transformers`` only for that would pull in
a large dependency tree, so the (small) algorithm is implemented here and checked
against ``transformers`` by ``tests/test_melspec_golden.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = 15.0
_LOG_STEP = 27.0 / np.log(6.4)


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    log_region = freq >= _MIN_LOG_HZ
    mels = np.where(
        log_region, _MIN_LOG_MEL + np.log(np.maximum(freq, 1e-10) / _MIN_LOG_HZ) * _LOG_STEP, mels
    )
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    log_region = mels >= _MIN_LOG_MEL
    return np.where(log_region, _MIN_LOG_HZ * np.exp(np.log(6.4) / 27.0 * (mels - _MIN_LOG_MEL)), freq)


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    sampling_rate: int,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
) -> np.ndarray:
    """Slaney-scale, Slaney-normalised triangular filters, shape ``(num_frequency_bins, num_mel_filters)``."""
    mel_freqs = np.linspace(
        _hz_to_mel(np.array(min_frequency)), _hz_to_mel(np.array(max_frequency)), num_mel_filters + 2
    )
    filter_freqs = _mel_to_hz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    bank = np.maximum(0.0, np.minimum(down, up))

    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    return bank * enorm[None, :]


@dataclass(frozen=True)
class LogMelExtractor:
    feature_size: int = 80
    n_fft: int = 400
    hop_length: int = 160
    sampling_rate: int = 16_000
    n_samples: int = 480_000

    @classmethod
    def from_preprocessor_config(cls, path: str | Path) -> LogMelExtractor:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(
            feature_size=int(config.get("feature_size", 80)),
            n_fft=int(config.get("n_fft", 400)),
            hop_length=int(config.get("hop_length", 160)),
            sampling_rate=int(config.get("sampling_rate", 16_000)),
            n_samples=int(config.get("n_samples", 480_000)),
        )

    @property
    def filters(self) -> np.ndarray:
        return _cached_filters(self.n_fft, self.feature_size, self.sampling_rate)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Return ``(feature_size, n_samples // hop_length)`` float32 features for one 30 s window.

        The audio is zero padded (or truncated) to ``n_samples`` first, like the reference extractor.
        """
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size < self.n_samples:
            waveform = np.pad(waveform, (0, self.n_samples - waveform.size))
        else:
            waveform = waveform[: self.n_samples]

        # Centered STFT: reflect-pad by n_fft // 2, periodic Hann window.
        pad = self.n_fft // 2
        padded = np.pad(waveform.astype(np.float64), (pad, pad), mode="reflect")
        window = np.hanning(self.n_fft + 1)[:-1]
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.n_fft)[:: self.hop_length]
        power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2  # (frames, n_fft // 2 + 1)

        mel = np.maximum(1e-10, self.filters.T @ power.T)  # (mel, frames)
        log_spec = np.log10(mel)[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _cached_filters(n_fft: int, feature_size: int, sampling_rate: int) -> np.ndarray:
    key = (n_fft, feature_size, sampling_rate)
    bank = _FILTER_CACHE.get(key)
    if bank is None:
        bank = mel_filter_bank(1 + n_fft // 2, feature_size, sampling_rate)
        _FILTER_CACHE[key] = bank
    return bank


_FILTER_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
