"""Exercise the installed decoder, including its compatibility with faster-whisper."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from hibiki_asr.worker.loaders import RealLoaders


def test_runtime_decodes_and_resamples_audio(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes((np.sin(np.arange(8000) / 5) * 8000).astype("<i2").tobytes())

    decoded = RealLoaders().decode(path)

    assert decoded.shape == (16000,)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()
    assert abs(decoded).max() > 0.1
