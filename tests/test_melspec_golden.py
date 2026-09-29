"""The numpy log-mel must match transformers' WhisperFeatureExtractor (the VAD's training front end)."""

from __future__ import annotations

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from hibiki_asr.pipeline.melspec import LogMelExtractor, mel_filter_bank  # noqa: E402


def _audio(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16_000)) / 16_000
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    return (tone + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.mark.parametrize("feature_size", [80, 128])
def test_filters_match_transformers(feature_size: int) -> None:
    reference = transformers.WhisperFeatureExtractor(feature_size=feature_size)
    ours = mel_filter_bank(201, feature_size, 16_000)
    np.testing.assert_allclose(ours, reference.mel_filters, atol=1e-7)


@pytest.mark.parametrize("feature_size", [80, 128])
@pytest.mark.parametrize("seconds", [3.0, 30.0, 41.0])
def test_features_match_transformers(feature_size: int, seconds: float) -> None:
    reference = transformers.WhisperFeatureExtractor(feature_size=feature_size)
    audio = _audio(seconds)

    expected = reference(audio, sampling_rate=16_000, return_tensors="np").input_features[0]
    actual = LogMelExtractor(feature_size=feature_size)(audio)

    assert actual.shape == expected.shape == (feature_size, 3000)
    np.testing.assert_allclose(actual, expected, atol=2e-4)


def test_silence_is_stable() -> None:
    features = LogMelExtractor()(np.zeros(16_000, dtype=np.float32))
    assert features.shape == (80, 3000)
    assert np.isfinite(features).all()


def test_from_preprocessor_config(tmp_path) -> None:
    path = tmp_path / "preprocessor_config.json"
    path.write_text('{"feature_size": 128, "n_fft": 400, "hop_length": 160, "n_samples": 480000, "sampling_rate": 16000}')
    assert LogMelExtractor.from_preprocessor_config(path) == LogMelExtractor(feature_size=128)
