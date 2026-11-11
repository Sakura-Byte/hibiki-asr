"""Boundary regressions: retain context without exceeding Whisper's audio window."""

from itertools import pairwise

import pytest

from hibiki_asr.pipeline.chunking import create_contiguous_chunks
from hibiki_asr.pipeline.types import SpeechSpan


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        ([SpeechSpan(0, 28), SpeechSpan(34, 70)], 29.0),
        ([SpeechSpan(0, 16), SpeechSpan(22, 70)], 20.0),
    ],
)
def test_silence_crossing_search_window_edge_is_usable(spans, expected) -> None:
    chunks = create_contiguous_chunks(spans, 30.0, 70.0)
    assert chunks[0].end == pytest.approx(expected)


def test_searches_earlier_pause_before_hard_cutting() -> None:
    chunks = create_contiguous_chunks([SpeechSpan(0, 14), SpeechSpan(16, 70)], 30.0, 70.0)
    assert chunks[0].end == 15.0


def test_preferred_window_wins_over_longer_early_pause() -> None:
    spans = [SpeechSpan(0, 12), SpeechSpan(16, 24), SpeechSpan(25, 70)]
    assert create_contiguous_chunks(spans, 30.0, 70.0)[0].end == 24.5


def test_ignores_a_gap_with_only_a_tiny_usable_sliver() -> None:
    # The whole pause is long, but only 20 ms lies before the 30 s limit.
    spans = [SpeechSpan(0, 29.98), SpeechSpan(35, 70)]
    assert create_contiguous_chunks(spans, 30.0, 70.0)[0].end == 30.0


def test_final_pair_uses_earlier_pause_that_keeps_both_chunks_in_range() -> None:
    spans = [SpeechSpan(0, 15), SpeechSpan(17, 27), SpeechSpan(28, 42)]
    chunks = create_contiguous_chunks(spans, 30.0, 42.0)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 16.0), (16.0, 42.0)]


def test_no_subsecond_tail_when_all_audio_is_speech() -> None:
    chunks = create_contiguous_chunks([SpeechSpan(0, 180.78)], 30.0, 180.78)
    assert [c.end for c in chunks[:-2]] == [30.0, 60.0, 90.0, 120.0, 150.0]
    assert chunks[-2].duration_s == pytest.approx(15.39)
    assert chunks[-1].duration_s == pytest.approx(15.39)


def test_pause_near_end_does_not_create_tiny_tail() -> None:
    chunks = create_contiguous_chunks([SpeechSpan(0, 28), SpeechSpan(29, 31)], 30.0, 31.0)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 15.5), (15.5, 31.0)]


@pytest.mark.parametrize("max_duration", [0.2, 1.0, 7.0, 30.0])
@pytest.mark.parametrize("duration_factor", [1.01, 2.01, 6.03])
def test_plans_cover_all_samples_and_respect_window_and_tail_limits(max_duration, duration_factor) -> None:
    duration = max_duration * duration_factor
    spans = [
        SpeechSpan(0, max_duration * 0.6),
        SpeechSpan(max_duration * 0.8, duration),
    ]
    chunks = create_contiguous_chunks(spans, max_duration, duration)
    assert chunks[0].start == 0.0 and chunks[-1].end == duration
    assert all(a.end == b.start for a, b in pairwise(chunks))
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(0 < c.duration_s <= max_duration + 1e-9 for c in chunks)
    assert chunks[-1].duration_s >= min(5.0, max_duration / 2) - 1e-9


def test_zero_search_window_keeps_fixed_cuts() -> None:
    chunks = create_contiguous_chunks([SpeechSpan(0, 20), SpeechSpan(25, 70)], 30.0, 70.0, 0)
    assert [c.end for c in chunks] == [30.0, 60.0, 70.0]


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_search_fraction(factor) -> None:
    with pytest.raises(ValueError, match="split_window_factor"):
        create_contiguous_chunks([], 30.0, 70.0, factor)
