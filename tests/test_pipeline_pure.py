"""Pure-logic pipeline tests: no model, no GPU, no ONNX."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hibiki_asr.pipeline.cancel import JobCancelled
from hibiki_asr.pipeline.chunking import create_contiguous_chunks, speech_spans_from_vad
from hibiki_asr.pipeline.merge import MergeOptions, enforce_timeline, merge_segments
from hibiki_asr.pipeline.speech_map import SpeechTimeMap, concat_speech
from hibiki_asr.pipeline.transcribe import PipelineOptions, run_pipeline
from hibiki_asr.pipeline.types import Segment, SpeechSpan
from hibiki_asr.pipeline.vad import VadOptions, speech_chunks_from_probs
from hibiki_asr.pipeline.writers import render, to_lrc, to_srt, to_vtt

SR = 16_000


# --- chunking ---------------------------------------------------------------------------------


def test_short_audio_is_one_chunk() -> None:
    chunks = create_contiguous_chunks([], 30.0, 12.5)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 12.5)]


def test_empty_audio_has_no_chunks() -> None:
    assert create_contiguous_chunks([], 30.0, 0.0) == []


def test_invalid_max_duration() -> None:
    with pytest.raises(ValueError):
        create_contiguous_chunks([], 0, 10)


def test_cut_lands_in_the_middle_of_the_longest_silence() -> None:
    # 70 s of audio, window 30 s: decision zone is [18, 30]. Silences: 20-20.5 and 25-27 (longest).
    spans = [SpeechSpan(0, 20), SpeechSpan(20.5, 25), SpeechSpan(27, 60), SpeechSpan(61, 70)]
    chunks = create_contiguous_chunks(spans, 30.0, 70.0)
    assert chunks[0].end == pytest.approx(26.0)
    # chunks are contiguous and cover everything
    assert chunks[0].start == 0.0
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert chunks[-1].end == 70.0
    assert all(c.duration_s <= 30.0 + 1e-9 for c in chunks)


def test_hard_cut_when_no_usable_silence() -> None:
    chunks = create_contiguous_chunks([SpeechSpan(0, 100)], 30.0, 100.0)
    assert [c.end for c in chunks] == [30.0, 60.0, 90.0, 100.0]


def test_spans_from_vad_drops_empty() -> None:
    spans = speech_spans_from_vad([{"start": 0, "end": SR}, {"start": SR, "end": SR}], SR)
    assert spans == [SpeechSpan(0.0, 1.0)]


# --- merge / timeline -------------------------------------------------------------------------


def test_merge_extends_when_next_text_starts_with_previous() -> None:
    out = merge_segments([Segment(0, 1000, "こんにちは"), Segment(1000, 2500, "こんにちは、元気")])
    assert out == [Segment(0, 2500, "こんにちは、元気")]


def test_merge_drops_repeat() -> None:
    out = merge_segments([Segment(0, 2000, "abc def"), Segment(2000, 3000, "def")])
    assert out == [Segment(0, 3000, "abc def")]


def test_merge_respects_gap_and_duration_limits() -> None:
    far = merge_segments([Segment(0, 1000, "a"), Segment(9000, 10_000, "a")], MergeOptions(max_gap_ms=2000))
    assert len(far) == 2
    long = merge_segments([Segment(0, 15_000, "a"), Segment(15_000, 25_000, "a")], MergeOptions(max_duration_ms=20_000))
    assert len(long) == 2


def test_merge_disabled_only_sorts_and_filters() -> None:
    out = merge_segments([Segment(2000, 3000, "b"), Segment(0, 1000, "a"), Segment(5, 6, "  ")], MergeOptions(enabled=False))
    assert out == [Segment(0, 1000, "a"), Segment(2000, 3000, "b")]


def test_enforce_timeline_is_monotonic_and_caps_length() -> None:
    out = enforce_timeline(
        [Segment(0, 5000, "a"), Segment(3000, 4000, "b"), Segment(3500, 40_000, "c")], max_duration_ms=10_000
    )
    assert out == [Segment(0, 5000, "a"), Segment(5000, 15_000, "c")]


# --- writers ----------------------------------------------------------------------------------

SEGS = [Segment(0, 1500, "one"), Segment(1500, 3_723_004, "two\n\nlines --> x")]


def test_vtt_has_standard_header_and_sanitised_cues() -> None:
    text = to_vtt(SEGS)
    assert text.startswith("WEBVTT\n\n")

    header, *cues = text.rstrip("\n").split("\n\n")
    assert header == "WEBVTT"
    assert len(cues) == 2  # a blank line inside a cue would have split it into more blocks
    assert cues[0] == "1\n00:00:00.000 --> 00:00:01.500\none"
    # the second cue keeps its two text lines, with the stray arrow neutralised
    assert cues[1] == "2\n00:00:01.500 --> 01:02:03.004\ntwo\nlines -> x"


def test_srt_uses_comma_milliseconds() -> None:
    assert "00:00:01,500" in to_srt(SEGS)


def test_lrc_omits_redundant_end_marker() -> None:
    text = to_lrc([Segment(0, 1000, "a"), Segment(1000, 2000, "b")])
    assert text == "[00:00.00]a\n[00:01.00]b\n[00:02.00]\n"


def test_render_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        render(SEGS, "ass")


# --- speech map -------------------------------------------------------------------------------


def test_speech_time_map_restores_original_times() -> None:
    # Speech at 1-2 s and 5-6 s of the original audio; the concatenated audio is 2 s long.
    chunks = [{"start": 1 * SR, "end": 2 * SR}, {"start": 5 * SR, "end": 6 * SR}]
    mapping = SpeechTimeMap(chunks, SR)
    assert mapping.original_time(0.25) == pytest.approx(1.25)
    assert mapping.original_time(1.25) == pytest.approx(5.25)
    assert mapping.original_time(1.0, is_end=True) == pytest.approx(2.0)  # end of first chunk stays in it
    assert mapping.original_time(2.0, is_end=True) == pytest.approx(6.0)


def test_concat_speech() -> None:
    audio = np.arange(10, dtype=np.float32)
    assert concat_speech(audio, [{"start": 1, "end": 3}, {"start": 7, "end": 9}]).tolist() == [1, 2, 7, 8]
    assert concat_speech(audio, []).size == 0


# --- VAD hysteresis ---------------------------------------------------------------------------


def _probs(pattern: str) -> np.ndarray:
    return np.array([0.9 if c == "S" else 0.0 for c in pattern], dtype=np.float32)


def test_vad_pads_and_separates_segments() -> None:
    options = VadOptions(min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=40)
    # 20 ms frames: 5 frames speech, 10 frames silence, 5 frames speech, silence tail
    probs = _probs("SSSSS" + "-" * 10 + "SSSSS" + "-" * 10)
    chunks = speech_chunks_from_probs(probs, options)
    frame = 320
    assert chunks == [
        {"start": 0, "end": 7 * frame},  # 5 frames + 2 frames padding
        {"start": 13 * frame, "end": 22 * frame},
    ]


def test_vad_drops_too_short_speech() -> None:
    options = VadOptions(min_speech_duration_ms=300, min_silence_duration_ms=100, speech_pad_ms=0)
    assert speech_chunks_from_probs(_probs("SS" + "-" * 10), options) == []


def test_vad_keeps_speech_running_to_the_end() -> None:
    options = VadOptions(min_speech_duration_ms=100, speech_pad_ms=0)
    assert speech_chunks_from_probs(_probs("-----SSSSSSS"), options) == [{"start": 5 * 320, "end": 12 * 320}]


def test_vad_empty_input() -> None:
    assert speech_chunks_from_probs(np.zeros(0), VadOptions()) == []


# --- run_pipeline with fakes ------------------------------------------------------------------


class FakeVad:
    device = "cpu"

    def __init__(self, ranges_s: list[tuple[float, float]]) -> None:
        self.ranges_s = ranges_s
        self.calls = 0

    def speech_chunks(self, audio, options, *, check=lambda: None, on_window=None):
        self.calls += 1
        check()
        total = len(audio) / SR
        out = []
        base = 0.0 if len(audio) >= self._full_len else self._chunk_origin
        for s, e in self.ranges_s:
            s2, e2 = max(s - base, 0.0), min(e - base, total)
            if e2 > s2:
                out.append({"start": int(s2 * SR), "end": int(e2 * SR)})
        if on_window:
            on_window(1, 1)
        return out

    _full_len = 0
    _chunk_origin = 0.0


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), **kwargs})
        # One segment covering the whole (speech-only) audio.
        return iter([SimpleNamespace(start=0.0, end=len(audio) / SR, text=" hello ")]), SimpleNamespace()


def test_pipeline_short_audio_single_chunk() -> None:
    audio = np.zeros(10 * SR, dtype=np.float32)
    vad = FakeVad([(2.0, 4.0)])
    vad._full_len = len(audio)
    model = FakeModel()
    progress: list[tuple[str, float]] = []

    result = run_pipeline(
        model=model,
        vad=vad,
        audio=audio,
        options=PipelineOptions(language="ja", task="translate"),
        on_progress=lambda stage, frac, _msg: progress.append((stage, frac)),
    )

    assert result.segments == [Segment(2000, 4000, "hello")]
    assert result.duration_s == 10.0
    assert result.speech_s == 2.0
    call = model.calls[0]
    assert (call["language"], call["task"], call["vad_filter"]) == ("ja", "translate", False)
    assert call["beam_size"] == 1 and call["repetition_penalty"] == 1.1
    assert call["samples"] == 2 * SR  # only the speech was decoded
    assert progress[0][0] == "vad" and progress[-1][0] == "transcribing"
    assert progress[-1][1] == pytest.approx(1.0)


def test_pipeline_no_speech_returns_empty() -> None:
    vad = FakeVad([])
    result = run_pipeline(
        model=FakeModel(), vad=vad, audio=np.zeros(SR, dtype=np.float32), options=PipelineOptions("ja", "transcribe")
    )
    assert result.segments == [] and result.chunk_count == 0


def test_pipeline_cancels_between_segments() -> None:
    audio = np.zeros(10 * SR, dtype=np.float32)
    vad = FakeVad([(1.0, 3.0)])
    vad._full_len = len(audio)
    state = {"n": 0}

    def check() -> None:
        state["n"] += 1
        if state["n"] > 3:
            raise JobCancelled

    with pytest.raises(JobCancelled):
        run_pipeline(model=FakeModel(), vad=vad, audio=audio, options=PipelineOptions("ja", "transcribe"), check=check)


def test_pipeline_rejects_bad_chunk_target() -> None:
    with pytest.raises(ValueError):
        run_pipeline(
            model=FakeModel(),
            vad=FakeVad([]),
            audio=np.zeros(SR, dtype=np.float32),
            options=PipelineOptions("ja", "transcribe", chunk_target_s=0),
        )
