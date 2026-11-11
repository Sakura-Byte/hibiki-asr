"""Whisper on the Apple GPU through MLX. Imported lazily, only in the worker, only when the device is ``mlx``.

mlx-whisper is used for what it does well (the model, the mel spectrogram, the 30 second window loop), and this
module adds what the engine needs on top of it:

* the weights come straight from the CTranslate2 ``model.bin`` the catalog installs (see ``ct2_weights``), so the
  same model directory serves both backends;
* ``repetition_penalty``, which the fine-tuned translation models are run with and mlx-whisper does not have;
* faster-whisper's ``transcribe`` signature, which is the pipeline's ``WhisperLike`` protocol.

mlx-whisper has no beam search: ``beam_size`` above 1 is decoded greedily (the engine's default is 1).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_whisper import decoding, whisper
from mlx_whisper.transcribe import ModelHolder

from .ct2_weights import UnsupportedModel, dimensions, iter_whisper_weights, read_index

logger = logging.getLogger(__name__)

COMPUTE_TYPES = ("float16", "float32")

# faster-whisper option -> mlx-whisper option. Options with the same meaning under another name.
_RENAMED = {"log_prob_threshold": "logprob_threshold"}
# Options mlx-whisper understands as they are.
_PASSED = frozenset(
    {
        "language",
        "task",
        "temperature",
        "compression_ratio_threshold",
        "logprob_threshold",
        "no_speech_threshold",
        "condition_on_previous_text",
        "initial_prompt",
        "prefix",
        "suppress_blank",
        "suppress_tokens",
        "without_timestamps",
        "max_initial_timestamp",
        "hallucination_silence_threshold",
        "length_penalty",
        "best_of",
    }
)
# Handled here, or meaningless for this backend.
_CONSUMED = frozenset({"beam_size", "patience", "repetition_penalty", "vad_filter"})


class RepetitionPenalty(decoding.LogitFilter):  # type: ignore[misc]
    """CTranslate2's rule: the logit of every token already in the sequence is divided by the penalty when it is
    positive and multiplied by it when it is negative."""

    def __init__(self, penalty: float, n_vocab: int) -> None:
        self.penalty = penalty
        self.n_vocab = n_vocab

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        seen = np.zeros((tokens.shape[0], self.n_vocab), dtype=bool)
        np.put_along_axis(seen, np.asarray(tokens), True, axis=1)
        penalized = mx.where(logits > 0, logits / self.penalty, logits * self.penalty)
        return mx.where(mx.array(seen), penalized, logits)


@contextmanager
def _decoding_with(penalty: float) -> Iterator[None]:
    """mlx-whisper builds its logit filters inside ``DecodingTask``; add ours for the duration of one call."""
    if penalty == 1.0:
        yield
        return
    original = decoding.DecodingTask

    class PenalizedTask(original):  # type: ignore[misc, valid-type]
        def __init__(self, model: Any, options: Any) -> None:
            super().__init__(model, options)
            # First, like CTranslate2: the timestamp rules must judge the penalized distribution.
            self.logit_filters.insert(0, RepetitionPenalty(penalty, model.dims.n_vocab))

    decoding.DecodingTask = PenalizedTask
    try:
        yield
    finally:
        decoding.DecodingTask = original


def load_weights(model_dir: Path, dtype: Any) -> Any:
    """Build mlx-whisper's model from a CTranslate2 model directory."""
    path = model_dir / "model.bin"
    index = read_index(path)
    dims = dimensions(path, index)
    model = whisper.Whisper(whisper.ModelDimensions(**vars(dims)), dtype)

    parameters: list[tuple[str, Any]] = []
    tree_flatten(model.parameters(), destination=parameters)
    expected = {
        name: tuple(value.shape)
        for name, value in parameters
        if name != "alignment_heads" and not name.rsplit(".", 1)[-1].startswith("_")
    }
    weights: list[tuple[str, mx.array]] = []
    for name, value in iter_whisper_weights(path, index, dims):
        if value.ndim == 3:
            value = value.swapaxes(1, 2)  # (out, in, kernel) -> MLX's (out, kernel, in)
        if expected.get(name) != tuple(value.shape):
            raise UnsupportedModel(
                f"{name} has the shape {tuple(value.shape)}, the model expects {expected.get(name)}"
            )
        weights.append((name, mx.array(value).astype(dtype)))
        mx.eval(weights[-1][1])  # convert now, so the float32 copy can be freed before the next one is read
    missing = expected.keys() - {name for name, _ in weights}
    if missing:
        raise UnsupportedModel(f"the model file has no weights for {', '.join(sorted(missing)[:5])}")

    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


class MlxWhisper:
    """The pipeline's ``WhisperLike``, on MLX."""

    def __init__(self, model_dir: Path, compute_type: str = "float16") -> None:
        if compute_type not in COMPUTE_TYPES:
            raise ValueError(
                f"compute type {compute_type!r} is not available on MLX; use one of {', '.join(COMPUTE_TYPES)}"
            )
        self._fp16 = compute_type == "float16"
        self._key = f"hibiki-asr:{model_dir}:{compute_type}"
        self._model = load_weights(model_dir, mx.float16 if self._fp16 else mx.float32)
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("%s", message)

    def _options(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], float]:
        options: dict[str, Any] = {}
        for name, value in kwargs.items():
            name = _RENAMED.get(name, name)
            if name in _PASSED:
                options[name] = value
            elif name not in _CONSUMED:
                self._warn_once(name, f"the MLX backend ignores the generation option {name!r}")
        if (kwargs.get("beam_size") or 1) > 1:
            self._warn_once(
                "beam_size", "the MLX backend has no beam search; decoding greedily (beam_size=1)"
            )
        if isinstance(options.get("temperature"), list):
            options["temperature"] = tuple(options["temperature"])
        return options, float(kwargs.get("repetition_penalty") or 1.0)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[Any], Any]:
        options, penalty = self._options(kwargs)
        # mlx-whisper loads models by path through this holder; hand it ours for the call, and take it back so
        # that dropping this object really frees the memory.
        ModelHolder.model, ModelHolder.model_path = self._model, self._key
        try:
            with _decoding_with(penalty):
                result = mlx_whisper.transcribe(
                    audio, path_or_hf_repo=self._key, fp16=self._fp16, verbose=None, **options
                )
        finally:
            ModelHolder.model, ModelHolder.model_path = None, None
        segments = [
            SimpleNamespace(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]))
            for s in result["segments"]
        ]
        return segments, SimpleNamespace(language=result.get("language"))
