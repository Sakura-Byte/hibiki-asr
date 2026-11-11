"""Read the weights of a CTranslate2 Whisper model (``model.bin``) with NumPy only.

The MLX backend runs the very files the catalog installs for CTranslate2: this module turns them into the
parameter names and layouts of the reference Whisper implementation (which mlx-whisper follows), so a model needs
no second download and no conversion step on disk. It has no MLX import and is tested on every platform.

Layout of ``model.bin`` (``ctranslate2.specs.ModelSpec._serialize``, binary version 6), little endian:

    uint32 version, string spec, uint32 revision, uint32 count,
    count x (string name, uint8 rank, rank x uint32 dims, uint8 dtype, uint32 bytes, data),
    uint32 aliases, aliases x (string alias, string name)

A string is a uint16 length (including the terminator), the bytes and a NUL.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

SUPPORTED_BINARY_VERSIONS = (6,)
_DTYPES = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
_STORAGE = {
    "float32": "<f4",
    "int8": "i1",
    "int16": "<i2",
    "int32": "<i4",
    "float16": "<f2",
    "bfloat16": "<u2",
}


class UnsupportedModel(Exception):
    """The file is not a CTranslate2 Whisper model this reader understands. The message is meant for the user."""


@dataclass(frozen=True)
class Variable:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int


@dataclass(frozen=True)
class WhisperDims:
    """The reference implementation's ``ModelDimensions``."""

    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def _read(handle: BinaryIO, fmt: str) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise UnsupportedModel("the model file ends unexpectedly; it may be damaged or incomplete")
    return struct.unpack(fmt, data)


def _read_string(handle: BinaryIO) -> str:
    (length,) = _read(handle, "<H")
    data = handle.read(length)
    if len(data) != length:
        raise UnsupportedModel("the model file ends unexpectedly; it may be damaged or incomplete")
    return data[:-1].decode("utf-8")


def read_index(path: Path) -> dict[str, Variable]:
    """Where each variable lives in the file. Reads the headers only, not the weights."""
    variables: dict[str, Variable] = {}
    with open(path, "rb") as handle:
        (version,) = _read(handle, "<I")
        if version not in SUPPORTED_BINARY_VERSIONS:
            raise UnsupportedModel(
                f"CTranslate2 binary version {version} is not supported (supported: "
                f"{', '.join(map(str, SUPPORTED_BINARY_VERSIONS))})"
            )
        spec = _read_string(handle)
        if spec != "WhisperSpec":
            raise UnsupportedModel(f"this is a {spec} model, not a Whisper model")
        _read(handle, "<I")  # spec revision
        (count,) = _read(handle, "<I")
        for _ in range(count):
            name = _read_string(handle)
            (rank,) = _read(handle, "<B")
            shape = _read(handle, f"<{rank}I") if rank else ()
            (dtype_id,) = _read(handle, "<B")
            (size,) = _read(handle, "<I")
            if dtype_id >= len(_DTYPES):
                raise UnsupportedModel(f"variable {name} has the unknown data type {dtype_id}")
            variables[name] = Variable(name, tuple(shape), _DTYPES[dtype_id], handle.tell(), size)
            handle.seek(size, 1)
        (aliases,) = _read(handle, "<I")
        for _ in range(aliases):
            alias, target = _read_string(handle), _read_string(handle)
            if target in variables:
                variables[alias] = variables[target]
    return variables


def _raw(path: Path, variable: Variable) -> np.ndarray:
    data = np.fromfile(
        path,
        dtype=_STORAGE[variable.dtype],
        count=int(np.prod(variable.shape, dtype=np.int64)),
        offset=variable.offset,
    )
    if data.nbytes != variable.size:
        raise UnsupportedModel(
            f"variable {variable.name} is shorter than its header says; the file is damaged"
        )
    return data.reshape(variable.shape)


def read_float(path: Path, index: dict[str, Variable], name: str) -> np.ndarray:
    """One variable as float32, undoing CTranslate2's storage type (bfloat16) or quantization (int8, int16)."""
    variable = index[name]
    data = _raw(path, variable)
    if variable.dtype == "bfloat16":
        # bfloat16 is the upper half of a float32.
        return (data.astype(np.uint32) << 16).view(np.float32)
    if variable.dtype in ("int8", "int16"):
        scale_name = f"{name}_scale"
        if scale_name not in index:
            raise UnsupportedModel(f"quantized variable {name} has no scale")
        scale = _raw(path, index[scale_name]).astype(np.float32)
        # int8 is quantized per output row, int16 with one scale for the whole matrix.
        return data.astype(np.float32) / (
            scale.reshape(-1, *([1] * (data.ndim - 1))) if scale.ndim else scale
        )
    return data.astype(np.float32, copy=False)


def _scalar(path: Path, index: dict[str, Variable], name: str) -> int:
    return int(_raw(path, index[name]).reshape(-1)[0])


def _layers(index: dict[str, Variable], side: str) -> int:
    found = {int(m.group(1)) for name in index if (m := re.match(rf"{side}/layer_(\d+)/", name))}
    if not found or found != set(range(len(found))):
        raise UnsupportedModel(f"the {side} layers are not numbered 0..n")
    return len(found)


def dimensions(path: Path, index: dict[str, Variable]) -> WhisperDims:
    try:
        conv1 = index["encoder/conv1/weight"].shape
        audio_positions = index["encoder/position_encodings/encodings"].shape
        embeddings = index["decoder/embeddings/weight"].shape
        text_positions = index["decoder/position_encodings/encodings"].shape
        return WhisperDims(
            n_mels=conv1[1],
            n_audio_ctx=audio_positions[0],
            n_audio_state=conv1[0],
            n_audio_head=_scalar(path, index, "encoder/num_heads"),
            n_audio_layer=_layers(index, "encoder"),
            n_vocab=embeddings[0],
            n_text_ctx=text_positions[0],
            n_text_state=embeddings[1],
            n_text_head=_scalar(path, index, "decoder/num_heads"),
            n_text_layer=_layers(index, "decoder"),
        )
    except (KeyError, IndexError) as exc:
        raise UnsupportedModel(f"the model has no {exc.args[0]!r}: not a standard Whisper layout") from None


def _linear(
    path: Path, index: dict[str, Variable], source: str, target: str
) -> Iterator[tuple[str, np.ndarray]]:
    yield f"{target}.weight", read_float(path, index, f"{source}/weight")
    yield f"{target}.bias", read_float(path, index, f"{source}/bias")


def _norm(
    path: Path, index: dict[str, Variable], source: str, target: str
) -> Iterator[tuple[str, np.ndarray]]:
    yield f"{target}.weight", read_float(path, index, f"{source}/gamma")
    yield f"{target}.bias", read_float(path, index, f"{source}/beta")


def _fused(
    path: Path, index: dict[str, Variable], source: str, parts: tuple[tuple[str, bool], ...]
) -> Iterator[tuple[str, np.ndarray]]:
    """CTranslate2 stacks the query/key/value projections of one attention into a single matrix; undo that.

    ``parts`` names each slice and says whether it has a bias: Whisper's key projection has none, and CTranslate2
    stores zeros in its place.
    """
    weight = read_float(path, index, f"{source}/weight")
    bias = read_float(path, index, f"{source}/bias")
    width = weight.shape[0] // len(parts)
    for i, (target, has_bias) in enumerate(parts):
        yield f"{target}.weight", weight[i * width : (i + 1) * width]
        if has_bias:
            yield f"{target}.bias", bias[i * width : (i + 1) * width]


def iter_whisper_weights(
    path: Path, index: dict[str, Variable], dims: WhisperDims
) -> Iterator[tuple[str, np.ndarray]]:
    """``(parameter name, float32 array)`` for every parameter of the reference Whisper model.

    One array is materialised at a time, so a multi-gigabyte model never sits in memory twice. Convolutions are
    kept as CTranslate2 (and PyTorch) store them, ``(out, in, kernel)``; the caller moves the axes its framework
    wants. The encoder's position encodings are left out: they are the fixed sinusoids every implementation
    computes itself. The output projection is the token embedding (tied weights) and is not repeated.
    """
    yield from _linear(path, index, "encoder/conv1", "encoder.conv1")
    yield from _linear(path, index, "encoder/conv2", "encoder.conv2")
    for i in range(dims.n_audio_layer):
        src, dst = f"encoder/layer_{i}", f"encoder.blocks.{i}"
        attn = f"{dst}.attn"
        yield from _fused(
            path,
            index,
            f"{src}/self_attention/linear_0",
            ((f"{attn}.query", True), (f"{attn}.key", False), (f"{attn}.value", True)),
        )
        yield from _linear(path, index, f"{src}/self_attention/linear_1", f"{attn}.out")
        yield from _norm(path, index, f"{src}/self_attention/layer_norm", f"{dst}.attn_ln")
        yield from _linear(path, index, f"{src}/ffn/linear_0", f"{dst}.mlp1")
        yield from _linear(path, index, f"{src}/ffn/linear_1", f"{dst}.mlp2")
        yield from _norm(path, index, f"{src}/ffn/layer_norm", f"{dst}.mlp_ln")
    yield from _norm(path, index, "encoder/layer_norm", "encoder.ln_post")

    yield "decoder.token_embedding.weight", read_float(path, index, "decoder/embeddings/weight")
    yield "decoder.positional_embedding", read_float(path, index, "decoder/position_encodings/encodings")
    for i in range(dims.n_text_layer):
        src, dst = f"decoder/layer_{i}", f"decoder.blocks.{i}"
        attn, cross = f"{dst}.attn", f"{dst}.cross_attn"
        yield from _fused(
            path,
            index,
            f"{src}/self_attention/linear_0",
            ((f"{attn}.query", True), (f"{attn}.key", False), (f"{attn}.value", True)),
        )
        yield from _linear(path, index, f"{src}/self_attention/linear_1", f"{attn}.out")
        yield from _norm(path, index, f"{src}/self_attention/layer_norm", f"{dst}.attn_ln")
        yield from _linear(path, index, f"{src}/attention/linear_0", f"{cross}.query")
        yield from _fused(
            path, index, f"{src}/attention/linear_1", ((f"{cross}.key", False), (f"{cross}.value", True))
        )
        yield from _linear(path, index, f"{src}/attention/linear_2", f"{cross}.out")
        yield from _norm(path, index, f"{src}/attention/layer_norm", f"{dst}.cross_attn_ln")
        yield from _linear(path, index, f"{src}/ffn/linear_0", f"{dst}.mlp1")
        yield from _linear(path, index, f"{src}/ffn/linear_1", f"{dst}.mlp2")
        yield from _norm(path, index, f"{src}/ffn/layer_norm", f"{dst}.mlp_ln")
    yield from _norm(path, index, "decoder/layer_norm", "decoder.ln")
