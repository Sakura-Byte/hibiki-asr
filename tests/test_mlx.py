"""The Apple Silicon (MLX) runtime: reading CTranslate2 weights, choosing the device, and what doctor says.

Everything here runs without MLX except the tests at the end, which are skipped where it is not installed.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import numpy as np
import pytest

from helpers import CPU_TYPES, RTX4090, hw, rt
from hibiki_asr.diagnostics.findings import evaluate_findings
from hibiki_asr.diagnostics.probe import probe_hardware
from hibiki_asr.diagnostics.schema import GpuInfo, RuntimeFacts
from hibiki_asr.diagnostics.selection import normalize_device, select_runtime
from hibiki_asr.diagnostics.system import SystemAccess
from hibiki_asr.provision.variants import get_variant, recommend_variant
from hibiki_asr.worker.ct2_weights import (
    UnsupportedModel,
    WhisperDims,
    dimensions,
    iter_whisper_weights,
    read_float,
    read_index,
)

M4 = GpuInfo(vendor="apple", name="Apple M4 GPU", vram_mb=16_384, integrated=True)


def mac(*gpus: GpuInfo):
    return hw(*gpus, os="darwin", avx2=None).model_copy(update={"arch": "arm64"})


def rt_mlx(**kwargs) -> RuntimeFacts:
    facts = rt(mlx_version="0.29.3", mlx_gpu=True, **kwargs)
    facts.compute_types["mlx"] = ["float16", "float32"]
    return facts


def run(probe, runtime, *, device="auto", compute="auto", variant=None):
    selection = select_runtime(probe, runtime, device, compute, variant)
    return selection, evaluate_findings(probe, runtime, selection, variant=variant, requested_compute=compute)


# --- a CTranslate2 model file, written the way ctranslate2 writes it ---------------------------------

_DTYPE_ID = {"float32": 0, "int8": 1, "int16": 2, "int32": 3, "float16": 4, "bfloat16": 5}


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_ct2(
    path: Path, variables: dict[str, tuple[str, np.ndarray]], *, spec: str = "WhisperSpec", version: int = 6
) -> None:
    def string(text: str) -> bytes:
        data = text.encode()
        return struct.pack("<H", len(data) + 1) + data + b"\0"

    out = [struct.pack("<I", version), string(spec), struct.pack("<I", 3), struct.pack("<I", len(variables))]
    for name, (dtype, value) in variables.items():
        raw = value.tobytes()
        out += [string(name), struct.pack("<B", value.ndim), struct.pack(f"<{value.ndim}I", *value.shape)]
        out += [struct.pack("<B", _DTYPE_ID[dtype]), struct.pack("<I", len(raw)), raw]
    out += [struct.pack("<I", 1), string("decoder/projection/weight"), string("decoder/embeddings/weight")]
    path.write_bytes(b"".join(out))


STATE, HEADS, MELS, VOCAB, AUDIO_CTX, TEXT_CTX = 8, 2, 80, 51865, 6, 5


def tiny_whisper(seed: int = 0) -> dict[str, np.ndarray]:
    """One encoder and one decoder layer, float32, under CTranslate2's names. Values are exact in bfloat16."""
    rng = np.random.default_rng(seed)

    def w(*shape: int) -> np.ndarray:
        return (rng.integers(-64, 64, size=shape) / 32).astype(np.float32)

    v: dict[str, np.ndarray] = {
        "encoder/conv1/weight": w(STATE, MELS, 3),
        "encoder/conv1/bias": w(STATE),
        "encoder/conv2/weight": w(STATE, STATE, 3),
        "encoder/conv2/bias": w(STATE),
        "encoder/position_encodings/encodings": w(AUDIO_CTX, STATE),
        "encoder/layer_norm/gamma": w(STATE),
        "encoder/layer_norm/beta": w(STATE),
        "decoder/embeddings/weight": w(VOCAB, STATE),
        "decoder/position_encodings/encodings": w(TEXT_CTX, STATE),
        "decoder/layer_norm/gamma": w(STATE),
        "decoder/layer_norm/beta": w(STATE),
    }
    for side in ("encoder", "decoder"):
        base = f"{side}/layer_0"
        v[f"{base}/self_attention/linear_0/weight"] = w(3 * STATE, STATE)
        v[f"{base}/self_attention/linear_0/bias"] = w(3 * STATE)
        v[f"{base}/self_attention/linear_1/weight"] = w(STATE, STATE)
        v[f"{base}/self_attention/linear_1/bias"] = w(STATE)
        for block in ("self_attention", "ffn") + (("attention",) if side == "decoder" else ()):
            v[f"{base}/{block}/layer_norm/gamma"] = w(STATE)
            v[f"{base}/{block}/layer_norm/beta"] = w(STATE)
        v[f"{base}/ffn/linear_0/weight"] = w(4 * STATE, STATE)
        v[f"{base}/ffn/linear_0/bias"] = w(4 * STATE)
        v[f"{base}/ffn/linear_1/weight"] = w(STATE, 4 * STATE)
        v[f"{base}/ffn/linear_1/bias"] = w(STATE)
    v["decoder/layer_0/attention/linear_0/weight"] = w(STATE, STATE)
    v["decoder/layer_0/attention/linear_0/bias"] = w(STATE)
    v["decoder/layer_0/attention/linear_1/weight"] = w(2 * STATE, STATE)
    v["decoder/layer_0/attention/linear_1/bias"] = w(2 * STATE)
    v["decoder/layer_0/attention/linear_2/weight"] = w(STATE, STATE)
    v["decoder/layer_0/attention/linear_2/bias"] = w(STATE)
    return v


def write_tiny(path: Path, values: dict[str, np.ndarray], storage: str = "bfloat16") -> None:
    encode = {"bfloat16": to_bfloat16, "float16": lambda a: a.astype(np.float16), "float32": lambda a: a}
    variables = {name: (storage, encode[storage](value)) for name, value in values.items()}
    for side in ("encoder", "decoder"):
        variables[f"{side}/num_heads"] = ("int16", np.array(HEADS, dtype=np.int16))
    variables["decoder/scale_embeddings"] = ("int8", np.array(0, dtype=np.int8))  # a flag, as real files have
    write_ct2(path, variables)


@pytest.mark.parametrize("storage", ["bfloat16", "float16", "float32"])
def test_weights_come_back_under_whisper_names_with_the_attention_split(tmp_path: Path, storage: str) -> None:
    source = tiny_whisper()
    path = tmp_path / "model.bin"
    write_tiny(path, source, storage)

    index = read_index(path)
    dims = dimensions(path, index)
    assert dims == WhisperDims(MELS, AUDIO_CTX, STATE, HEADS, 1, VOCAB, TEXT_CTX, STATE, HEADS, 1)
    got = dict(iter_whisper_weights(path, index, dims))
    assert all(a.dtype == np.float32 for a in got.values())

    fused = source["encoder/layer_0/self_attention/linear_0/weight"]
    bias = source["encoder/layer_0/self_attention/linear_0/bias"]
    attn = "encoder.blocks.0.attn"
    np.testing.assert_array_equal(got[f"{attn}.query.weight"], fused[:STATE])
    np.testing.assert_array_equal(got[f"{attn}.key.weight"], fused[STATE : 2 * STATE])
    np.testing.assert_array_equal(got[f"{attn}.value.weight"], fused[2 * STATE :])
    np.testing.assert_array_equal(got[f"{attn}.query.bias"], bias[:STATE])
    np.testing.assert_array_equal(got[f"{attn}.value.bias"], bias[2 * STATE :])
    assert f"{attn}.key.bias" not in got  # Whisper's key projection has no bias

    cross = source["decoder/layer_0/attention/linear_1/weight"]
    np.testing.assert_array_equal(got["decoder.blocks.0.cross_attn.key.weight"], cross[:STATE])
    np.testing.assert_array_equal(got["decoder.blocks.0.cross_attn.value.weight"], cross[STATE:])
    np.testing.assert_array_equal(
        got["decoder.blocks.0.cross_attn.query.weight"], source["decoder/layer_0/attention/linear_0/weight"]
    )
    np.testing.assert_array_equal(
        got["decoder.blocks.0.cross_attn.out.bias"], source["decoder/layer_0/attention/linear_2/bias"]
    )
    np.testing.assert_array_equal(
        got["encoder.blocks.0.mlp1.weight"], source["encoder/layer_0/ffn/linear_0/weight"]
    )
    np.testing.assert_array_equal(
        got["decoder.blocks.0.mlp_ln.weight"], source["decoder/layer_0/ffn/layer_norm/gamma"]
    )
    np.testing.assert_array_equal(
        got["decoder.blocks.0.mlp_ln.bias"], source["decoder/layer_0/ffn/layer_norm/beta"]
    )
    np.testing.assert_array_equal(got["encoder.ln_post.weight"], source["encoder/layer_norm/gamma"])
    np.testing.assert_array_equal(got["decoder.token_embedding.weight"], source["decoder/embeddings/weight"])
    np.testing.assert_array_equal(
        got["decoder.positional_embedding"], source["decoder/position_encodings/encodings"]
    )
    assert got["encoder.conv1.weight"].shape == (STATE, MELS, 3)  # the caller reorders the axes
    assert not any("position" in name for name in got if name.startswith("encoder."))  # fixed sinusoids
    # two convolutions, one encoder block, ln_post, both embeddings, one decoder block, ln
    assert len(got) == 4 + 15 + 2 + 2 + 24 + 2


def test_quantized_weights_are_scaled_back(tmp_path: Path) -> None:
    weight = np.array([[0.5, -1.0, 0.25], [2.0, 4.0, -8.0]], dtype=np.float32)
    row_scale = np.array([100.0, 10.0], dtype=np.float32)
    path = tmp_path / "model.bin"
    write_ct2(
        path,
        {
            "a/weight": ("int8", np.round(weight * row_scale[:, None]).astype(np.int8)),
            "a/weight_scale": ("float32", row_scale),
            "b/weight": ("int16", np.round(weight * 1000).astype(np.int16)),
            "b/weight_scale": ("float32", np.array(1000.0, dtype=np.float32)),
            "c/weight": ("int8", np.zeros((2, 2), dtype=np.int8)),
        },
    )
    index = read_index(path)
    np.testing.assert_allclose(read_float(path, index, "a/weight"), weight)
    np.testing.assert_allclose(read_float(path, index, "b/weight"), weight)
    with pytest.raises(UnsupportedModel, match="has no scale"):
        read_float(path, index, "c/weight")


def test_files_that_are_not_a_whisper_model_are_refused_with_a_reason(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    write_ct2(path, {}, spec="TransformerSpec")
    with pytest.raises(UnsupportedModel, match="TransformerSpec model, not a Whisper model"):
        read_index(path)

    write_ct2(path, {}, version=7)
    with pytest.raises(UnsupportedModel, match="binary version 7 is not supported"):
        read_index(path)

    write_tiny(path, tiny_whisper())
    path.write_bytes(path.read_bytes()[:2000])
    with pytest.raises(UnsupportedModel, match="ends unexpectedly"):
        read_index(path)

    values = tiny_whisper()
    del values["encoder/conv1/weight"]
    write_tiny(path, values)
    with pytest.raises(UnsupportedModel, match="not a standard Whisper layout"):
        dimensions(path, read_index(path))


# --- which device ---------------------------------------------------------------------------------------


def test_device_names_for_the_apple_gpu() -> None:
    assert [normalize_device(v) for v in ("mlx", "MLX", "apple", "metal")] == ["mlx"] * 4
    with pytest.raises(ValueError, match="unknown device"):
        normalize_device("mps")


def test_a_mac_with_the_mlx_runtime_uses_the_gpu_and_has_nothing_to_report() -> None:
    selection, findings = run(mac(M4), rt_mlx(), variant="mlx")
    assert (selection.device, selection.compute_type, selection.vad_device, selection.degraded) == (
        "mlx",
        "float16",
        "cpu",
        False,
    )
    assert findings == []


def test_mlx_honours_an_explicit_precision_and_reports_one_it_does_not_have() -> None:
    selection, _ = run(mac(M4), rt_mlx(), compute="float32", variant="mlx")
    assert selection.compute_type == "float32"

    selection, findings = run(mac(M4), rt_mlx(), compute="int8", variant="mlx")
    assert selection.compute_type == "float16"
    downgraded = next(f for f in findings if f.code == "COMPUTE_TYPE_DOWNGRADED")
    assert "float16, float32" in downgraded.hint


def test_device_cpu_keeps_a_mac_on_the_cpu_without_complaint() -> None:
    selection, findings = run(mac(M4), rt_mlx(), device="cpu", variant="mlx")
    assert (selection.device, selection.compute_type, selection.degraded) == ("cpu", "int8", False)
    assert [f.code for f in findings] == ["CPU_ONLY"]


def test_a_mac_with_the_cpu_runtime_is_told_about_the_mlx_runtime() -> None:
    selection, findings = run(mac(M4), rt(), variant="cpu")
    assert (selection.device, selection.degraded) == ("cpu", False)
    assert [f.code for f in findings] == ["APPLE_GPU_NOT_USED", "CPU_ONLY"]
    assert "Apple M4 GPU is not used" in findings[0].message
    assert "hibiki-asr setup --variant mlx" in findings[0].hint
    assert "No supported GPU" not in findings[1].message


@pytest.mark.parametrize(
    "runtime,reason",
    [
        (rt(), "the MLX runtime is not installed"),
        (rt(mlx_error="ImportError: libmlx.dylib"), "MLX failed to load: ImportError: libmlx.dylib"),
        (rt(mlx_version="0.29.3"), "Metal is not available"),
    ],
)
def test_the_mlx_runtime_without_a_usable_gpu_is_degraded_and_says_why(
    runtime: RuntimeFacts, reason: str
) -> None:
    selection, findings = run(mac(M4), runtime, variant="mlx")
    assert (selection.device, selection.degraded) == ("cpu", True)
    codes = [f.code for f in findings]
    assert codes[0] == "MLX_UNAVAILABLE" and codes[-1] == "DEGRADED_TO_CPU"
    assert reason in findings[0].message and "setup --variant mlx" in findings[0].hint


def test_asking_for_mlx_on_another_machine_is_degraded_not_broken() -> None:
    selection, findings = run(hw(RTX4090), rt(1), device="mlx")
    assert (selection.device, selection.degraded) == ("cpu", True)
    unavailable = next(f for f in findings if f.code == "MLX_UNAVAILABLE")
    assert "only runs on Apple Silicon" in unavailable.message


def test_cuda_and_cpu_machines_are_untouched_by_mlx() -> None:
    assert select_runtime(hw(RTX4090), rt(1)).device == "cuda"
    assert select_runtime(hw(), rt()).device == "cpu"
    assert "mlx" not in rt().compute_types and rt().compute_types["cpu"] == CPU_TYPES


# --- probe and setup --------------------------------------------------------------------------------


def apple_system(commands: dict[str, tuple[int, str]]) -> SystemAccess:
    def run_command(argv, timeout):
        return commands.get(" ".join(argv), (127, ""))

    return SystemAccess(
        platform="darwin",
        machine="arm64",
        cpu_count=lambda: 10,
        env={},
        run=run_command,
        read_text=lambda p: None,
        exists=lambda p: False,
        listdir=lambda p: [],
        which=lambda name: None,
    )


def test_probe_reports_the_apple_gpu_with_the_machine_memory() -> None:
    probe = probe_hardware(
        apple_system(
            {
                "sysctl -n machdep.cpu.brand_string": (0, "Apple M4\n"),
                "sysctl -n hw.memsize": (0, "17179869184\n"),
            }
        )
    )
    assert probe.gpus == [M4]

    bare = probe_hardware(apple_system({}))
    assert [(g.vendor, g.name, g.vram_mb) for g in bare.gpus] == [("apple", "Apple Silicon GPU", None)]


def test_an_intel_mac_has_no_apple_gpu() -> None:
    system = dataclasses.replace(apple_system({}), machine="x86_64")
    assert probe_hardware(system).gpus == []


def test_setup_auto_picks_mlx_on_apple_silicon_only() -> None:
    assert recommend_variant(mac(M4), "darwin").id == "mlx"
    assert recommend_variant(mac(), "darwin").id == "cpu"  # an Intel Mac
    assert recommend_variant(hw(M4), "linux").id == "cpu"
    mlx = get_variant("mlx")
    assert mlx.installable and not mlx.experimental and not mlx.docker and mlx.os == ("darwin",)


# --- with MLX installed (Apple Silicon) ---------------------------------------------------------------


def test_the_mlx_model_gets_exactly_the_weights_in_the_file(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten

    from hibiki_asr.worker.mlx_backend import load_weights

    source = tiny_whisper()
    write_tiny(tmp_path / "model.bin", source)
    model = load_weights(tmp_path, mx.float32)
    got = {name: np.array(value) for name, value in tree_flatten(model.parameters())}

    np.testing.assert_array_equal(got["encoder.conv1.weight"], source["encoder/conv1/weight"].swapaxes(1, 2))
    np.testing.assert_array_equal(
        got["decoder.blocks.0.attn.value.weight"],
        source["decoder/layer_0/self_attention/linear_0/weight"][2 * STATE :],
    )
    np.testing.assert_array_equal(got["decoder.token_embedding.weight"], source["decoder/embeddings/weight"])
    assert (model.dims.n_audio_head, model.dims.n_text_ctx, model.is_multilingual) == (HEADS, TEXT_CTX, True)

    values = tiny_whisper()
    values["encoder/layer_0/ffn/linear_0/weight"] = values["encoder/layer_0/ffn/linear_0/weight"][:, :4]
    write_tiny(tmp_path / "model.bin", values)
    with pytest.raises(UnsupportedModel, match=r"encoder\.blocks\.0\.mlp1\.weight has the shape"):
        load_weights(tmp_path, mx.float32)


def test_repetition_penalty_follows_ctranslate2() -> None:
    mx = pytest.importorskip("mlx.core")
    from hibiki_asr.worker.mlx_backend import RepetitionPenalty

    logits = mx.array([[2.0, -2.0, 1.0, -1.0], [2.0, -2.0, 1.0, -1.0]])
    tokens = mx.array([[0, 1], [3, 3]])
    out = np.array(RepetitionPenalty(2.0, 4).apply(logits, tokens))
    np.testing.assert_allclose(out, [[1.0, -4.0, 1.0, -1.0], [2.0, -2.0, 1.0, -2.0]])


def test_generation_options_are_translated_for_mlx_whisper(tmp_path: Path, caplog) -> None:
    pytest.importorskip("mlx_whisper")
    from hibiki_asr.worker.mlx_backend import MlxWhisper

    write_tiny(tmp_path / "model.bin", tiny_whisper())
    model = MlxWhisper(tmp_path, "float32")
    options, penalty = model._options(
        dict(
            language="ja",
            task="translate",
            beam_size=5,
            repetition_penalty=1.1,
            log_prob_threshold=-0.5,
            temperature=[0.0, 0.2],
            condition_on_previous_text=False,
            max_initial_timestamp=30,
            vad_filter=False,
            no_repeat_ngram_size=3,
            initial_prompt=None,
        )
    )
    assert penalty == 1.1
    assert options == dict(
        language="ja",
        task="translate",
        logprob_threshold=-0.5,
        temperature=(0.0, 0.2),
        condition_on_previous_text=False,
        max_initial_timestamp=30,
        initial_prompt=None,
    )
    assert (
        "no beam search" in caplog.text
        and "ignores the generation option 'no_repeat_ngram_size'" in caplog.text
    )

    options, _ = model._options(
        dict(
            compression_ratio_threshold=None,
            log_prob_threshold=None,
            no_speech_threshold=None,
        )
    )
    assert options == dict(compression_ratio_threshold=None, logprob_threshold=None, no_speech_threshold=None)

    with pytest.raises(ValueError, match="not available on MLX"):
        MlxWhisper(tmp_path, "int8")
