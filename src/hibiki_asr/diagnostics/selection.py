"""Choose the device and precision, given what the machine has and what the runtime supports."""

from __future__ import annotations

from .schema import GpuInfo, HardwareProbe, RuntimeFacts, Selection

# Same preference order as upstream: best quality that is still fast and small.
COMPUTE_PREFERENCE = (
    "bfloat16",
    "float16",
    "int16",
    "int8_bfloat16",
    "int8_float16",
    "int8_float32",
    "int8",
    "float32",
)
# On CPU the fp16/bf16 weights are converted anyway; int8 keeps a 1.5B model near 1.5 GB instead of 6 GB.
CPU_PREFERENCE = ("int8", "int8_float32", "int16", "float32")

# Below this much VRAM the half precision weights plus activations do not fit comfortably.
LOW_VRAM_MB = 6_000

_GPU_ALIASES = {"amd", "rocm", "hip"}
DEVICE_CHOICES = ("auto", "cpu", "cuda", "amd")


def normalize_device(requested: str | None) -> str:
    """``auto|cpu|cuda``. The HIP backend is exposed to CTranslate2 as ``cuda`` too."""
    value = (requested or "auto").strip().lower()
    if value in _GPU_ALIASES:
        return "cuda"
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device {requested!r}; expected one of {', '.join(DEVICE_CHOICES)}")
    return value


def compute_gpu_expected(probe: HardwareProbe, requested: str) -> bool:
    """Whether the user should reasonably expect a GPU to be used."""
    if requested == "cpu":
        return False
    if requested == "cuda":
        return True
    return any(g.vendor in ("nvidia", "amd") and not g.integrated for g in probe.gpus)


def _gpu_env_disables(probe: HardwareProbe) -> bool:
    return any(probe.env.get(name) in ("", "-1") for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"))


def gpu_usable(probe: HardwareProbe, runtime: RuntimeFacts) -> bool:
    return runtime.cuda_device_count > 0 and not _gpu_env_disables(probe)


def _primary_gpu(probe: HardwareProbe) -> GpuInfo | None:
    return next((g for g in probe.gpus if g.vendor in ("nvidia", "amd")), None)


def pick_compute_type(device: str, supported: list[str], requested: str, vram_mb: int | None) -> str:
    """Honour an explicit, supported choice; otherwise the best supported type for the device."""
    if requested not in ("auto", "default") and requested in supported:
        return requested

    preference = CPU_PREFERENCE if device == "cpu" else COMPUTE_PREFERENCE
    if device == "cuda" and vram_mb is not None and vram_mb < LOW_VRAM_MB:
        preference = ("int8_float16", "int8_bfloat16", *preference)

    for candidate in preference:
        if candidate in supported:
            return candidate
    # The runtime could not report anything (broken install); pick a value that loads everywhere.
    return "int8" if device == "cpu" else "float16"


def select_runtime(
    probe: HardwareProbe, runtime: RuntimeFacts, requested_device: str = "auto", requested_compute: str = "auto"
) -> Selection:
    requested = normalize_device(requested_device)
    expected = compute_gpu_expected(probe, requested)
    use_gpu = requested != "cpu" and gpu_usable(probe, runtime)
    device = "cuda" if use_gpu else "cpu"

    gpu = _primary_gpu(probe)
    compute = pick_compute_type(
        device, runtime.compute_types.get(device, []), requested_compute, gpu.vram_mb if gpu else None
    )
    vad_device = "cuda" if use_gpu and "CUDAExecutionProvider" in runtime.onnxruntime_providers else "cpu"

    return Selection(
        requested_device=requested,
        device=device,
        compute_type=compute,
        degraded=expected and not use_gpu,
        vad_device=vad_device,
    )
