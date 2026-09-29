"""Rules that explain why the engine is (or is not) using the GPU, and what to do about it.

Every rule is a pure function of the probed hardware, the runtime facts and the
selection, so they are tested with tables instead of real machines. A finding
always carries a stable ``code``; the message and hint are for humans and contain
commands that can be pasted as they are.
"""

from __future__ import annotations

from ..provision.variants import (
    Variant,
    driver_major,
    get_variant,
    load_variants,
    parse_cc,
    recommend_variant,
)
from .schema import Finding, GpuInfo, HardwareProbe, RuntimeFacts, Selection, Severity
from .selection import LOW_VRAM_MB, compute_gpu_expected, is_gpu_variant

IMAGE = "ghcr.io/sakura-byte/hibiki-asr"

_ORDER = {Severity.error: 0, Severity.warning: 1, Severity.info: 2}


def _switch_hint(target: Variant, probe: HardwareProbe) -> str:
    if probe.in_container:
        return (
            f"This engine is running in a container built for a different runtime. "
            f"Use the {target.id} image instead: {IMAGE}:latest-{target.id}"
        )
    return f"Install the {target.title} runtime with `hibiki-asr setup --variant {target.id}` and restart the engine."


def _gpus(probe: HardwareProbe, vendor: str) -> list[GpuInfo]:
    return [g for g in probe.gpus if g.vendor == vendor]


def evaluate_findings(
    probe: HardwareProbe,
    runtime: RuntimeFacts,
    selection: Selection,
    *,
    variant: str | None = None,
    requested_compute: str = "auto",
) -> list[Finding]:
    """All findings for this machine, most severe first."""
    out: list[Finding] = []
    add = out.append

    recommended = recommend_variant(probe, probe.os)
    nvidia = _gpus(probe, "nvidia")
    amd = _gpus(probe, "amd")
    amd_discrete = [g for g in amd if not g.integrated]
    gpu_wanted = selection.requested_device != "cpu"
    gpu_missing = gpu_wanted and runtime.cuda_device_count == 0  # the runtime sees no GPU at all
    gpu_expected = compute_gpu_expected(probe, selection.requested_device, variant)

    # ---- the runtime itself ----------------------------------------------------------------
    if not runtime.probe_ok:
        add(
            Finding(
                code="RUNTIME_PROBE_FAILED",
                severity=Severity.error,
                message=f"Could not inspect the inference runtime: {runtime.probe_error}",
                hint="Run `hibiki-asr doctor` in a terminal to see the full error. The runtime may be missing or corrupt: "
                f"reinstall it with `hibiki-asr setup --variant {recommended.id}`.",
            )
        )
    if runtime.probe_ok and runtime.ctranslate2_version is None:
        text = runtime.ctranslate2_error or "unknown error"
        libs = any(
            token in text.lower() for token in ("cudnn", "cublas", "cudart", "cuda", "hip", "dll", "libcu")
        )
        add(
            Finding(
                code="CT2_IMPORT_FAILED",
                severity=Severity.error,
                message=f"CTranslate2 failed to load: {text}",
                hint=("A GPU runtime library is missing. " if libs else "")
                + f"Reinstall the runtime: `hibiki-asr setup --variant {recommended.id}`.",
            )
        )
    if runtime.probe_ok and runtime.faster_whisper_version is None:
        add(
            Finding(
                code="FASTER_WHISPER_MISSING",
                severity=Severity.error,
                message=f"faster-whisper failed to import: {runtime.faster_whisper_error or 'unknown error'}",
                hint=f"Run `hibiki-asr setup --variant {recommended.id}` to install the inference stack.",
            )
        )
    if runtime.probe_ok and runtime.onnxruntime_version is None:
        add(
            Finding(
                code="ORT_UNAVAILABLE",
                severity=Severity.error,
                message=f"onnxruntime failed to import, so voice activity detection cannot run: {runtime.onnxruntime_error}",
                hint=f"Run `hibiki-asr setup --variant {recommended.id}` to reinstall it.",
            )
        )

    # ---- the environment hides the GPU ------------------------------------------------------
    gpu_hardware = bool(nvidia or amd_discrete)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if gpu_wanted and gpu_hardware and probe.env.get(name) in ("", "-1"):
            add(
                Finding(
                    code="CUDA_DISABLED_BY_ENV",
                    severity=Severity.warning,
                    message=f"{name} is set to '{probe.env[name]}', which hides every GPU from the engine.",
                    hint=f"Unset {name} (or set it to the GPU index, e.g. {name}=0) and restart the engine.",
                )
            )

    # ---- no GPU at all is visible, although one is expected ------------------------------------------
    # The classic Docker mistake: a CUDA/ROCm image started without the GPU passed in shows an empty machine.
    if (
        gpu_missing
        and not gpu_hardware
        and probe.gpus == []
        and (selection.requested_device == "cuda" or is_gpu_variant(variant))
    ):
        wanted = (
            "a CUDA/ROCm runtime is installed" if is_gpu_variant(variant) else "the device is set to 'cuda'"
        )
        add(
            Finding(
                code="NO_GPU_VISIBLE",
                severity=Severity.warning,
                message=f"No NVIDIA or AMD GPU is visible to the engine, although {wanted}.",
                hint=(
                    "Pass the GPU into the container. NVIDIA (needs the NVIDIA Container Toolkit on the host): `gpus: all` in "
                    "docker-compose, or `docker run --gpus all`. AMD: `devices: [/dev/kfd, /dev/dri]`, `group_add: [video, render]` "
                    "and `security_opt: [seccomp=unconfined]`. If this machine has no GPU, use the CPU image or set HIBIKI_ASR_DEVICE=cpu."
                    if probe.in_container
                    else "Check that the GPU and its driver are installed and working (`nvidia-smi` on NVIDIA, `rocminfo` on AMD). "
                    "If this machine has no GPU, set HIBIKI_ASR_DEVICE=cpu or install the CPU runtime: `hibiki-asr setup --variant cpu`."
                ),
            )
        )

    # ---- NVIDIA -----------------------------------------------------------------------------
    if nvidia and gpu_missing:
        if not probe.nvidia_smi_found or all(g.driver is None for g in nvidia):
            add(
                Finding(
                    code="NVIDIA_DRIVER_MISSING",
                    severity=Severity.error,
                    message=f"An NVIDIA GPU ({nvidia[0].name}) is present but the driver is not usable.",
                    hint=(
                        "Install the NVIDIA Container Toolkit on the host and start this container with GPU access "
                        "(`docker run --gpus all ...`, or `gpus: all` in docker-compose)."
                        if probe.in_container
                        else "Install the NVIDIA driver from https://www.nvidia.com/Download/index.aspx, "
                        "reboot, and check that `nvidia-smi` works."
                    ),
                )
            )
        elif runtime.ctranslate2_version is not None:
            installed = load_variants().get(variant or "")
            driver = driver_major(nvidia[0].driver)
            if installed and installed.min_driver and driver is not None and driver < installed.min_driver:
                older = get_variant("cuda11") if driver >= (get_variant("cuda11").min_driver or 0) else None
                add(
                    Finding(
                        code="NVIDIA_DRIVER_TOO_OLD",
                        severity=Severity.error,
                        message=f"NVIDIA driver {nvidia[0].driver} is older than the {installed.min_driver} that the "
                        f"'{installed.id}' runtime needs.",
                        hint="Update the NVIDIA driver."
                        + (
                            f" Or switch to the older runtime: `hibiki-asr setup --variant {older.id}`."
                            if older
                            else ""
                        ),
                    )
                )
            elif not any(f.code == "CUDA_DISABLED_BY_ENV" for f in out):
                add(
                    Finding(
                        code="CT2_NO_GPU_SUPPORT",
                        severity=Severity.warning,
                        message="An NVIDIA GPU and driver are present, but CTranslate2 sees no CUDA device.",
                        hint=(
                            "The engine container has no GPU access: start it with `gpus: all` (docker-compose) or "
                            "`--gpus all`, and check that the image is a CUDA one. "
                            if probe.in_container
                            else "This is usually a CPU-only install. "
                        )
                        + _switch_hint(recommended, probe),
                    )
                )

    for gpu in nvidia:
        cc = parse_cc(gpu.compute_capability)
        if cc and cc >= (12, 0) and variant and variant != "cuda12-blackwell" and variant != "cpu":
            add(
                Finding(
                    code="BLACKWELL_NEEDS_CU128",
                    severity=Severity.warning,
                    message=f"{gpu.name} (compute capability {gpu.compute_capability}) needs a CUDA 12.8 build.",
                    hint=_switch_hint(get_variant("cuda12-blackwell"), probe),
                )
            )
            break

    # ---- AMD --------------------------------------------------------------------------------
    if amd and not nvidia and gpu_missing:
        if amd and not amd_discrete:
            add(
                Finding(
                    code="AMD_IGPU_UNSUPPORTED",
                    # Only a warning when a GPU was actually asked for; under 'auto' it is just useful to know.
                    severity=Severity.warning if gpu_expected else Severity.info,
                    message=f"{amd[0].name} is an integrated GPU, which the ROCm builds do not support.",
                    hint="The CPU is used. Set HIBIKI_ASR_DEVICE=cpu to make that explicit and silence this warning.",
                )
            )
        elif probe.kfd_present is False:
            add(
                Finding(
                    code="AMD_DEVICE_NOT_VISIBLE",
                    severity=Severity.error,
                    message="An AMD GPU is present but /dev/kfd (the ROCm compute device) is not available.",
                    hint=(
                        "Give the container the devices: `--device=/dev/kfd --device=/dev/dri --group-add video "
                        "--group-add render --security-opt seccomp=unconfined` (docker-compose: `devices`, `group_add`, "
                        "`security_opt`)."
                        if probe.in_container
                        else "Install the amdgpu driver and ROCm, add your user to the `video` and `render` groups, "
                        "and check that /dev/kfd exists."
                    ),
                )
            )
        elif runtime.ctranslate2_version is not None:
            add(
                Finding(
                    code="AMD_HIP_WHEEL_MISSING",
                    severity=Severity.warning,
                    message=f"An AMD GPU ({amd_discrete[0].name}) is present but this CTranslate2 build has no ROCm/HIP support.",
                    hint=(
                        _switch_hint(recommended, probe)
                        if recommended.gpu_vendor == "amd"
                        else "No ROCm build matches this GPU (gfx target "
                        f"{amd_discrete[0].gfx or 'unknown'}); the CPU is used."
                    ),
                )
            )

    for gpu in amd_discrete:
        if probe.os.startswith("win") and gpu.gfx and gpu.gfx.startswith("gfx103"):
            add(
                Finding(
                    code="AMD_RDNA2_RUNTIME_CRASH",
                    severity=Severity.info,
                    message=f"{gpu.name} (RDNA2, {gpu.gfx}) has known crashes with the bundled ROCm 7.x runtime.",
                    hint="If the engine crashes when loading the model, install HIP SDK 6.4.2 and replace "
                    "amdhip64_7.dll with amdhip64_6.dll in the engine's runtime folder.",
                )
            )
            break

    # ---- what was selected ------------------------------------------------------------------
    if selection.device == "cpu" and not selection.degraded:
        add(
            Finding(
                code="CPU_ONLY",
                severity=Severity.info,
                message="Running on the CPU."
                + ("" if selection.requested_device == "cpu" else " No supported GPU was detected."),
                hint="Large models run slower than real time on a CPU; a supported GPU makes a big difference.",
            )
        )
    if selection.device == "cpu" and probe.cpu_avx2 is False:
        add(
            Finding(
                code="CPU_NO_AVX2",
                severity=Severity.warning,
                message="This CPU has no AVX2 support; CPU inference will be extremely slow or may not start.",
                hint="Use a GPU, or run the engine on a newer machine.",
            )
        )

    primary = next(iter(nvidia or amd_discrete), None)
    if (
        selection.device == "cuda"
        and primary
        and primary.vram_mb is not None
        and primary.vram_mb < LOW_VRAM_MB
    ):
        add(
            Finding(
                code="VRAM_LOW",
                severity=Severity.info,
                message=f"{primary.name} has {primary.vram_mb} MB of VRAM; using {selection.compute_type} to save memory.",
                hint="Close other programs that use the GPU. If loading still fails, the engine retries on the CPU.",
            )
        )
    if (
        requested_compute not in ("auto", "default")
        and selection.compute_type != requested_compute
        and runtime.ctranslate2_version is not None
    ):
        add(
            Finding(
                code="COMPUTE_TYPE_DOWNGRADED",
                severity=Severity.info,
                message=f"compute_type '{requested_compute}' is not supported on {selection.device}; using "
                f"'{selection.compute_type}'.",
                hint=f"Set HIBIKI_ASR_COMPUTE_TYPE=auto or one of: {', '.join(runtime.compute_types.get(selection.device, []))}.",
            )
        )
    if (
        selection.device == "cuda"
        and selection.vad_device == "cpu"
        and runtime.onnxruntime_version is not None
    ):
        add(
            Finding(
                code="VAD_ON_CPU",
                severity=Severity.info,
                message="Voice activity detection runs on the CPU (onnxruntime has no CUDA provider).",
                hint="Expected with ROCm builds. On NVIDIA it only costs a little time; the transcription itself uses the GPU.",
            )
        )

    # ---- summary: never leave a degraded state without a warning -----------------------------
    if selection.degraded:
        add(
            Finding(
                code="DEGRADED_TO_CPU",
                severity=Severity.warning,
                message="A GPU was expected but the CPU is being used, so transcription will be much slower.",
                hint=(
                    "The findings above say why. Fix the first error or warning and restart the engine; "
                    if out
                    else "Check that the GPU is visible to the engine and restart it; "
                )
                + "set HIBIKI_ASR_DEVICE=cpu if the CPU is what you want.",
            )
        )

    return sorted(out, key=lambda f: _ORDER[f.severity])
