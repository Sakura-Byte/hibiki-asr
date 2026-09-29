"""Hardware probe. Standard library only, so it works before the heavy runtime is installed."""

from __future__ import annotations

import platform as _platform
import re

from .schema import GpuInfo, HardwareProbe
from .system import SystemAccess

# Environment variables that change which GPU (if any) the runtime can see.
GPU_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
)

# Windows only reports a marketing name. Map it to the gfx target the ROCm builds are keyed on.
# Order matters: the first match wins.
_AMD_NAME_TO_GFX: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), gfx)
    for pattern, gfx in (
        (r"\bRX\s*9070\b", "gfx1201"),  # Navi 48
        (r"\bRX\s*906\d\b", "gfx1200"),  # Navi 44
        (r"\bRX\s*7900\b", "gfx1100"),  # Navi 31
        (r"\bRX\s*7[78]00\b", "gfx1101"),  # Navi 32
        (r"\bRX\s*76\d0\b", "gfx1102"),  # Navi 33
        (r"\bRX\s*6[89][05]0\b", "gfx1030"),  # Navi 21
        (r"\bRX\s*67[05]0\b", "gfx1031"),  # Navi 22
        (r"\bRX\s*6[45]00\b", "gfx1034"),  # Navi 24
        (r"\bRX\s*6[56][05]0\b", "gfx1032"),  # Navi 23
        (r"\bRX\s*5[67]00\b", "gfx1010"),  # Navi 10
        (r"\bRX\s*5[345]00\b", "gfx1012"),  # Navi 14
        (r"\b(740M|760M|780M)\b", "gfx1103"),  # Phoenix iGPU
        (r"\b(880M|890M)\b", "gfx1150"),  # Strix iGPU
    )
)
_INTEGRATED_NAME = re.compile(r"\b(\d{3}M|Radeon\(TM\) Graphics|Radeon Graphics|Vega \d+)\b", re.IGNORECASE)


def _detect_container(system: SystemAccess) -> bool:
    if system.exists("/.dockerenv") or system.exists("/run/.containerenv"):
        return True
    if system.env.get("container"):
        return True
    cgroup = system.read_text("/proc/1/cgroup") or ""
    return any(token in cgroup for token in ("docker", "containerd", "kubepods", "libpod"))


def _cpu_info(system: SystemAccess) -> tuple[str | None, bool | None]:
    """(model name, AVX2 support). AVX2 is only known on Linux."""
    model = _platform.processor() or None
    avx2: bool | None = None
    cpuinfo = system.read_text("/proc/cpuinfo") if system.is_linux() else None
    if cpuinfo:
        name = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if name:
            model = name.group(1).strip()
        flags = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if flags:
            avx2 = "avx2" in flags.group(1).split()
    return model, avx2


def _probe_nvidia(system: SystemAccess) -> tuple[list[GpuInfo], bool]:
    fields = "name,driver_version,memory.total"
    for query in (f"{fields},compute_cap", fields):
        code, out = system.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], 10.0)
        if code == 127:
            return [], False
        if code == 0 and out.strip():
            break
    else:
        # nvidia-smi exists but fails (no driver loaded, NVML mismatch, ...).
        return [], True

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        vram = int(parts[2]) if parts[2].isdigit() else None
        cc = parts[3] if len(parts) > 3 and re.fullmatch(r"\d+\.\d+", parts[3]) else None
        gpus.append(
            GpuInfo(vendor="nvidia", name=parts[0], driver=parts[1] or None, vram_mb=vram, compute_capability=cc)
        )
    return gpus, True


def _gfx_from_kfd_version(version: int) -> str:
    """KFD stores the target as major*10000 + minor*100 + stepping, e.g. 110000 -> gfx1100."""
    major, minor, step = version // 10000, (version // 100) % 100, version % 100
    return f"gfx{major}{minor:x}{step:x}"


def _pci_display_names(system: SystemAccess, vendor_pattern: str) -> list[str]:
    """Names of PCI display adapters from ``lspci`` whose description matches ``vendor_pattern``."""
    code, out = system.run(["lspci"], 10.0)
    if code != 0:
        return []
    return [
        line.split(": ", 1)[-1].strip()
        for line in out.splitlines()
        if re.search(r"VGA|Display|3D", line) and re.search(vendor_pattern, line, re.IGNORECASE)
    ]


def _amd_linux(system: SystemAccess) -> list[GpuInfo]:
    gfx_targets: list[str] = []

    code, out = system.run(["rocm_agent_enumerator"], 10.0)
    if code == 0:
        gfx_targets = [t for t in out.split() if t.startswith("gfx") and t != "gfx000"]

    if not gfx_targets:
        # The amdgpu kernel driver publishes the target without any ROCm userspace.
        base = "/sys/class/kfd/kfd/topology/nodes"
        for node in system.listdir(base):
            props = system.read_text(f"{base}/{node}/properties") or ""
            match = re.search(r"^gfx_target_version\s+(\d+)$", props, re.MULTILINE)
            if match and int(match.group(1)) > 0:
                gfx_targets.append(_gfx_from_kfd_version(int(match.group(1))))

    names = _pci_display_names(system, r"AMD|ATI|Advanced Micro Devices")

    return [
        GpuInfo(
            vendor="amd",
            name=names[i] if i < len(names) else f"AMD GPU ({gfx})",
            gfx=gfx,
            integrated=bool(re.match(r"gfx115\d|gfx90c", gfx)) or None,
        )
        for i, gfx in enumerate(gfx_targets)
    ]


def _gpus_windows(system: SystemAccess) -> list[GpuInfo]:
    """Display adapters from the WMI video controller list."""
    names: list[str] = []
    code, out = system.run(
        ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
        15.0,
    )
    if code == 0:
        names = [line.strip() for line in out.splitlines() if line.strip()]
    else:
        code, out = system.run(["wmic", "path", "win32_VideoController", "get", "name"], 15.0)
        if code == 0:
            names = [line.strip() for line in out.splitlines()[1:] if line.strip()]

    gpus: list[GpuInfo] = []
    for name in names:
        if re.search(r"NVIDIA", name, re.IGNORECASE):
            gpus.append(GpuInfo(vendor="nvidia", name=name))
        elif re.search(r"AMD|ATI|Radeon", name, re.IGNORECASE):
            gfx = next((target for pattern, target in _AMD_NAME_TO_GFX if pattern.search(name)), None)
            gpus.append(GpuInfo(vendor="amd", name=name, gfx=gfx, integrated=bool(_INTEGRATED_NAME.search(name))))
        elif re.search(r"Intel", name, re.IGNORECASE):
            gpus.append(GpuInfo(vendor="intel", name=name))
    return gpus


def probe_hardware(system: SystemAccess | None = None) -> HardwareProbe:
    system = system or SystemAccess()

    cpu_model, avx2 = _cpu_info(system)
    nvidia, smi_found = _probe_nvidia(system)

    gpus: list[GpuInfo] = list(nvidia)
    kfd_present: bool | None = None
    if system.is_linux():
        kfd_present = system.exists("/dev/kfd")
        gpus.extend(_amd_linux(system))
        if not nvidia:
            # The card is there but nvidia-smi is not: report it so the missing driver can be diagnosed.
            gpus.extend(GpuInfo(vendor="nvidia", name=n) for n in _pci_display_names(system, r"NVIDIA"))
    elif system.is_windows():
        # nvidia-smi is the authority for NVIDIA cards; adapter names only fill in when it is unusable.
        gpus.extend(g for g in _gpus_windows(system) if g.vendor != "nvidia" or not nvidia)

    return HardwareProbe(
        os=system.platform,
        arch=system.machine,
        in_container=_detect_container(system),
        cpu_model=cpu_model,
        cpu_cores=system.cpu_count(),
        cpu_avx2=avx2,
        gpus=gpus,
        nvidia_smi_found=smi_found,
        kfd_present=kfd_present,
        env={name: system.env[name] for name in GPU_ENV_VARS if name in system.env},
    )
