"""Lightweight live VRAM probing without importing the inference runtime.

Use the minimum across visible physical adapters: CUDA's logical device ordering
need not match nvidia-smi. Unknown or partitioned devices must remain serial.
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from ..diagnostics.system import SystemAccess


def free_vram_mb(device: str) -> int | None:
    if device == "cpu":
        return None
    system = SystemAccess()
    if device == "mlx" or (device == "auto" and system.is_apple_silicon()):
        return _mac_available_mb(system) if system.is_apple_silicon() else None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (not visible.strip() or visible == "-1" or "MIG-" in visible):
        return None
    if device != "amd":
        code, output = system.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], 2.0
        )
        if code == 0:
            try:
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                return min(values) if values and min(values) >= 0 else None
            except ValueError:
                return None
    if device == "cuda":
        return None
    # Linux AMD exposes live VRAM usage through amdgpu sysfs.
    values = []
    for card in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            total = int((card / "mem_info_vram_total").read_text())
            used = int((card / "mem_info_vram_used").read_text())
            values.append(max(0, total - used) // (1024 * 1024))
        except (OSError, ValueError):
            continue
    return min(values) if values else None


_MLX_MARKER = "HIBIKI_ASR_MLX_MEMORY="


def print_mlx_memory_limits() -> None:
    """Isolated helper: importing MLX never touches the API process's Metal state."""
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable")
    info = mx.metal.device_info()
    print(_MLX_MARKER + json.dumps([info["memory_size"], info["max_recommended_working_set_size"]]))


@lru_cache(maxsize=1)
def _mlx_memory_limits() -> tuple[int, int]:
    """Cache hardware limits, never live usage. Failed probes are not cached."""
    command = (
        [sys.executable, "--mlx-memory-probe"]
        if getattr(sys, "frozen", False)
        else [sys.executable, "-m", "hibiki_asr.jobs.memory"]
    )
    code, output = SystemAccess().run(command, 5.0)
    if code == 0:
        for line in reversed(output.splitlines()):
            if line.startswith(_MLX_MARKER):
                limits = json.loads(line[len(_MLX_MARKER) :])
                if (
                    isinstance(limits, list)
                    and len(limits) == 2
                    and all(type(value) is int and value > 0 for value in limits)
                    and limits[1] <= limits[0]
                ):
                    return limits[0], limits[1]
    raise ValueError("MLX memory limits are unavailable")


def _mac_available_mb(system: SystemAccess) -> int | None:
    try:
        total, recommended = _mlx_memory_limits()
    except (ValueError, OSError):
        return None
    code, output = system.run(["/usr/bin/vm_stat"], 2.0)
    if code != 0:
        return None
    page_size = re.search(r"page size of (\d+) bytes", output)
    if page_size is None or int(page_size[1]) <= 0:
        return None
    counts = []
    for name in ("free", "inactive", "speculative"):
        match = re.search(rf"^Pages {name}:\s+(\d+)\.\s*$", output, re.MULTILINE)
        if match is None:
            return None
        counts.append(int(match[1]))
    available = sum(counts) * int(page_size[1])
    if available > total:
        return None
    # Unified memory is shared across all workers and applications. Charge all
    # non-reclaimable system use against the recommended Metal working set as
    # a conservative bound; MLX's get_active_memory only sees one process.
    used = total - available
    return max(0, min(available, recommended - used)) // (1024 * 1024)


if __name__ == "__main__":
    print_mlx_memory_limits()
