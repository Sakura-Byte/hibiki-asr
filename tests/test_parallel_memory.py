import sys

import pytest

from hibiki_asr.diagnostics.system import SystemAccess
from hibiki_asr.jobs import memory
from hibiki_asr.jobs.memory import free_vram_mb


def test_nvidia_uses_minimum_free_memory(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "hibiki_asr.jobs.memory.SystemAccess", lambda: SystemAccess(run=lambda *_: (0, "12000\n6000\n"))
    )
    assert free_vram_mb("cuda") == 6000


def test_unavailable_or_invalid_memory_is_unknown(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for answer in ((127, ""), (124, ""), (0, "[N/A]"), (0, ""), (0, "-1")):
        monkeypatch.setattr(
            "hibiki_asr.jobs.memory.SystemAccess", lambda answer=answer: SystemAccess(run=lambda *_: answer)
        )
        assert free_vram_mb("cuda") is None


def test_hidden_or_partitioned_cuda_devices_remain_serial(monkeypatch):
    for visible in ("", "-1", "MIG-example"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert free_vram_mb("cuda") is None
    assert free_vram_mb("cpu") is None
    monkeypatch.setattr("hibiki_asr.jobs.memory.SystemAccess", lambda: SystemAccess(platform="win32"))
    assert free_vram_mb("mlx") is None


GIB = 1024**3


def mac_vm_stat(available_gib=20, page_size=16384):
    return (
        f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
        f"Pages free: {int(available_gib * GIB / page_size) - 300}.\n"
        "Pages inactive: 200.\nPages speculative: 100.\n"
        "Pages purgeable: 50.\nPages occupied by compressor: 90000.\n"
    )


@pytest.mark.parametrize("device", ["auto", "mlx"])
@pytest.mark.parametrize("page_size", [4096, 16384])
def test_mac_accounts_for_system_usage_and_metal_budget(monkeypatch, device, page_size):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # unrelated to Metal
    monkeypatch.setattr(memory, "_mlx_memory_limits", lambda: (32 * GIB, 24 * GIB))
    sample = {"available": 20}

    def run(command, timeout):
        assert command == ["/usr/bin/vm_stat"]
        return 0, mac_vm_stat(sample["available"], page_size)

    monkeypatch.setattr(
        memory, "SystemAccess", lambda: SystemAccess(platform="darwin", machine="arm64", run=run)
    )
    assert free_vram_mb(device) == 12 * 1024
    sample["available"] = 10
    assert free_vram_mb(device) == 2 * 1024  # live pressure is not cached
    sample["available"] = 4
    assert free_vram_mb(device) == 0


@pytest.mark.parametrize(
    "output",
    ["", "page size of 0 bytes", mac_vm_stat().replace("Pages inactive:", "missing:"), mac_vm_stat(64)],
)
def test_mac_invalid_stats_do_not_admit_extra_jobs(monkeypatch, output):
    monkeypatch.setattr(memory, "_mlx_memory_limits", lambda: (32 * GIB, 24 * GIB))
    monkeypatch.setattr(
        memory,
        "SystemAccess",
        lambda: SystemAccess(platform="darwin", machine="arm64", run=lambda *_: (0, output)),
    )
    assert free_vram_mb("mlx") is None


def test_mac_probe_failure_does_not_admit_extra_jobs(monkeypatch):
    def unavailable():
        raise ValueError("probe failed")

    monkeypatch.setattr(memory, "_mlx_memory_limits", unavailable)
    monkeypatch.setattr(memory, "SystemAccess", lambda: SystemAccess(platform="darwin", machine="arm64"))
    assert free_vram_mb("mlx") is None


@pytest.mark.parametrize("frozen", [False, True])
def test_mlx_limits_are_isolated_and_success_is_cached(monkeypatch, frozen):
    memory._mlx_memory_limits.cache_clear()
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    calls = []

    def run(command, timeout):
        calls.append(command)
        return 0, memory._MLX_MARKER + f"[{32 * GIB}, {24 * GIB}]"

    monkeypatch.setattr(memory, "SystemAccess", lambda: SystemAccess(run=run))
    try:
        assert memory._mlx_memory_limits() == (32 * GIB, 24 * GIB)
        assert memory._mlx_memory_limits() == (32 * GIB, 24 * GIB)
        expected = (
            [sys.executable, "--mlx-memory-probe"]
            if frozen
            else [sys.executable, "-m", "hibiki_asr.jobs.memory"]
        )
        assert calls == [expected]
    finally:
        memory._mlx_memory_limits.cache_clear()


def test_mlx_failed_limits_probe_is_retried(monkeypatch):
    memory._mlx_memory_limits.cache_clear()
    replies = iter([(124, ""), (0, memory._MLX_MARKER + "[1000, 800]")])
    monkeypatch.setattr(memory, "SystemAccess", lambda: SystemAccess(run=lambda *_: next(replies)))
    try:
        with pytest.raises(ValueError):
            memory._mlx_memory_limits()
        assert memory._mlx_memory_limits() == (1000, 800)
    finally:
        memory._mlx_memory_limits.cache_clear()


def test_frozen_entry_dispatches_mlx_memory_probe(monkeypatch, tmp_path):
    from hibiki_asr import standalone

    (tmp_path / "bundle.json").write_text('{"variant":"mlx"}')
    monkeypatch.setattr(standalone, "__file__", str(tmp_path / "standalone.py"))
    monkeypatch.setattr(sys, "argv", ["hibiki-asr", "--mlx-memory-probe"])
    monkeypatch.setenv("HIBIKI_ASR_VARIANT", "mlx")
    called = []
    monkeypatch.setattr(memory, "print_mlx_memory_limits", lambda: called.append(True))
    assert standalone.main() == 0
    assert called == [True]
