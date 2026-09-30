"""The human readable report and the start-up log lines."""

from __future__ import annotations

import logging

from helpers import RTX4090, hw, rt
from hibiki_asr.diagnostics.findings import evaluate_findings
from hibiki_asr.diagnostics.report import format_report, log_findings
from hibiki_asr.diagnostics.schema import Diagnostics, RuntimeFacts
from hibiki_asr.diagnostics.selection import select_runtime


def diagnostics(probe, runtime, *, variant=None, device="auto") -> Diagnostics:
    selection = select_runtime(probe, runtime, device, "auto", variant)
    return Diagnostics(
        api_version=1,
        engine_version="9.9.9",
        variant=variant,
        hardware=probe,
        runtime=runtime,
        selection=selection,
        findings=evaluate_findings(probe, runtime, selection, variant=variant),
        settings={},
    )


def test_report_for_a_healthy_gpu_machine() -> None:
    runtime = rt(1, onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    text = format_report(diagnostics(hw(RTX4090), runtime, variant="cuda12"))
    assert "hibiki-asr 9.9.9 (api 1), runtime variant: cuda12" in text
    assert "NVIDIA GeForce RTX 4090 (driver 555.42.06, 24564 MB, compute capability 8.9)" in text
    assert "Device   cuda (bfloat16), VAD on cuda" in text and "No problems found." in text
    assert "<--" not in text


def test_report_for_a_gpu_container_without_the_gpu_explains_the_fix() -> None:
    text = format_report(diagnostics(hw(container=True), rt(0), variant="cuda12"))
    assert "in a container" in text and "GPU      none detected" in text
    assert "<-- a GPU was expected, running on the CPU instead" in text
    assert "[WARNING] NO_GPU_VISIBLE" in text and "gpus: all" in text  # the fix is in the report
    assert all(len(line) <= 106 for line in text.splitlines())  # wrapped for a terminal


def test_report_when_the_runtime_cannot_be_inspected() -> None:
    failed = RuntimeFacts(probe_ok=False, probe_error="runtime probe timed out after 60s")
    text = format_report(diagnostics(hw(), failed))
    assert "could not be inspected: runtime probe timed out" in text and "RUNTIME_PROBE_FAILED" in text


def test_startup_log_levels_follow_severity(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="t"):
        log_findings(logging.getLogger("t"), diagnostics(hw(container=True), rt(0), variant="cuda12"))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("running on the CPU although a GPU was expected" in m for m in warnings)
    assert any(
        m.startswith("NO_GPU_VISIBLE") and "| fix:" in m for m in warnings
    )  # the cause and its fix are logged


def test_startup_log_for_a_normal_cpu_run_is_calm(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="t"):
        log_findings(logging.getLogger("t"), diagnostics(hw(), rt(0)))
    assert {r.levelno for r in caplog.records} == {logging.INFO}
