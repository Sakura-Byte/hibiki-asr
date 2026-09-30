"""Human readable diagnostics, for `hibiki-asr doctor` and the engine's start-up log."""

from __future__ import annotations

import logging
import textwrap

from .schema import Diagnostics, Finding, Severity

_MARK = {Severity.error: "ERROR  ", Severity.warning: "WARNING", Severity.info: "info   "}
_LEVEL = {Severity.error: logging.ERROR, Severity.warning: logging.WARNING, Severity.info: logging.INFO}


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def format_report(d: Diagnostics) -> str:
    hw, rt, sel = d.hardware, d.runtime, d.selection
    lines = [
        f"hibiki-asr {d.engine_version} (api {d.api_version}), runtime variant: {d.variant or 'unknown'}",
        "",
        f"System   {hw.os} {hw.arch}"
        + (", in a container" if hw.in_container else "")
        + f", {hw.cpu_cores or '?'} CPU cores"
        + ("" if hw.cpu_avx2 is None else f", AVX2 {'yes' if hw.cpu_avx2 else 'NO'}"),
    ]
    if hw.gpus:
        for gpu in hw.gpus:
            detail = ", ".join(
                x
                for x in (
                    f"driver {gpu.driver}" if gpu.driver else "no usable driver",
                    f"{gpu.vram_mb} MB" if gpu.vram_mb else "",
                    f"compute capability {gpu.compute_capability}" if gpu.compute_capability else "",
                    gpu.gfx or "",
                    "integrated" if gpu.integrated else "",
                )
                if x
            )
            lines.append(f"GPU      {gpu.name} ({detail})")
    else:
        lines.append("GPU      none detected")

    if rt.probe_ok:
        providers = ", ".join(p.replace("ExecutionProvider", "") for p in rt.onnxruntime_providers) or "-"
        lines.append(
            f"Runtime  ctranslate2 {rt.ctranslate2_version or 'MISSING'}, faster-whisper {rt.faster_whisper_version or 'MISSING'}, "
            f"onnxruntime {rt.onnxruntime_version or 'MISSING'} [{providers}]"
        )
        lines.append(f"         CUDA/ROCm devices seen by CTranslate2: {rt.cuda_device_count}")
    else:
        lines.append(f"Runtime  could not be inspected: {rt.probe_error}")

    verdict = f"{sel.device} ({sel.compute_type}), VAD on {sel.vad_device}"
    if sel.degraded:
        verdict += "   <-- a GPU was expected, running on the CPU instead"
    lines += ["", f"Device   {verdict}", ""]

    if d.findings:
        lines.append("Findings")
        for finding in d.findings:
            lines.append(_wrap(f"[{_MARK[finding.severity]}] {finding.code}: {finding.message}", "  "))
            if finding.hint:
                lines.append(_wrap(f"-> {finding.hint}", "      "))
    else:
        lines.append("No problems found.")
    return "\n".join(lines)


def log_findings(logger: logging.Logger, d: Diagnostics) -> None:
    """One log record per finding, at a level that matches its severity, with the fix inline."""
    sel = d.selection
    summary = f"device={sel.device} compute_type={sel.compute_type} vad_device={sel.vad_device}"
    if sel.degraded:
        logger.warning("running on the CPU although a GPU was expected (%s)", summary)
    else:
        logger.info("inference %s", summary)
    for finding in d.findings:
        logger.log(_LEVEL[finding.severity], "%s", _one_line(finding))


def _one_line(f: Finding) -> str:
    return f"{f.code}: {f.message}" + (f" | fix: {f.hint}" if f.hint else "")
