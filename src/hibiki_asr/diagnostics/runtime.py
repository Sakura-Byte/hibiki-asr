"""What the installed inference stack can actually do.

Importing ctranslate2 or onnxruntime on a machine with a half-installed GPU
runtime can fail with an ImportError, or even crash the interpreter. So the API
process never imports them: it runs this module in a short-lived subprocess and
reads the JSON it prints. The worker calls ``collect_runtime_facts`` directly,
because it imports the stack anyway.
"""

from __future__ import annotations

import json
import subprocess
import sys

from .schema import RuntimeFacts

_MARKER = "HIBIKI_ASR_RUNTIME_FACTS="


def collect_runtime_facts() -> RuntimeFacts:
    """Import the stack in this process and report what it supports. Never raises."""
    facts = RuntimeFacts(python=sys.version.split()[0])

    try:
        import ctranslate2

        facts.ctranslate2_version = getattr(ctranslate2, "__version__", None)
        try:
            facts.cuda_device_count = int(ctranslate2.get_cuda_device_count())
        except Exception as exc:  # noqa: BLE001 - a broken driver can raise anything
            facts.ctranslate2_error = f"get_cuda_device_count failed: {exc}"
        for device in ("cpu", "cuda"):
            if device == "cuda" and facts.cuda_device_count == 0:
                continue
            try:
                facts.compute_types[device] = sorted(ctranslate2.get_supported_compute_types(device))
            except Exception as exc:  # noqa: BLE001
                facts.ctranslate2_error = facts.ctranslate2_error or f"get_supported_compute_types({device}) failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - ImportError, OSError from missing shared libraries, ...
        facts.ctranslate2_error = f"{type(exc).__name__}: {exc}"

    try:
        import onnxruntime

        facts.onnxruntime_version = onnxruntime.__version__
        facts.onnxruntime_providers = list(onnxruntime.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        facts.onnxruntime_error = f"{type(exc).__name__}: {exc}"

    try:
        import faster_whisper

        facts.faster_whisper_version = getattr(faster_whisper, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        facts.faster_whisper_error = f"{type(exc).__name__}: {exc}"

    return facts


def probe_runtime(timeout: float = 60.0) -> RuntimeFacts:
    """Collect the facts in an isolated subprocess. A crash or hang becomes ``probe_ok=False``."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-m", "hibiki_asr.diagnostics.runtime"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RuntimeFacts(probe_ok=False, probe_error=f"runtime probe timed out after {timeout:.0f}s")
    except OSError as exc:
        return RuntimeFacts(probe_ok=False, probe_error=f"cannot start runtime probe: {exc}")

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_MARKER):
            try:
                return RuntimeFacts.model_validate(json.loads(line[len(_MARKER) :]))
            except ValueError as exc:
                return RuntimeFacts(probe_ok=False, probe_error=f"runtime probe returned invalid data: {exc}")

    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return RuntimeFacts(
        probe_ok=False,
        probe_error=f"runtime probe exited with code {completed.returncode}"
        + (f": {detail[-1]}" if detail else " (no output; the interpreter may have crashed)"),
    )


if __name__ == "__main__":
    print(_MARKER + collect_runtime_facts().model_dump_json())
