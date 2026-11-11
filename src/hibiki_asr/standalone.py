"""Dispatch the frozen runtime probe and verify the bundled inference stack."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    bundle = Path(__file__).with_name("bundle.json")
    variant = json.loads(bundle.read_text(encoding="utf-8"))["variant"]
    os.environ["HIBIKI_ASR_VARIANT"] = variant
    if sys.argv[1:] == ["--runtime-probe"]:
        from .diagnostics.runtime import _MARKER, collect_runtime_facts

        print(_MARKER + collect_runtime_facts().model_dump_json())
        return 0
    if sys.argv[1:] == ["--self-test"]:
        from .diagnostics.runtime import probe_runtime
        from .worker.process import ProcessWorker

        facts = probe_runtime()
        assert facts.probe_ok, facts.probe_error
        assert facts.ctranslate2_version, facts.ctranslate2_error
        assert facts.onnxruntime_version, facts.onnxruntime_error
        assert facts.faster_whisper_version, facts.faster_whisper_error
        if variant == "mlx":
            assert facts.mlx_version, facts.mlx_error
            import mlx_whisper  # noqa: F401

        worker = ProcessWorker()
        worker.shutdown(30)
        assert worker.exit_code() == 0, worker.exit_code()
        print(facts.model_dump_json())
        return 0
    from .cli import main as cli_main

    return cli_main()
