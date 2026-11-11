"""Frozen commands must not invoke the executable as a Python interpreter."""
import subprocess
import sys

from hibiki_asr import cli
from hibiki_asr.diagnostics import runtime


def test_frozen_probe_uses_the_internal_probe_dispatch(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, runtime._MARKER + '{"probe_ok":true}', "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.probe_runtime().probe_ok
    assert commands == [[sys.executable, "--runtime-probe"]]


def test_frozen_setup_and_update_refuse_to_modify_the_bundle(monkeypatch, capsys):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    for command in ("setup", "update"):
        assert cli.main([command]) == 2
        assert "replace the binary bundle" in capsys.readouterr().err
