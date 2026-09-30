"""install.sh and install.ps1 say the same thing, and only recommend commands the CLI really has."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from hibiki_asr import cli

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["install.sh", "install.ps1"]
REPO = "https://github.com/Sakura-Byte/hibiki-asr"


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def suggested_commands(script: str) -> list[str]:
    """The `hibiki-asr ...` lines of the 'Next steps' block, without their trailing comments."""
    block = script[script.index("Next steps:") :]
    found = re.findall(r"^  (hibiki-asr [^#\n]+?)\s*(?:#.*)?$", block, re.MULTILINE)
    return [c.strip() for c in found]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_engine_is_installed_with_the_cpu_baseline_from_the_repository(name: str) -> None:
    script = text(name)
    assert f'"{REPO}"' in script  # REPO_URL / $RepoUrl
    assert "hibiki-asr[runtime] @ git+" in script
    assert "tool" in script and "install" in script and "--force" in script  # safe to run again
    assert "HIBIKI_ASR_REF" in script  # pin a branch, tag or commit
    assert "https://astral.sh/uv/install" in script  # the official uv installer, only when uv is missing


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_runtime_is_chosen_by_setup_without_prompting(name: str) -> None:
    script = text(name)
    assert "setup" in script and "--variant" in script and "--yes" in script
    assert not re.search(r"(^|[|;&]\s*)sudo\b", script, re.MULTILINE)  # a comment may say it needs none


@pytest.mark.parametrize("name", SCRIPTS)
def test_next_steps_are_commands_the_cli_accepts(name: str) -> None:
    commands = suggested_commands(text(name))
    assert [c.split()[1] for c in commands] == ["models", "models", "serve", "service", "update"]
    assert "hibiki-asr models download chickenrice@v2" in commands
    parser = cli.build_parser()
    for command in commands:
        argv = shlex.split(command)[1:]
        assert parser.parse_args(argv).command == argv[0], command


def test_the_scripts_do_not_leak_their_own_variables_to_the_engine() -> None:
    assert "unset HIBIKI_ASR_REF HIBIKI_ASR_INSTALL_VARIANT" in text("install.sh")
    assert "Remove-Item Env:HIBIKI_ASR_REF, Env:HIBIKI_ASR_INSTALL_VARIANT" in text("install.ps1")


def test_the_shell_script_stops_on_errors() -> None:
    script = text("install.sh")
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "\nset -euo pipefail\n" in script


def test_the_powershell_script_throws_instead_of_exiting() -> None:
    script = text("install.ps1")
    assert '$ErrorActionPreference = "Stop"' in script
    assert not re.search(
        r"^\s*exit\b", script, re.MULTILINE
    )  # `exit` would close the window under `irm | iex`
