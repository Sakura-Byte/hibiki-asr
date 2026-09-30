"""The README only shows commands the CLI accepts, and tells the truth about each runtime variant."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from hibiki_asr import cli
from hibiki_asr.provision.variants import load_variants

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def shown_commands() -> list[str]:
    blocks = re.findall(r"```(?:bash|powershell)?\n(.*?)```", README, re.DOTALL)
    lines = [line for block in blocks for line in block.splitlines()]
    return [re.sub(r"\s+#.*$", "", line).strip() for line in lines if line.startswith("hibiki-asr ")]


def test_the_readme_shows_the_commands_of_this_phase() -> None:
    commands = shown_commands()
    assert len(commands) >= 12
    for expected in (
        "hibiki-asr setup",
        "hibiki-asr doctor",
        "hibiki-asr models sources",
        "hibiki-asr models download chickenrice@v2",
        "hibiki-asr serve",
    ):
        assert expected in commands
    assert any(c.startswith("hibiki-asr models import ") and "--model chickenrice@v2" in c for c in commands)


@pytest.mark.parametrize("command", shown_commands())
def test_every_command_in_the_readme_is_accepted_by_the_parser(command: str) -> None:
    argv = shlex.split(command)[1:]
    args = cli.build_parser().parse_args(argv)
    assert args.command == argv[0]


@pytest.mark.parametrize(
    "phrase",
    ["hibiki-asr update", "hibiki-asr service install", "setup --dry-run", "install.sh", "install.ps1"],
)
def test_the_readme_mentions_every_new_entry_point(phrase: str) -> None:
    assert phrase in README


def test_the_readme_says_which_variants_have_no_pins_and_that_rocm_is_unverified() -> None:
    for variant in load_variants().values():
        family = "rocm-win-gfx*" if variant.id.startswith("rocm-win") else variant.id
        assert f"`{family}`" in README or family in README, variant.id
    unpinned = [v.id for v in load_variants().values() if not v.installable]
    assert unpinned and "**no pins" in README
    assert "not verified" in README and "Nothing about ROCm is claimed to work" in README
    assert "**never run on a GPU**" in README
