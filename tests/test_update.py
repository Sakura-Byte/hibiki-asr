"""`hibiki-asr update`: how the engine was installed, the upgrade command, and re-applying the runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import run_cli as run
from hibiki_asr import cli
from hibiki_asr.provision.commands import Installer
from hibiki_asr.provision.pins import lockfile_sha256
from hibiki_asr.provision.state import write_variant
from hibiki_asr.provision.update import (
    InstallSource,
    UpdateRefused,
    detect_source,
    plan_update,
    upgrade_command,
)
from hibiki_asr.provision.variants import get_variant

UV = "/opt/bin/uv"
PYTHON = "/venv/bin/python"
WITH_UV = Installer(PYTHON, UV)
WITH_PIP = Installer(PYTHON)
REPO = "https://github.com/Sakura-Byte/hibiki-asr"


def git(revision: str | None = None, url: str = REPO) -> str:
    vcs = {"vcs": "git", "commit_id": "a" * 40, **({"requested_revision": revision} if revision else {})}
    return json.dumps({"url": url, "vcs_info": vcs})


# --- how it was installed ---------------------------------------------------------------------------


def test_a_uv_tool_environment_is_recognised_by_its_receipt(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_text("[tool]\n")
    assert detect_source(tmp_path, None) == InstallSource("uv-tool")
    assert detect_source(tmp_path, git("v1")) == InstallSource("uv-tool")  # the receipt wins


@pytest.mark.parametrize(
    "direct_url,expected",
    [
        (None, InstallSource("index")),
        (git(), InstallSource("git", REPO, None)),
        (git("v0.2.0"), InstallSource("git", REPO, "v0.2.0")),
        (
            json.dumps({"url": "file:///src/hibiki-asr", "dir_info": {"editable": True}}),
            InstallSource("editable", "file:///src/hibiki-asr"),
        ),
        (
            json.dumps({"url": "file:///src/hibiki-asr", "dir_info": {}}),
            InstallSource("local", "file:///src/hibiki-asr"),
        ),
        (
            json.dumps({"url": "file:///dist/h.whl", "archive_info": {}}),
            InstallSource("local", "file:///dist/h.whl"),
        ),
        (
            json.dumps({"url": "https://hg/x", "vcs_info": {"vcs": "hg"}}),
            InstallSource("local", "https://hg/x"),
        ),
        ("not json", InstallSource("local")),
        ("{}", InstallSource("local")),
    ],
)
def test_direct_url_json_says_where_it_came_from(
    tmp_path: Path, direct_url: str | None, expected: InstallSource
) -> None:
    assert detect_source(tmp_path, direct_url) == expected


# --- the upgrade command ----------------------------------------------------------------------------


def test_a_uv_tool_is_upgraded_by_uv() -> None:
    assert upgrade_command(InstallSource("uv-tool"), WITH_UV).argv == (UV, "tool", "upgrade", "hibiki-asr")
    with pytest.raises(UpdateRefused, match="`uv` is not on PATH"):
        upgrade_command(InstallSource("uv-tool"), WITH_PIP)


def test_a_git_install_is_upgraded_from_the_same_repository_and_ref() -> None:
    source = InstallSource("git", REPO, "v0.2.0")
    assert upgrade_command(source, WITH_PIP).argv == (
        PYTHON, "-m", "pip", "install", "--upgrade", f"hibiki-asr @ git+{REPO}@v0.2.0",
    )  # fmt: skip
    assert upgrade_command(InstallSource("git", "https://example.org/fork"), WITH_UV).argv == (
        UV, "pip", "install", "--python", PYTHON, "--upgrade", "hibiki-asr @ git+https://example.org/fork",
    )  # fmt: skip


def test_an_index_install_is_upgraded_from_the_index() -> None:
    assert upgrade_command(InstallSource("index"), WITH_PIP).argv[-2:] == ("--upgrade", "hibiki-asr")


def test_a_source_checkout_is_never_overwritten() -> None:
    with pytest.raises(UpdateRefused, match=r"editable mode from file:///src/h.*git"):
        upgrade_command(InstallSource("editable", "file:///src/h"), WITH_UV)


def test_a_local_file_install_cannot_be_refreshed() -> None:
    with pytest.raises(UpdateRefused, match="Reinstall it from the new source"):
        upgrade_command(InstallSource("local", "file:///dist/h.whl"), WITH_UV)


# --- the plan ---------------------------------------------------------------------------------------


def test_a_uv_tool_gets_setup_again_because_uv_rebuilt_its_environment() -> None:
    plan = plan_update(InstallSource("uv-tool"), WITH_UV, "cuda12")
    assert plan.upgrade.argv == (UV, "tool", "upgrade", "hibiki-asr")
    assert plan.setup is not None and plan.setup.argv == (
        PYTHON, "-m", "hibiki_asr.cli", "setup", "--variant", "cuda12", "--yes",
    )  # fmt: skip
    assert "removes the runtime packages" in plan.notes[0]


def test_a_pip_install_runs_setup_only_when_the_lockfile_changed() -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_PIP, "cpu")
    assert plan.setup is not None and plan.setup.argv[-1] == "--if-changed"
    assert "only if its lockfile changed" in plan.notes[-1]


def test_an_experimental_variant_keeps_the_consent_it_was_installed_with() -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_UV, "cuda12-blackwell")
    assert plan.setup is not None and "--allow-experimental" in plan.setup.argv
    stable = plan_update(InstallSource("git", REPO), WITH_UV, "cuda12")
    assert stable.setup is not None and "--allow-experimental" not in stable.setup.argv


def test_a_variant_this_version_has_never_heard_of_is_passed_on_unchanged() -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_UV, "cuda13")
    assert plan.setup is not None and plan.setup.argv[4:7] == ("--variant", "cuda13", "--yes")


def test_the_config_file_reaches_the_setup_process() -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_UV, "cpu", config=Path("/etc/h.toml"))
    assert plan.setup is not None
    assert plan.setup.argv[3:6] == ("--config", str(Path("/etc/h.toml")), "setup")


def test_without_a_recorded_variant_only_the_engine_is_upgraded() -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_UV, None)
    assert plan.setup is None and "run `hibiki-asr setup`" in plan.notes[-1]


def test_a_pinned_ref_is_mentioned() -> None:
    plan = plan_update(InstallSource("git", REPO, "v0.1.0"), WITH_UV, None)
    assert "Installed from 'v0.1.0'" in plan.notes[0]


@pytest.mark.parametrize("variant_id", ["cpu", "cuda12", "cuda12-blackwell"])
def test_the_setup_command_it_builds_is_one_the_cli_accepts(variant_id: str) -> None:
    plan = plan_update(InstallSource("git", REPO), WITH_UV, variant_id, config=Path("/etc/h.toml"))
    assert plan.setup is not None
    args = cli.build_parser().parse_args(list(plan.setup.argv[3:]))  # what follows `python -m hibiki_asr.cli`
    assert (args.command, args.variant, args.yes, args.if_changed) == ("setup", variant_id, True, True)
    assert args.allow_experimental is get_variant(variant_id).experimental


# --- the command ------------------------------------------------------------------------------------


@pytest.fixture
def installed(env, monkeypatch):
    """A pip-style install of a git checkout, with the cuda12 runtime recorded by an earlier `setup`."""
    monkeypatch.setattr(cli, "_platform", lambda: "linux")
    monkeypatch.setattr(cli, "detect_installer", lambda python: Installer(python, UV))
    monkeypatch.setattr(cli, "read_direct_url", lambda: git("main"))
    write_variant(env.settings.data_dir, "cuda12", lockfile_sha256(get_variant("cuda12")))
    return env


def test_update_upgrades_then_reapplies_the_runtime_and_says_to_restart(installed, commands, capsys) -> None:
    code, out, _ = run(capsys, "update")
    assert code == 0
    assert commands[0][:5] == [UV, "pip", "install", "--python", commands[0][4]]
    assert commands[0][-1] == f"hibiki-asr @ git+{REPO}@main"
    assert commands[1][1:] == [
        "-m",
        "hibiki_asr.cli",
        "setup",
        "--variant",
        "cuda12",
        "--yes",
        "--if-changed",
    ]
    assert "systemctl --user restart hibiki-asr.service" in out


def test_update_dry_run_shows_the_commands_and_changes_nothing(installed, commands, capsys) -> None:
    code, out, _ = run(capsys, "update", "--dry-run")
    assert code == 0 and commands == []
    assert "hibiki-asr @ git+" in out and "setup --variant cuda12 --yes --if-changed" in out
    assert "Dry run: nothing was changed." in out and "Restart" not in out


def test_a_failed_upgrade_stops_before_setup(installed, capsys, monkeypatch) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(cli, "command_runner", lambda argv: (ran.append(list(argv)), 9)[1])
    code, _, err = run(capsys, "update")
    assert code == 9 and len(ran) == 1 and "status 9" in err


def test_a_failed_setup_says_the_engine_was_upgraded(installed, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "command_runner", lambda argv: 4 if "setup" in argv else 0)
    code, out, err = run(capsys, "update")
    assert code == 4 and "The engine itself was upgraded" in err and "Restart" not in out


def test_update_without_a_recorded_variant_leaves_the_runtime_alone(
    env, commands, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_platform", lambda: "linux")
    monkeypatch.setattr(cli, "detect_installer", lambda python: Installer(python, UV))
    monkeypatch.setattr(cli, "read_direct_url", lambda: None)
    code, out, _ = run(capsys, "update")
    assert code == 0 and len(commands) == 1 and "run `hibiki-asr setup`" in out


def test_update_uses_uv_for_a_uv_tool(installed, commands, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "detect_source", lambda prefix, direct_url: InstallSource("uv-tool"))
    assert run(capsys, "update")[0] == 0
    assert commands[0] == [UV, "tool", "upgrade", "hibiki-asr"]
    assert "--if-changed" not in commands[1]  # uv rebuilt the environment: always re-apply


def test_a_source_checkout_is_refused_and_nothing_runs(installed, commands, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "read_direct_url", lambda: json.dumps({"url": "file:///src/h", "dir_info": {"editable": True}})
    )
    code, _, err = run(capsys, "update")
    assert code == 2 and "editable mode" in err and commands == []


def test_a_container_is_updated_by_pulling_an_image(installed, commands, capsys, monkeypatch) -> None:
    monkeypatch.setenv("HIBIKI_ASR_VARIANT", "cuda12")
    code, _, err = run(capsys, "update")
    assert code == 2 and "pulling a newer image" in err and commands == []
