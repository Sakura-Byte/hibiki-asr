"""`hibiki-asr service`: the rendered unit and task definition, the plans, and the command.

systemctl and schtasks are never called: plans are data, and the CLI's runner is replaced (conftest.py).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from helpers import run_cli as run
from hibiki_asr import cli
from hibiki_asr.provision.commands import Command
from hibiki_asr.provision.service import (
    RemoveFile,
    ServicePlan,
    ServiceUnsupported,
    Step,
    WriteFile,
    apply_plan,
    describe,
    engine_argv,
    plan_service,
    render_systemd_unit,
    render_task_xml,
    restart_hint,
    unit_path,
    windows_user,
)

ENGINE = ("/home/me/.local/bin/hibiki-asr", "serve")
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def plan(action: str, platform: str, tmp_path: Path, argv=ENGINE, user: str | None = None) -> ServicePlan:
    return plan_service(
        action, platform, argv, config_home=tmp_path / "config", data_dir=tmp_path / "data", user=user
    )


# --- what starts the engine -------------------------------------------------------------------------


def test_the_launcher_is_used_when_it_is_on_the_path() -> None:
    assert engine_argv(lambda _n: "/home/me/.local/bin/hibiki-asr", "/py") == ENGINE


def test_without_a_launcher_the_interpreter_runs_the_cli_module() -> None:
    assert engine_argv(lambda _n: None, "/venv/bin/python") == (
        "/venv/bin/python",
        "-m",
        "hibiki_asr.cli",
        "serve",
    )


def test_a_config_file_is_passed_before_the_subcommand() -> None:
    argv = engine_argv(lambda _n: "/bin/hibiki-asr", "/py", Path("/etc/h.toml"))
    assert argv == ("/bin/hibiki-asr", "--config", str(Path("/etc/h.toml")), "serve")


# --- systemd ----------------------------------------------------------------------------------------


def test_the_unit_starts_serve_and_restarts_it_when_it_fails() -> None:
    unit = render_systemd_unit(ENGINE)
    assert "ExecStart=/home/me/.local/bin/hibiki-asr serve\n" in unit
    assert "Restart=on-failure" in unit and "RestartSec=5" in unit
    assert "WantedBy=default.target" in unit  # a user unit: default.target, not multi-user.target
    # a broken configuration must not restart forever
    assert "StartLimitBurst=5" in unit and "StartLimitIntervalSec=120" in unit


@pytest.mark.parametrize(
    "argument,expected",
    [
        ("/plain/path", "/plain/path"),
        ("/with space/h.toml", '"/with space/h.toml"'),
        ("/100%/h.toml", '"/100%%/h.toml"'),  # systemd expands specifiers such as %h
        ("$HOME/h.toml", '"$$HOME/h.toml"'),  # ... and environment variables
        ('a"b', '"a\\"b"'),
        ("a\\b", '"a\\\\b"'),
    ],
)
def test_exec_start_words_are_quoted_the_way_systemd_reads_them(argument: str, expected: str) -> None:
    line = next(
        x
        for x in render_systemd_unit(("/bin/hibiki-asr", argument)).splitlines()
        if x.startswith("ExecStart=")
    )
    assert line == f"ExecStart=/bin/hibiki-asr {expected}"


def test_the_unit_lives_in_the_users_systemd_directory() -> None:
    assert unit_path(Path("/home/me/.config")) == Path("/home/me/.config/systemd/user/hibiki-asr.service")


def test_linux_install_writes_the_unit_then_enables_and_starts_it(tmp_path: Path) -> None:
    result = plan("install", "linux", tmp_path)
    write, reload, enable = result.actions
    assert isinstance(write, WriteFile) and write.path == tmp_path / "config/systemd/user/hibiki-asr.service"
    assert write.text == render_systemd_unit(ENGINE)
    assert isinstance(reload, Step) and reload.command.argv == ("systemctl", "--user", "daemon-reload")
    assert isinstance(enable, Step)
    assert enable.command.argv == ("systemctl", "--user", "enable", "--now", "hibiki-asr.service")
    assert any("loginctl enable-linger" in note for note in result.notes)


def test_linux_uninstall_stops_removes_and_reloads_in_that_order(tmp_path: Path) -> None:
    disable, remove, reload = plan("uninstall", "linux", tmp_path).actions
    assert isinstance(disable, Step) and disable.command.argv[2:] == (
        "disable",
        "--now",
        "hibiki-asr.service",
    )
    assert not disable.required  # not being installed is fine
    assert isinstance(remove, RemoveFile) and remove.path.name == "hibiki-asr.service"
    assert isinstance(reload, Step) and reload.command.argv[2:] == ("daemon-reload",)


def test_linux_status_hands_back_systemctls_exit_code(tmp_path: Path) -> None:
    (status,) = plan("status", "linux", tmp_path).actions
    assert isinstance(status, Step) and status.passthrough
    assert status.command.argv == ("systemctl", "--user", "status", "hibiki-asr.service", "--no-pager")


# --- Task Scheduler ---------------------------------------------------------------------------------


def task(
    argv=("C:\\Users\\me\\.local\\bin\\hibiki-asr.exe", "serve"), user: str | None = "PC\\me"
) -> ET.Element:
    return ET.fromstring(render_task_xml(argv, user).replace('encoding="UTF-16"', ""))


def test_the_task_starts_the_engine_at_logon_and_never_times_out() -> None:
    root = task()
    assert root.find("t:Triggers/t:LogonTrigger/t:UserId", NS).text == "PC\\me"
    assert root.find("t:Actions/t:Exec/t:Command", NS).text == "C:\\Users\\me\\.local\\bin\\hibiki-asr.exe"
    assert root.find("t:Actions/t:Exec/t:Arguments", NS).text == "serve"
    settings = {child.tag.split("}")[1]: child.text for child in root.find("t:Settings", NS)}
    assert settings["ExecutionTimeLimit"] == "PT0S"  # schtasks flags alone give 72 hours
    assert settings["DisallowStartIfOnBatteries"] == "false" and settings["StopIfGoingOnBatteries"] == "false"
    assert settings["MultipleInstancesPolicy"] == "IgnoreNew"
    assert root.find("t:Principals/t:Principal/t:RunLevel", NS).text == "LeastPrivilege"  # never elevated
    assert "RestartOnFailure" not in render_task_xml(("x.exe", "serve"))  # documented as unsupported


def test_task_arguments_are_quoted_and_the_xml_stays_well_formed() -> None:
    root = task(("C:\\Program Files\\h&h\\hibiki-asr.exe", "--config", "C:\\my dir\\<h>.toml", "serve"))
    assert root.find("t:Actions/t:Exec/t:Command", NS).text == "C:\\Program Files\\h&h\\hibiki-asr.exe"
    assert root.find("t:Actions/t:Exec/t:Arguments", NS).text == '--config "C:\\my dir\\<h>.toml" serve'


def test_a_task_without_a_known_user_omits_the_user_elements() -> None:
    root = task(user=None)
    assert root.find("t:Triggers/t:LogonTrigger/t:UserId", NS) is None
    assert root.find("t:Principals/t:Principal/t:UserId", NS) is None


def test_the_windows_user_is_domain_and_name() -> None:
    assert windows_user({"USERDOMAIN": "PC", "USERNAME": "me"}) == "PC\\me"
    assert windows_user({"USERNAME": "me"}) == "me"
    assert windows_user({}) is None


def test_windows_install_writes_utf16_xml_and_creates_then_runs_the_task(tmp_path: Path) -> None:
    result = plan("install", "win32", tmp_path, user="PC\\me")
    write, create, run_now = result.actions
    assert isinstance(write, WriteFile) and write.encoding == "utf-16"
    assert write.path == tmp_path / "data/service/hibiki-asr-task.xml"
    assert isinstance(create, Step)
    assert create.command.argv == ("schtasks", "/Create", "/TN", "hibiki-asr", "/XML", str(write.path), "/F")
    assert isinstance(run_now, Step) and run_now.command.argv == ("schtasks", "/Run", "/TN", "hibiki-asr")
    assert not run_now.required
    assert any("not set up to restart" in note for note in result.notes)  # the limitation is stated


def test_windows_uninstall_and_status(tmp_path: Path) -> None:
    end, delete, remove = plan("uninstall", "win32", tmp_path).actions
    assert (
        isinstance(end, Step)
        and end.command.argv == ("schtasks", "/End", "/TN", "hibiki-asr")
        and not end.required
    )
    assert isinstance(delete, Step) and delete.command.argv == (
        "schtasks",
        "/Delete",
        "/TN",
        "hibiki-asr",
        "/F",
    )
    assert isinstance(remove, RemoveFile)
    (status,) = plan("status", "win32", tmp_path).actions
    assert (
        isinstance(status, Step)
        and status.passthrough
        and status.command.argv[:3] == ("schtasks", "/Query", "/TN")
    )


@pytest.mark.parametrize("platform", ["darwin", "freebsd14"])
def test_other_systems_are_told_to_run_serve(platform: str, tmp_path: Path) -> None:
    with pytest.raises(ServiceUnsupported, match="Run `hibiki-asr serve`"):
        plan("install", platform, tmp_path)


def test_an_unknown_action_is_a_programming_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restart"):
        plan("restart", "linux", tmp_path)


# --- applying a plan --------------------------------------------------------------------------------


def test_applying_writes_files_and_runs_steps_in_order(tmp_path: Path) -> None:
    unit = tmp_path / "deep" / "dir" / "u.service"
    seen: list[tuple[str, ...]] = []
    result = ServicePlan(
        (
            WriteFile(unit, "héllo"),
            Step(Command(("one",))),
            RemoveFile(tmp_path / "missing"),  # removing what is not there is fine
            Step(Command(("two",))),
        )
    )
    said: list[str] = []
    assert apply_plan(result, lambda argv: (seen.append(tuple(argv)), 0)[1], said.append) == 0
    assert seen == [("one",), ("two",)] and unit.read_text(encoding="utf-8") == "héllo"


def test_the_windows_task_file_is_really_utf16_with_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "t.xml"
    apply_plan(ServicePlan((WriteFile(path, "<Task/>", "utf-16"),)), lambda argv: 0, lambda _m: None)
    assert path.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff")


def test_a_failing_required_step_stops_the_plan(tmp_path: Path) -> None:
    seen: list[str] = []
    result = ServicePlan((Step(Command(("a",))), Step(Command(("b",))), Step(Command(("c",)))))

    def runner(argv) -> int:
        seen.append(argv[0])
        return 4 if argv[0] == "b" else 0

    said: list[str] = []
    assert apply_plan(result, runner, said.append) == 4
    assert seen == ["a", "b"] and "status 4" in said[-1]


def test_a_failing_optional_step_only_warns(tmp_path: Path) -> None:
    said: list[str] = []
    result = ServicePlan((Step(Command(("a",)), required=False), Step(Command(("b",)))))
    assert apply_plan(result, lambda argv: 1 if argv[0] == "a" else 0, said.append) == 0
    assert any(line.startswith("warning:") for line in said)


def test_a_passthrough_step_returns_its_code_without_calling_it_an_error() -> None:
    said: list[str] = []
    result = ServicePlan((Step(Command(("systemctl",)), passthrough=True),))
    assert apply_plan(result, lambda argv: 3, said.append) == 3
    assert not any("error" in line for line in said)


def test_describe_lists_files_and_commands_without_running_anything(tmp_path: Path) -> None:
    lines = describe(plan("uninstall", "linux", tmp_path))
    assert lines[0].endswith("(failure is ignored)") and lines[1].startswith("remove ")
    text = "\n".join(describe(plan("install", "linux", tmp_path)))
    assert "[Service]" in text and "systemctl --user enable --now hibiki-asr.service" in text


def test_restart_advice_names_the_way_the_engine_runs() -> None:
    assert "systemctl --user restart hibiki-asr.service" in restart_hint("linux")
    assert "schtasks /End /TN hibiki-asr" in restart_hint("win32")
    assert (
        restart_hint("darwin")
        == "Restart the engine to run the new version: stop `hibiki-asr serve` and start it again."
    )


# --- the command ------------------------------------------------------------------------------------


@pytest.fixture
def linux(env, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli, "_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "shutil.which", lambda name: "/home/me/.local/bin/hibiki-asr" if name == "hibiki-asr" else None
    )
    return tmp_path / "xdg" / "systemd" / "user" / "hibiki-asr.service"


def test_service_install_writes_the_unit_and_runs_systemctl(linux, commands, capsys) -> None:
    code, out, _ = run(capsys, "service", "install")
    assert code == 0
    assert "ExecStart=/home/me/.local/bin/hibiki-asr serve" in linux.read_text(encoding="utf-8")
    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "hibiki-asr.service"],
    ]
    assert "loginctl enable-linger" in out


def test_service_install_dry_run_touches_nothing(linux, commands, capsys) -> None:
    code, out, _ = run(capsys, "service", "install", "--dry-run")
    assert code == 0 and commands == [] and not linux.exists()
    assert "[Service]" in out and "Dry run: nothing was changed." in out


def test_service_install_passes_the_config_file_to_serve(linux, commands, capsys, tmp_path: Path) -> None:
    config = tmp_path / "my.toml"
    assert run(capsys, "--config", str(config), "service", "install")[0] == 0
    unit = render_systemd_unit(("/home/me/.local/bin/hibiki-asr", "--config", str(config.resolve()), "serve"))
    assert linux.read_text(encoding="utf-8") == unit  # the path is quoted for systemd, whatever the platform


def test_service_uninstall_removes_the_unit(linux, commands, capsys) -> None:
    run(capsys, "service", "install")
    commands.clear()
    code, _, _ = run(capsys, "service", "uninstall")
    assert code == 0 and not linux.exists()
    assert [c[2] for c in commands] == ["disable", "daemon-reload"]


def test_a_failing_systemctl_is_reported(linux, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "command_runner", lambda argv: 1 if "enable" in argv else 0)
    code, out, _ = run(capsys, "service", "install")
    assert code == 1 and "status 1" in out
    assert linux.exists()  # the unit stays, so the user can enable it by hand


def test_service_status_returns_the_exit_code_of_systemctl(linux, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "command_runner", lambda argv: 3)  # systemctl: inactive
    assert run(capsys, "service", "status")[0] == 3


def test_macos_is_told_to_run_serve(env, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_platform", lambda: "darwin")
    code, _, err = run(capsys, "service", "install")
    assert code == 2 and "hibiki-asr serve" in err


def test_windows_install_creates_the_task_from_an_xml_file(env, commands, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_platform", lambda: "win32")
    monkeypatch.setattr("shutil.which", lambda name: "C:\\bin\\hibiki-asr.exe")
    code, out, _ = run(capsys, "service", "install")
    xml = env.settings.data_dir / "service" / "hibiki-asr-task.xml"
    assert code == 0 and xml.is_file()
    assert commands[0] == ["schtasks", "/Create", "/TN", "hibiki-asr", "/XML", str(xml), "/F"]
    assert commands[1] == ["schtasks", "/Run", "/TN", "hibiki-asr"]
    assert "not set up to restart" in out
