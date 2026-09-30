"""`hibiki-asr service`: start the engine automatically, as a systemd user unit (Linux) or a scheduled task (Windows).

The unit and the task definition are rendered by pure functions and turned into a plan (files to write, commands
to run) that ``apply_plan`` executes. Nothing here calls systemctl or schtasks except through the runner the
caller passes in.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .commands import Command, Runner

SERVICE_NAME = "hibiki-asr"
UNIT_NAME = f"{SERVICE_NAME}.service"
TASK_NAME = SERVICE_NAME
ACTIONS = ("install", "uninstall", "status")


class ServiceUnsupported(Exception):
    """This operating system has no supported way to run the engine as a service. The message is for the user."""


@dataclass(frozen=True)
class WriteFile:
    path: Path
    text: str
    encoding: str = "utf-8"


@dataclass(frozen=True)
class RemoveFile:
    path: Path


@dataclass(frozen=True)
class Step:
    command: Command
    required: bool = True  # a failing optional step is reported and skipped
    passthrough: bool = False  # its exit code is the result of the whole action (`status`)


Action = WriteFile | RemoveFile | Step


@dataclass(frozen=True)
class ServicePlan:
    actions: tuple[Action, ...]
    notes: tuple[str, ...] = ()


# -- what to run -----------------------------------------------------------------------------------------


def engine_argv(
    which: Callable[[str], str | None], python: str, config: Path | None = None
) -> tuple[str, ...]:
    """The command line that starts the engine: the `hibiki-asr` launcher when there is one."""
    launcher = which("hibiki-asr")
    base = (launcher,) if launcher else (python, "-m", "hibiki_asr.cli")
    return (*base, *(("--config", str(config)) if config else ()), "serve")


# -- Linux: systemd user unit ----------------------------------------------------------------------------


def _systemd_arg(arg: str) -> str:
    """One ExecStart word: systemd treats % and $ specially and splits on whitespace unless quoted."""
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
    return f'"{escaped}"' if escaped != arg or any(c.isspace() for c in arg) else arg


def render_systemd_unit(argv: Sequence[str]) -> str:
    return f"""[Unit]
Description=hibiki-asr speech recognition engine
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
ExecStart={" ".join(_systemd_arg(a) for a in argv)}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def unit_path(config_home: Path) -> Path:
    return config_home / "systemd" / "user" / UNIT_NAME


def _systemctl(*args: str) -> Command:
    return Command(("systemctl", "--user", *args))


def _linux_plan(action: str, argv: Sequence[str], config_home: Path) -> ServicePlan:
    path = unit_path(config_home)
    if action == "install":
        return ServicePlan(
            (
                WriteFile(path, render_systemd_unit(argv)),
                Step(_systemctl("daemon-reload")),
                Step(_systemctl("enable", "--now", UNIT_NAME)),
            ),
            (
                f"Installed {path} and started it.",
                "It runs while you are logged in. To have it start at boot and keep running after you log out: "
                "`loginctl enable-linger $USER`.",
                f"Logs: `journalctl --user -u {SERVICE_NAME}`. Status: `hibiki-asr service status`.",
            ),
        )
    if action == "uninstall":
        return ServicePlan(
            (
                Step(_systemctl("disable", "--now", UNIT_NAME), required=False),
                RemoveFile(path),
                Step(_systemctl("daemon-reload")),
            ),
            (f"Stopped and removed {path}.",),
        )
    return ServicePlan((Step(_systemctl("status", UNIT_NAME, "--no-pager"), passthrough=True),))


# -- Windows: Task Scheduler -----------------------------------------------------------------------------


def render_task_xml(argv: Sequence[str], user: str | None = None) -> str:
    """A Task Scheduler definition: run the engine at logon, with no time limit.

    A task created with plain `schtasks /Create` flags is stopped after 72 hours and when the laptop goes on
    battery, which would silently kill the engine; only a task definition can switch that off.
    """
    who = f"\n        <UserId>{escape(user)}</UserId>" if user else ""
    principal_user = f"\n      <UserId>{escape(user)}</UserId>" if user else ""
    arguments = subprocess.list2cmdline(list(argv[1:]))
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>hibiki-asr speech recognition engine</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>{who}
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">{principal_user}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(argv[0])}</Command>
      <Arguments>{escape(arguments)}</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def windows_user(env: Mapping[str, str]) -> str | None:
    """``DOMAIN\\name`` of the current user, or None when the environment does not say."""
    name = env.get("USERNAME")
    if not name:
        return None
    return f"{env['USERDOMAIN']}\\{name}" if env.get("USERDOMAIN") else name


def _schtasks(*args: str) -> Command:
    return Command(("schtasks", *args))


def _windows_plan(action: str, argv: Sequence[str], xml_path: Path, user: str | None) -> ServicePlan:
    if action == "install":
        return ServicePlan(
            (
                WriteFile(xml_path, render_task_xml(argv, user), encoding="utf-16"),
                Step(_schtasks("/Create", "/TN", TASK_NAME, "/XML", str(xml_path), "/F")),
                Step(_schtasks("/Run", "/TN", TASK_NAME), required=False),
            ),
            (
                f"Created the scheduled task '{TASK_NAME}': it starts hibiki-asr when you log on.",
                "Task Scheduler is not set up to restart the engine if it crashes; run "
                f"`schtasks /Run /TN {TASK_NAME}` or log on again to start it.",
                "It only runs while you are logged on.",
            ),
        )
    if action == "uninstall":
        return ServicePlan(
            (
                Step(_schtasks("/End", "/TN", TASK_NAME), required=False),
                Step(_schtasks("/Delete", "/TN", TASK_NAME, "/F")),
                RemoveFile(xml_path),
            ),
            (f"Stopped and removed the scheduled task '{TASK_NAME}'.",),
        )
    return ServicePlan((Step(_schtasks("/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"), passthrough=True),))


# -- planning and applying -------------------------------------------------------------------------------


def plan_service(
    action: str,
    platform: str,
    argv: Sequence[str],
    *,
    config_home: Path,
    data_dir: Path,
    user: str | None = None,
) -> ServicePlan:
    if action not in ACTIONS:
        raise ValueError(f"unknown service action {action!r}")
    if platform.startswith("linux"):
        return _linux_plan(action, argv, config_home)
    if platform.startswith("win"):
        return _windows_plan(action, argv, data_dir / "service" / f"{TASK_NAME}-task.xml", user)
    raise ServiceUnsupported(
        "Running the engine as a service is not supported on this system yet. "
        "Run `hibiki-asr serve`; it stays in the foreground until you stop it."
    )


def describe(plan: ServicePlan) -> list[str]:
    """What applying the plan would do, one line per action (file contents follow their path)."""
    lines: list[str] = []
    for action in plan.actions:
        if isinstance(action, WriteFile):
            lines += [f"write {action.path}:", *(f"    {line}" for line in action.text.splitlines())]
        elif isinstance(action, RemoveFile):
            lines.append(f"remove {action.path}")
        else:
            lines.append(action.command.display() + ("" if action.required else "   (failure is ignored)"))
    return lines


def apply_plan(plan: ServicePlan, runner: Runner, say: Callable[[str], None] = print) -> int:
    """Carry out the plan in order. Stops at the first failing required step and returns its exit code."""
    for action in plan.actions:
        if isinstance(action, WriteFile):
            action.path.parent.mkdir(parents=True, exist_ok=True)
            action.path.write_text(action.text, encoding=action.encoding)
            say(f"wrote {action.path}")
        elif isinstance(action, RemoveFile):
            action.path.unlink(missing_ok=True)
            say(f"removed {action.path}")
        else:
            say(f"+ {action.command.display()}")
            code = runner(action.command.argv)
            if action.passthrough:
                return code
            if code != 0 and action.required:
                say(f"error: the command exited with status {code}")
                return code
            if code != 0:
                say(f"warning: the command exited with status {code}; continuing")
    return 0


def restart_hint(platform: str) -> str:
    """How to make a running engine pick up new code."""
    if platform.startswith("linux"):
        return (
            f"Restart the engine to run the new version: `systemctl --user restart {UNIT_NAME}` "
            "if it runs as a service, otherwise stop `hibiki-asr serve` and start it again."
        )
    if platform.startswith("win"):
        return (
            f"Restart the engine to run the new version: `schtasks /End /TN {TASK_NAME}` then "
            f"`schtasks /Run /TN {TASK_NAME}` if it runs as a scheduled task, "
            "otherwise stop `hibiki-asr serve` and start it again."
        )
    return "Restart the engine to run the new version: stop `hibiki-asr serve` and start it again."
