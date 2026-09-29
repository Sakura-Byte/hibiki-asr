"""`hibiki-asr setup`: choosing a variant, building the install commands, and the command itself.

Nothing here installs anything: commands are built as data, and the CLI's runner is replaced (conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import GTX1050, IGPU, RTX4090, RTX5090, RX7900, hw, rt
from helpers import run_cli as run
from hibiki_asr import cli
from hibiki_asr.diagnostics.schema import GpuInfo
from hibiki_asr.engine import Engine
from hibiki_asr.provision.commands import (
    Command,
    Installer,
    InstallerMissing,
    conflicting_distributions,
    detect_installer,
    run_command,
)
from hibiki_asr.provision.install import (
    Choice,
    SetupRefused,
    build_plan,
    check_installable,
    choose_variant,
    describe_gpu,
    execute,
)
from hibiki_asr.provision.pins import lockfile_path, lockfile_sha256
from hibiki_asr.provision.state import read_lockfile_sha256, read_variant
from hibiki_asr.provision.variants import get_variant

UV = "/opt/bin/uv"
PYTHON = "/venv/bin/python"


def choose(probe, requested: str = "auto", platform: str = "linux", allow: bool = False) -> Choice:
    return choose_variant(requested, probe, platform, allow_experimental=allow)


# --- choosing a variant -----------------------------------------------------------------------


def test_auto_follows_the_hardware_and_says_why() -> None:
    choice = choose(hw(RTX4090))
    assert choice.variant.id == "cuda12"
    assert (
        "RTX 4090" in choice.reason
        and "driver 555.42.06" in choice.reason
        and "compute capability 8.9" in choice.reason
    )

    choice = choose(hw())
    assert choice.variant.id == "cpu" and "no NVIDIA or AMD GPU was detected" in choice.reason


def test_auto_installs_the_cpu_runtime_when_no_gpu_runtime_fits() -> None:
    choice = choose(hw(GpuInfo(vendor="nvidia", name="Old", driver="390.1", compute_capability="6.1")))
    assert choice.variant.id == "cpu" and "no GPU runtime fits" in choice.reason and "Old" in choice.reason
    assert choose(hw(IGPU), platform="win32").variant.id == "cpu"


def test_a_maxwell_card_is_not_offered_a_runtime_its_wheel_cannot_run_on() -> None:
    maxwell = GpuInfo(vendor="nvidia", name="GeForce GTX 970", driver="550.1", compute_capability="5.2")
    choice = choose(hw(maxwell))
    assert choice.variant.id == "cpu"
    assert "cuda11 has no pinned runtime yet" in choice.reason and "GTX 970" in choice.reason


def test_auto_never_picks_an_experimental_runtime_unless_allowed() -> None:
    blackwell = choose(hw(RTX5090))
    assert blackwell.variant.id == "cpu"
    assert "cuda12-blackwell is experimental" in blackwell.reason
    assert "setup --variant cuda12-blackwell --allow-experimental" in blackwell.reason
    assert choose(hw(RTX5090), allow=True).variant.id == "cuda12-blackwell"


def test_auto_explains_an_amd_gpu_that_has_no_pinned_runtime() -> None:
    choice = choose(hw(RX7900, kfd=True))
    assert choice.variant.id == "cpu"
    assert "rocm-linux has no pinned runtime yet" in choice.reason and "gfx1100" in choice.reason
    assert choose(hw(RX7900, kfd=True), allow=True).variant.id == "cpu"  # allowing cannot conjure pins


def test_an_explicit_variant_is_taken_as_asked() -> None:
    choice = choose(hw(), requested="cuda12")
    assert choice.variant.id == "cuda12" and choice.reason == "chosen with --variant cuda12"
    with pytest.raises(SetupRefused, match=r"unknown variant 'cuda9'.*known variants: cpu"):
        choose(hw(), requested="cuda9")


def test_gpu_description_lists_only_what_is_known() -> None:
    assert describe_gpu(GTX1050) == "NVIDIA GeForce GTX 1050 (driver 536.23, compute capability 6.1)"
    assert describe_gpu(GpuInfo(vendor="nvidia", name="Mystery")) == "Mystery"
    assert describe_gpu(IGPU) == "AMD Radeon 890M (gfx1150, integrated)"


@pytest.mark.parametrize(
    "variant_id,platform,allow,message",
    [
        ("cuda12", "darwin", False, "not available on darwin; it supports linux, win32"),
        ("rocm-linux", "linux", True, "no runtime is pinned for it yet"),
        ("rocm-linux", "win32", False, "not available on win32"),
        ("cuda11", "linux", True, "faster-whisper 1.x requires ctranslate2>=4"),
        ("cuda12-blackwell", "linux", False, "--allow-experimental"),
    ],
)
def test_refusals_say_why(variant_id: str, platform: str, allow: bool, message: str) -> None:
    with pytest.raises(SetupRefused) as refused:
        check_installable(get_variant(variant_id), platform, allow)
    assert message in str(refused.value)


def test_refusing_an_experimental_variant_prints_what_is_unverified() -> None:
    with pytest.raises(SetupRefused) as refused:
        check_installable(get_variant("cuda12-blackwell"), "linux", False)
    assert get_variant("cuda12-blackwell").note in str(refused.value)


@pytest.mark.parametrize(
    "variant_id,platform,allow",
    [
        ("cpu", "linux", False),
        ("cpu", "darwin", False),
        ("cuda12", "win32", False),
        ("cuda12-blackwell", "linux", True),
    ],
)
def test_installable_variants_pass(variant_id: str, platform: str, allow: bool) -> None:
    check_installable(get_variant(variant_id), platform, allow)


# --- building commands ------------------------------------------------------------------------


def test_uv_commands_target_the_given_python_and_install_the_lockfile_as_is() -> None:
    installer = Installer(PYTHON, UV)
    lock = Path("/x/cpu.txt")
    assert installer.install_locked(lock).argv == (
        UV,
        "pip",
        "install",
        "--python",
        PYTHON,
        "--no-deps",
        "-r",
        str(lock),
    )
    assert installer.uninstall(["a", "b"]).argv == (UV, "pip", "uninstall", "--python", PYTHON, "a", "b")
    assert installer.upgrade("pkg @ git+https://x").argv == (
        UV, "pip", "install", "--python", PYTHON, "--upgrade", "pkg @ git+https://x",
    )  # fmt: skip


def test_pip_is_the_fallback_when_uv_is_missing() -> None:
    installer = Installer(PYTHON)
    lock = Path("/x/cpu.txt")
    assert installer.install_locked(lock).argv == (
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-r",
        str(lock),
    )
    assert installer.uninstall(["a"]).argv == (PYTHON, "-m", "pip", "uninstall", "-y", "a")
    assert installer.upgrade("pkg").argv == (PYTHON, "-m", "pip", "install", "--upgrade", "pkg")


def test_installer_detection_prefers_uv_then_pip_then_says_what_to_do() -> None:
    assert detect_installer(PYTHON, which=lambda _n: UV, pip_available=lambda: False) == Installer(PYTHON, UV)
    assert detect_installer(PYTHON, which=lambda _n: None, pip_available=lambda: True) == Installer(PYTHON)
    with pytest.raises(InstallerMissing, match="Install uv"):
        detect_installer(PYTHON, which=lambda _n: None, pip_available=lambda: False)


def test_command_display_quotes_what_needs_quoting() -> None:
    assert "with space" in Command(("uv", "-r", "/tmp/with space/x.txt")).display()


def test_conflicting_onnxruntime_builds_are_found_only_when_installed() -> None:
    everything = lambda _n: True  # noqa: E731
    assert conflicting_distributions({"onnxruntime-gpu", "numpy"}, everything) == ["onnxruntime"]
    assert conflicting_distributions({"onnxruntime", "numpy"}, everything) == ["onnxruntime-gpu"]
    assert conflicting_distributions({"onnxruntime-gpu"}, lambda n: n != "onnxruntime") == []
    assert conflicting_distributions({"numpy"}, everything) == []


def plan_for(variant_id: str, installed: set[str], installer: Installer | None = None):
    variant = get_variant(variant_id)
    with lockfile_path(variant) as lock:
        return build_plan(
            Choice(variant, "test"),
            installer or Installer(PYTHON, UV),
            lock,
            installed=installed.__contains__,
        ), lock


def test_a_gpu_runtime_removes_the_cpu_onnxruntime_before_installing() -> None:
    plan, lock = plan_for("cuda12", {"onnxruntime", "numpy"})
    assert [c.argv for c in plan.commands] == [
        (UV, "pip", "uninstall", "--python", PYTHON, "onnxruntime"),
        (UV, "pip", "install", "--python", PYTHON, "--no-deps", "-r", str(lock)),
    ]


def test_switching_back_to_cpu_removes_onnxruntime_gpu() -> None:
    plan, _ = plan_for("cpu", {"onnxruntime-gpu"})
    assert plan.commands[0].argv == (UV, "pip", "uninstall", "--python", PYTHON, "onnxruntime-gpu")
    assert len(plan.commands) == 2


def test_a_clean_environment_needs_one_command_and_the_lockfile_hash_is_kept() -> None:
    plan, lock = plan_for("cuda12", set())
    assert len(plan.commands) == 1
    assert plan.lockfile_sha256 == lockfile_sha256(get_variant("cuda12"))
    assert plan.variant.id == "cuda12" and plan.reason == "test" and lock.name == "cuda12.txt"


def test_plans_use_pip_when_that_is_all_there_is() -> None:
    plan, lock = plan_for("cuda12", {"onnxruntime"}, Installer(PYTHON))
    assert [c.argv[:4] for c in plan.commands] == [
        (PYTHON, "-m", "pip", "uninstall"),
        (PYTHON, "-m", "pip", "install"),
    ]
    assert plan.commands[1].argv[-1] == str(lock)


def test_execution_stops_at_the_first_failing_command() -> None:
    plan, _ = plan_for("cuda12", {"onnxruntime"})
    ran: list[tuple[str, ...]] = []
    said: list[str] = []

    def runner(argv) -> int:
        ran.append(tuple(argv))
        return 5 if len(ran) == 1 else 0

    assert execute(plan, runner, said.append) == 5
    assert len(ran) == 1 and "status 5" in said[-1]


def test_running_a_missing_program_is_a_clean_failure(capsys) -> None:
    assert run_command(["hibiki-asr-no-such-program-xyz"]) == 127
    assert "cannot run hibiki-asr-no-such-program-xyz" in capsys.readouterr().out


# --- the command ------------------------------------------------------------------------------


@pytest.fixture
def uv(monkeypatch):
    monkeypatch.setattr(cli, "detect_installer", lambda python: Installer(python, UV))
    monkeypatch.setattr(cli, "distribution_installed", lambda name: name == "onnxruntime")
    monkeypatch.setattr(cli, "_platform", lambda: "linux")


def test_dry_run_prints_the_commands_and_changes_nothing(env, commands, uv, capsys) -> None:
    env.machine["hardware"] = hw(RTX4090)
    code, out, _ = run(capsys, "setup", "--dry-run")
    assert code == 0 and commands == []
    assert "Runtime variant: cuda12" in out and "why: automatic choice: NVIDIA GeForce RTX 4090" in out
    assert f"{UV} pip uninstall --python" in out and " onnxruntime" in out
    assert "--no-deps -r " in out and "cuda12.txt" in out
    assert "Dry run: nothing was changed." in out
    assert read_variant(env.settings.data_dir) is None  # no state written


def test_setup_installs_records_the_variant_and_prints_the_doctor_report(env, commands, uv, capsys) -> None:
    code, out, _ = run(capsys, "setup", "--yes")
    assert code == 0
    assert [c[:3] for c in commands] == [[UV, "pip", "install"]]
    assert commands[0][-1].endswith("cpu.txt")
    assert read_variant(env.settings.data_dir) == "cpu"
    assert read_lockfile_sha256(env.settings.data_dir) == lockfile_sha256(get_variant("cpu"))
    assert (
        "Installed." in out and "Device   cpu (int8)" in out and "CPU_ONLY" in out
    )  # the same report as doctor


def test_setup_shows_at_once_whether_the_gpu_is_now_usable(env, commands, uv, capsys, monkeypatch) -> None:
    env.machine["hardware"] = hw(RTX4090)
    env.machine["runtime"] = rt(0)  # before: a CPU-only install

    def install(argv) -> int:
        commands.append(list(argv))
        env.machine["runtime"] = rt(1)  # after: CTranslate2 sees the card
        return 0

    monkeypatch.setattr(cli, "command_runner", install)
    code, out, _ = run(capsys, "setup", "--yes")
    assert code == 0 and len(commands) == 2
    assert "runtime variant: cuda12" in out and "Device   cuda (bfloat16), VAD on cpu" in out
    assert "CUDA/ROCm devices seen by CTranslate2: 1" in out and "a GPU was expected" not in out


def test_a_broken_install_is_reported_with_a_failing_exit_code(
    env, commands, uv, capsys, monkeypatch
) -> None:
    env.machine["hardware"] = hw(RTX4090)

    def install(argv) -> int:
        commands.append(list(argv))
        env.machine["runtime"] = rt(
            0, ctranslate2_version=None, ctranslate2_error="OSError: libcublas.so.12 missing"
        )
        return 0

    monkeypatch.setattr(cli, "command_runner", install)
    code, out, _ = run(capsys, "setup", "--yes")
    assert code == 1 and "CT2_IMPORT_FAILED" in out and "libcublas.so.12" in out


def test_a_failing_command_stops_setup_and_records_nothing(env, uv, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "command_runner", lambda argv: 7)
    code, out, _ = run(capsys, "setup", "--yes")
    assert code == 7 and "status 7" in out and "Installed." not in out
    assert read_variant(env.settings.data_dir) is None
    assert not (env.settings.data_dir / "variant.json").exists()


def test_a_refused_variant_prints_the_note_and_runs_nothing(env, commands, uv, capsys) -> None:
    code, _, err = run(capsys, "setup", "--variant", "rocm-linux", "--allow-experimental")
    assert code == 2 and "no runtime is pinned" in err and "GitHub releases page" in err

    code, _, err = run(capsys, "setup", "--variant", "cuda12-blackwell", "--yes")
    assert code == 2 and "--allow-experimental" in err

    code, _, err = run(capsys, "setup", "--variant", "nope", "--yes")
    assert code == 2 and "unknown variant 'nope'" in err
    assert commands == [] and read_variant(env.settings.data_dir) is None


def test_an_experimental_variant_installs_with_the_flag_and_says_what_is_unverified(
    env, commands, uv, capsys
) -> None:
    code, out, _ = run(capsys, "setup", "--variant", "cuda12-blackwell", "--allow-experimental", "--yes")
    assert code == 0
    assert "experimental: The pins resolve" in out
    assert commands[-1][-1].endswith("cuda12-blackwell.txt")
    assert read_variant(env.settings.data_dir) == "cuda12-blackwell"


def test_without_a_terminal_setup_asks_for_yes_instead_of_hanging(env, commands, uv, capsys) -> None:
    code, _, err = run(capsys, "setup")
    assert code == 2 and "pass --yes" in err and commands == []


def test_answering_at_the_prompt(env, commands, uv, capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    code, out, _ = run(capsys, "setup")
    assert code == 1 and "Cancelled; nothing was changed." in out and commands == []

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    code, _, _ = run(capsys, "setup")
    assert code == 0 and len(commands) == 1


def test_if_changed_skips_a_runtime_that_is_current(env, commands, uv, capsys) -> None:
    assert run(capsys, "setup", "--variant", "cpu", "--yes")[0] == 0
    assert len(commands) == 1

    code, out, _ = run(capsys, "setup", "--variant", "cpu", "--yes", "--if-changed")
    assert code == 0 and "nothing to do" in out and len(commands) == 1

    state = env.settings.data_dir / "variant.json"
    state.write_text(json.dumps({"variant": "cpu", "lockfile_sha256": "0" * 64}))  # an older lockfile
    assert run(capsys, "setup", "--variant", "cpu", "--yes", "--if-changed")[0] == 0
    assert len(commands) == 2

    # another variant is never "current"
    assert run(capsys, "setup", "--variant", "cuda12", "--yes", "--if-changed")[0] == 0
    assert len(commands) == 4  # uninstall onnxruntime + install


def test_pip_is_used_when_uv_is_not_installed(env, commands, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "detect_installer", lambda python: Installer(python))
    monkeypatch.setattr(cli, "distribution_installed", lambda name: False)
    assert run(capsys, "setup", "--yes")[0] == 0
    assert commands[0][1:4] == ["-m", "pip", "install"] and "--no-deps" in commands[0]


def test_setup_says_so_when_neither_uv_nor_pip_exist(env, commands, capsys, monkeypatch) -> None:
    def missing(_python: str):
        raise InstallerMissing("neither uv nor pip is available")

    monkeypatch.setattr(cli, "detect_installer", missing)
    code, _, err = run(capsys, "setup", "--yes")
    assert code == 2 and "neither uv nor pip" in err and commands == []


def test_setup_puts_the_pip_installed_cuda_libraries_on_the_path_before_probing(
    env, uv, capsys, monkeypatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "prepare_environment", lambda: events.append("prepare"))
    monkeypatch.setattr(cli, "command_runner", lambda argv: (events.append("install"), 0)[1])
    diagnostics = Engine.diagnostics

    def spy(self, **kwargs):
        events.append("probe")
        return diagnostics(self, **kwargs)

    monkeypatch.setattr(Engine, "diagnostics", spy)
    assert run(capsys, "setup", "--yes")[0] == 0
    # once at start-up, then again because the install may just have added the libraries
    assert events == ["prepare", "install", "prepare", "probe"]
