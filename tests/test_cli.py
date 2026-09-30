"""The command line, driven through main() with a fake Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import RTX4090, hw, rt
from helpers import run_cli as run
from hibiki_asr import cli


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.main(["--version"])
    assert exit_.value.code == 0 and "hibiki-asr" in capsys.readouterr().out


def test_models_download_list_use_and_delete(env, capsys) -> None:
    code, out, _ = run(capsys, "models", "list")
    assert code == 0 and "tiny" in out and "not_installed" in out

    code, out, err = run(capsys, "models", "download", "tiny@v1", "--threads", "2")
    assert code == 0 and "installed tiny@v1" in out

    code, out, _ = run(capsys, "models", "list")
    assert "installed" in out and "<- active" in out

    assert run(capsys, "models", "download", "tiny@v2")[0] == 0
    code, out, _ = run(capsys, "models", "use", "tiny@v2")
    assert code == 0 and "active version is now v2" in out

    code, out, _ = run(capsys, "models", "verify", "tiny@v2")
    assert code == 0 and out.strip() == "ok"

    code, _, err = run(capsys, "models", "delete", "tiny@v2")
    assert code == 1 and "active version" in err  # protected while another version exists
    assert run(capsys, "models", "delete", "tiny@v1")[0] == 0


def test_models_list_json(env, capsys) -> None:
    code, out, _ = run(capsys, "models", "list", "--json")
    assert code == 0 and json.loads(out)[0]["id"] == "tiny"


def test_download_from_a_chosen_mirror_only(env, capsys) -> None:
    code, _, _ = run(
        capsys, "models", "download", "tiny@v1", "--endpoint", "https://custom.example", "--no-fallback"
    )
    assert code == 0 and env.hub.hosts_used() == {"custom.example"}

    env.hub.down_hosts = {"custom.example"}
    code, _, err = run(
        capsys, "models", "download", "tiny@v2", "--endpoint", "https://custom.example", "--no-fallback"
    )
    assert code == 1 and "error:" in err


def test_a_failed_download_exits_nonzero_with_the_reason(env, capsys) -> None:
    env.hub.down_hosts = {"huggingface.co", "hf-mirror.com"}
    code, _, err = run(capsys, "models", "download", "tiny@v1")
    assert code == 1 and "HTTP 503" in err


def test_download_needs_a_version(env, capsys) -> None:
    code, _, err = run(capsys, "models", "download", "tiny")
    assert code == 1 and "ID@VERSION" in err


def test_sources_command_reports_reachability(env, capsys) -> None:
    code, out, _ = run(capsys, "models", "sources")
    assert code == 0 and "Hugging Face directly reachable: yes" in out and "recommended:" in out

    env.hub.down_hosts = {"huggingface.co"}
    _, out, _ = run(capsys, "models", "sources", "--endpoint", "https://my-mirror.example")
    assert (
        "reachable: NO" in out
        and "https://my-mirror.example" in out
        and "recommended: https://hf-mirror.com" in out
    )

    _, out, _ = run(capsys, "models", "sources", "--json")
    assert json.loads(out)["huggingface_reachable"] is False


def test_doctor_prints_the_report_and_json(env, capsys) -> None:
    code, out, _ = run(capsys, "doctor")
    assert code == 0 and "Device   cpu" in out and "CPU_ONLY" in out

    env.machine["hardware"] = hw(RTX4090, container=True)
    code, out, _ = run(capsys, "doctor", "--json")
    body = json.loads(out)
    assert code == 0 and body["selection"]["degraded"] is True
    assert "CT2_NO_GPU_SUPPORT" in [f["code"] for f in body["findings"]]


def test_doctor_exits_nonzero_when_there_is_an_error(env, capsys) -> None:
    env.machine["runtime"] = rt(
        0, ctranslate2_version=None, ctranslate2_error="OSError: libcudnn.so.9 missing"
    )
    code, out, _ = run(capsys, "doctor")
    assert code == 1 and "CT2_IMPORT_FAILED" in out


def test_config_show_hides_secrets_and_set_persists(env, capsys, monkeypatch) -> None:
    monkeypatch.setenv("HIBIKI_ASR_TOKEN", "topsecret")
    code, out, err = run(capsys, "config", "show")
    body = json.loads(out)
    assert code == 0 and body["token"] == "********" and "not created yet" in err

    code, out, _ = run(capsys, "config", "set", "hf_endpoint", "https://hf-mirror.com")
    assert code == 0 and "restart the engine" in out
    code, out, _ = run(capsys, "config", "path")
    assert Path(out.strip()).read_text().count("hf_endpoint") == 1

    code, _, err = run(capsys, "config", "set", "device", "tpu")
    assert code == 2 and "error:" in err
    assert run(capsys, "config", "set", "device")[0] == 2  # missing value


def test_invalid_environment_is_reported_not_a_traceback(env, capsys, monkeypatch) -> None:
    monkeypatch.setenv("HIBIKI_ASR_DEVICE", "tpu")
    code, _, err = run(capsys, "doctor")
    assert code == 2 and "invalid settings" in err


def test_serve_refuses_to_listen_on_the_network_without_a_token(env, capsys) -> None:
    code, _, err = run(capsys, "serve", "--host", "0.0.0.0")
    assert code == 2 and "without a token" in err


def test_serve_starts_uvicorn_with_the_engine_app(env, capsys, monkeypatch) -> None:
    import uvicorn

    started: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    monkeypatch.setenv("HIBIKI_ASR_TOKEN", "abc")
    code, _, err = run(capsys, "serve", "--host", "0.0.0.0", "--port", "9123")
    assert code == 0 and (started["host"], started["port"]) == ("0.0.0.0", 9123)
    assert "http://127.0.0.1:9123" in err and "token: abc" in err
    assert started["app"].title == "hibiki-asr"
