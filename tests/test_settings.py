"""Settings: layering, validation, the serving guard, and editing the config file."""

from __future__ import annotations

from pathlib import Path

import pytest

from hibiki_asr.settings import (
    Settings,
    config_file_path,
    default_config_dir,
    default_data_dir,
    load_settings,
    write_config_value,
)

NO_FILE = Path("/nonexistent/hibiki-asr.toml")


def load(env=None, overrides=None, path: Path = NO_FILE) -> Settings:
    return load_settings(env=env or {}, overrides=overrides, config_path=path)


def test_defaults_are_safe_and_zero_config() -> None:
    s = load()
    assert (s.host, s.port, s.token) == ("127.0.0.1", 8001, None)  # loopback: works without any setup
    assert (s.device, s.compute_type, s.allow_cpu_fallback) == ("auto", "auto", True)
    assert (s.download_threads, s.hf_endpoint, s.hf_mirrors) == (
        4,
        "https://huggingface.co",
        ["https://hf-mirror.com"],
    )
    assert (s.vad.threshold, s.vad.min_silence_duration_ms, s.merge.max_gap_ms) == (0.5, 100, 2000)
    assert s.resolved_models_dir == s.data_dir / "models"


def test_sources_are_layered_file_then_env_then_command_line(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text('port = 9001\ndevice = "cpu"\n[vad]\nthreshold = 0.3\nspeech_pad_ms = 50\n')

    s = load(
        env={"HIBIKI_ASR_DEVICE": "cuda", "HIBIKI_ASR_VAD__THRESHOLD": "0.7"},
        overrides={"port": 9100},
        path=config,
    )

    assert s.port == 9100  # command line beats the file
    assert s.device == "cuda"  # environment beats the file
    assert (
        s.vad.threshold == 0.7 and s.vad.speech_pad_ms == 50
    )  # nested values merge instead of replacing the table


def test_command_line_values_that_were_not_given_do_not_override() -> None:
    assert load(env={"HIBIKI_ASR_PORT": "9002"}, overrides={"port": None, "host": None}).port == 9002


def test_environment_values_are_typed() -> None:
    s = load(
        env={
            "HIBIKI_ASR_ALLOW_CPU_FALLBACK": "false",
            "HIBIKI_ASR_DOWNLOAD_THREADS": "8",
            "HIBIKI_ASR_CHUNK_TARGET_S": "20",
        }
    )
    assert (s.allow_cpu_fallback, s.download_threads, s.chunk_target_s) == (False, 8, 20.0)


def test_standard_huggingface_variables_are_honoured_but_ours_win() -> None:
    s = load(env={"HF_ENDPOINT": "https://hf-mirror.com", "HF_TOKEN": "hf_abc"})
    assert (s.hf_endpoint, s.hf_token) == ("https://hf-mirror.com", "hf_abc")
    both = load(env={"HF_ENDPOINT": "https://a.example", "HIBIKI_ASR_HF_ENDPOINT": "https://b.example"})
    assert both.hf_endpoint == "https://b.example"


def test_the_variant_variable_of_the_docker_images_is_not_a_setting() -> None:
    # HIBIKI_ASR_VARIANT names the installed runtime (see provision/state.py); it must not fail every command
    assert load(env={"HIBIKI_ASR_VARIANT": "cuda12", "HIBIKI_ASR_PORT": "9001"}).port == 9001


def test_mirrors_can_be_a_comma_separated_string() -> None:
    mirrors = load(env={"HIBIKI_ASR_HF_MIRRORS": "https://a.example, https://b.example"}).hf_mirrors
    assert mirrors == ["https://a.example", "https://b.example"]


def test_device_and_log_level_are_case_insensitive() -> None:
    s = load(env={"HIBIKI_ASR_DEVICE": " CUDA ", "HIBIKI_ASR_LOG_LEVEL": "debug"})
    assert (s.device, s.log_level) == ("cuda", "DEBUG")


@pytest.mark.parametrize(
    "env",
    [
        {"HIBIKI_ASR_DEVICE": "tpu"},
        {"HIBIKI_ASR_PORT": "0"},
        {"HIBIKI_ASR_VAD__THRESHOLD": "1.5"},
        {"HIBIKI_ASR_DOWNLOAD_THREADS": "99"},
        {"HIBIKI_ASR_NOT_A_SETTING": "x"},
        {"HIBIKI_ASR_CHUNK_TARGET_S": "45"},
    ],
)
def test_invalid_settings_are_rejected(env) -> None:
    with pytest.raises(ValueError):
        load(env=env)


def test_a_broken_config_file_names_the_file(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text("port = [")
    with pytest.raises(ValueError, match=r"c\.toml: invalid TOML"):
        load(path=config)


def test_listening_beyond_loopback_requires_a_token() -> None:
    load(overrides={"host": "127.0.0.1"}).validate_for_serving()
    load(overrides={"host": "localhost"}).validate_for_serving()
    with pytest.raises(ValueError, match="without a token"):
        load(overrides={"host": "0.0.0.0"}).validate_for_serving()
    load(overrides={"host": "0.0.0.0", "token": "s3cret"}).validate_for_serving()


def test_default_directories_per_platform() -> None:
    env = {
        "HOME": "/h",
        "APPDATA": "C:/Users/u/AppData/Roaming",
        "LOCALAPPDATA": "C:/Users/u/AppData/Local",
        "XDG_CONFIG_HOME": "/xdg/c",
        "XDG_DATA_HOME": "/xdg/d",
    }
    assert default_config_dir(env, "linux") == Path("/xdg/c/hibiki-asr")
    assert default_data_dir(env, "linux") == Path("/xdg/d/hibiki-asr")
    assert default_config_dir(env, "win32") == Path("C:/Users/u/AppData/Roaming/hibiki-asr")
    assert default_data_dir(env, "win32") == Path("C:/Users/u/AppData/Local/hibiki-asr")
    assert "Application Support" in str(default_data_dir(env, "darwin"))


def test_config_file_location_can_be_overridden() -> None:
    assert config_file_path({"HIBIKI_ASR_CONFIG": "/etc/asr.toml"}) == Path("/etc/asr.toml")
    assert config_file_path({"XDG_CONFIG_HOME": "/x"}).name == "hibiki-asr.toml"


# --- editing the config file ------------------------------------------------------------------------


def test_set_writes_typed_values_and_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "conf" / "hibiki-asr.toml"
    write_config_value(path, "device", "cpu")
    write_config_value(path, "vad.threshold", "0.4")
    write_config_value(path, "download_threads", "8")
    write_config_value(path, "hf_endpoint", "https://hf-mirror.com")
    write_config_value(path, "allow_cpu_fallback", "false")

    text = path.read_text()
    assert (
        'device = "cpu"' in text and "download_threads = 8" in text and "allow_cpu_fallback = false" in text
    )
    assert "[vad]\nthreshold = 0.4" in text
    s = load(path=path)
    assert (s.device, s.vad.threshold, s.download_threads, s.hf_endpoint, s.allow_cpu_fallback) == (
        "cpu",
        0.4,
        8,
        "https://hf-mirror.com",
        False,
    )


def test_set_rejects_bad_input_without_touching_the_file(tmp_path: Path) -> None:
    path = tmp_path / "hibiki-asr.toml"
    write_config_value(path, "device", "cpu")
    before = path.read_text()
    for key, value in [("device", "tpu"), ("no_such_key", "1"), ("vad.threshold", "9"), ("vad.nope", "1")]:
        with pytest.raises(ValueError):
            write_config_value(path, key, value)
    assert path.read_text() == before


def test_generation_values_are_parsed_as_json(tmp_path: Path) -> None:
    path = tmp_path / "hibiki-asr.toml"
    write_config_value(path, "generation.beam_size", "5")
    write_config_value(path, "generation.temperature", "[0.0, 0.2]")
    write_config_value(path, "generation.initial_prompt", "こんにちは")
    s = load(path=path)
    assert s.generation == {"beam_size": 5, "temperature": [0.0, 0.2], "initial_prompt": "こんにちは"}
