"""Fixtures shared by the command line tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import hw, rt
from hibiki_asr import cli
from hibiki_asr.engine import Engine
from hibiki_asr.models.manager import ModelManager
from hibiki_asr.models.store import ModelStore
from test_models import World


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """A private data/config directory and an engine factory backed by the fake hub."""
    world = World(tmp_path)
    world.manager.shutdown()
    monkeypatch.setenv("HIBIKI_ASR_DATA_DIR", str(world.settings.data_dir))
    monkeypatch.setenv("HIBIKI_ASR_CONFIG", str(tmp_path / "config" / "hibiki-asr.toml"))
    monkeypatch.setenv("HIBIKI_ASR_HF_MIRRORS", "https://hf-mirror.com")
    monkeypatch.delenv("HIBIKI_ASR_VARIANT", raising=False)
    machine = {"hardware": hw(), "runtime": rt(0)}

    def make_engine(settings) -> Engine:
        # every command is a new process in real life, so build a fresh manager over the same store each time
        manager = ModelManager(
            settings,
            world.manager.catalog,
            ModelStore(settings.resolved_models_dir),
            client_factory=world.hub.client,
            sleep=lambda _s: None,
        )
        return Engine(
            settings,
            models=manager,
            hardware_probe=lambda: machine["hardware"],
            runtime_probe=lambda: machine["runtime"],
        )

    monkeypatch.setattr(cli, "Engine", make_engine)
    world.machine = machine
    return world


def _refuse(argv):
    raise AssertionError(f"a test tried to run a real command: {list(argv)}")


@pytest.fixture(autouse=True)
def no_real_commands(monkeypatch) -> None:
    """Nothing the CLI would install, upgrade or start may really run in a test."""
    monkeypatch.setattr(cli, "command_runner", _refuse)


@pytest.fixture
def commands(monkeypatch) -> list[list[str]]:
    """Record the commands the CLI would run (each 'succeeds') instead of running them."""
    ran: list[list[str]] = []

    def record(argv) -> int:
        ran.append(list(argv))
        return 0

    monkeypatch.setattr(cli, "command_runner", record)
    return ran
