"""Adopting model files that are already on disk: `ModelManager.import_version` and `models import`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers import run_cli as run
from hibiki_asr.models.adopt import find_problems, stage_files
from hibiki_asr.models.catalog import Ref, merge_entries, parse_local_toml
from hibiki_asr.models.manager import Conflict, ModelManager, NotFound, Unprocessable
from hibiki_asr.models.schema import VersionStatus
from hibiki_asr.models.store import MARKER
from test_models import REPOS, World, sha

TINY_V1 = ("acme/tiny", "a" * 40)
TINY_V2 = ("acme/tiny", "b" * 40)
VAD = ("acme/vad", "c" * 40)
FE = ("acme/fe", "d" * 40)


def folder(tmp_path: Path, name: str, repo: tuple[str, str]) -> Path:
    """A directory holding exactly the files of one catalog version, as another tool's model folder would."""
    path = tmp_path / name
    path.mkdir(parents=True)
    for file, data in REPOS[repo].items():
        (path / file).write_bytes(data)
    return path


def flip_a_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))


@pytest.fixture
def world(tmp_path: Path):
    w = World(tmp_path)
    yield w
    w.manager.shutdown()


def store_files(world: World, ref: Ref) -> list[str]:
    root = world.store.version_dir(ref)
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def nothing_installed(world: World, ref: Ref) -> bool:
    return not world.store.version_dir(ref).exists() and not world.store.staging_dir(ref).exists()


# --- a good import ----------------------------------------------------------------------------------


def test_importing_adopts_the_files_without_downloading(world: World, tmp_path: Path) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    ref = Ref("tiny", "v1")
    result = world.manager.import_version(ref, source)

    assert (result.ref, result.files, result.moved) == (ref, 2, False)
    assert result.size_bytes == sum(len(d) for d in REPOS[TINY_V1].values())
    assert world.hub.requests == []  # nothing came over the network
    assert store_files(world, ref) == sorted([MARKER, "config.json", "model.bin"])
    assert (world.store.version_dir(ref) / "model.bin").read_bytes() == REPOS[TINY_V1]["model.bin"]
    assert world.store.is_intact(ref) and world.manager.verify(ref).ok
    assert world.store.active("tiny") == "v1"  # the first installed version becomes the active one
    assert not world.store.staging_dir(ref).exists()
    marker = world.store.read_marker(ref)
    assert marker is not None and {f["path"]: f["sha256"] for f in marker["files"]} == {
        name: sha(data) for name, data in REPOS[TINY_V1].items()
    }
    # the folder the files came from is untouched
    assert sorted(p.name for p in source.iterdir()) == ["config.json", "model.bin"]


def test_a_second_version_does_not_steal_the_active_one(world: World, tmp_path: Path) -> None:
    world.manager.import_version(Ref("tiny", "v1"), folder(tmp_path, "a", TINY_V1))
    world.manager.import_version(Ref("tiny", "v2"), folder(tmp_path, "b", TINY_V2))
    assert world.store.active("tiny") == "v1"
    assert world.store.installed_versions("tiny") == ["v1", "v2"]


def test_components_can_be_imported_and_complete_the_model(world: World, tmp_path: Path) -> None:
    manager = world.manager
    manager.import_version(Ref("tiny", "v1"), folder(tmp_path, "model", TINY_V1))
    manager.import_version(Ref("vad", "1"), folder(tmp_path, "vad", VAD))
    manager.import_version(Ref("fe", "1"), folder(tmp_path, "fe", FE))
    assert world.store.active("vad") is None  # components have no active version
    resolved = manager.resolve("tiny")
    assert resolved.ref == Ref("tiny", "v1") and set(resolved.components) == {"vad", "fe"}
    assert world.hub.requests == []


def test_downloading_after_an_import_fetches_only_what_is_missing(world: World, tmp_path: Path) -> None:
    world.manager.import_version(Ref("tiny", "v1"), folder(tmp_path, "model", TINY_V1))
    assert world.install(Ref("tiny", "v1")).state.value == "succeeded"
    assert {name for _, name, _ in world.hub.requests} == {
        "model.onnx",
        "model_metadata.json",
        "preprocessor_config.json",
    }  # the components; not the model that was imported


def test_the_import_reports_progress_for_files_it_hashes(world: World, tmp_path: Path) -> None:
    seen: list[str] = []
    world.manager.import_version(Ref("tiny", "v1"), folder(tmp_path, "s", TINY_V1), on_file=seen.append)
    assert sorted(seen) == ["config.json", "model.bin"]


# --- a wrong import installs nothing ----------------------------------------------------------------


def test_a_file_with_the_wrong_content_rejects_the_whole_import(world: World, tmp_path: Path) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    flip_a_byte(source / "model.bin")  # same size, different bytes
    ref = Ref("tiny", "v1")
    with pytest.raises(Unprocessable) as rejected:
        world.manager.import_version(ref, source)
    assert rejected.value.code == "IMPORT_REJECTED" and rejected.value.status == 422
    assert "model.bin: sha256 does not match the catalog" in rejected.value.message
    assert "nothing was installed" in rejected.value.message
    assert nothing_installed(world, ref) and world.store.active("tiny") is None
    assert world.manager.get_model("tiny").versions[0].status is VersionStatus.not_installed
    assert (source / "config.json").read_bytes() == REPOS[TINY_V1][
        "config.json"
    ]  # the good file was not taken


def test_every_problem_is_reported_at_once(world: World, tmp_path: Path) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    (source / "config.json").unlink()
    (source / "model.bin").write_bytes(b"short")
    with pytest.raises(Unprocessable) as rejected:
        world.manager.import_version(Ref("tiny", "v1"), source)
    assert "config.json: missing" in rejected.value.message
    assert f"model.bin: expected {len(REPOS[TINY_V1]['model.bin'])} bytes, found 5" in rejected.value.message


def test_a_truncated_file_is_caught_by_its_size_without_hashing_it(world: World, tmp_path: Path) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    (source / "model.bin").write_bytes(b"short")
    hashed: list[str] = []
    problems = find_problems(world.manager.catalog.resolve(Ref("tiny", "v1"))[1], source, hashed.append)
    assert len(problems) == 1 and hashed == ["config.json"]


def test_files_of_another_version_are_not_accepted(world: World, tmp_path: Path) -> None:
    v2_files = folder(tmp_path, "v2", TINY_V2)
    with pytest.raises(Unprocessable, match="does not match tiny@v1"):
        world.manager.import_version(Ref("tiny", "v1"), v2_files)


def test_a_directory_that_does_not_exist_is_refused(world: World, tmp_path: Path) -> None:
    with pytest.raises(Unprocessable) as refused:
        world.manager.import_version(Ref("tiny", "v1"), tmp_path / "nope")
    assert refused.value.code == "IMPORT_SOURCE_NOT_FOUND"


def test_an_unknown_version_is_refused(world: World, tmp_path: Path) -> None:
    with pytest.raises(NotFound):
        world.manager.import_version(Ref("tiny", "v9"), folder(tmp_path, "s", TINY_V1))


def test_the_model_store_itself_is_not_a_valid_source(world: World, tmp_path: Path) -> None:
    world.manager.import_version(Ref("tiny", "v1"), folder(tmp_path, "s", TINY_V1))
    installed = world.store.version_dir(Ref("tiny", "v1"))
    for inside in (installed, world.store.root, world.store.staging_dir(Ref("tiny", "v2"))):
        inside.mkdir(parents=True, exist_ok=True)
        with pytest.raises(Unprocessable) as refused:
            world.manager.import_version(Ref("tiny", "v2"), inside)
        assert refused.value.code == "IMPORT_SOURCE_IN_STORE"


def test_an_installed_version_is_not_replaced(world: World, tmp_path: Path) -> None:
    ref = Ref("tiny", "v1")
    world.manager.import_version(ref, folder(tmp_path, "one", TINY_V1))
    with pytest.raises(Conflict) as refused:
        world.manager.import_version(ref, folder(tmp_path, "two", TINY_V1))
    assert refused.value.code == "ALREADY_INSTALLED"


def test_a_damaged_install_is_replaced_by_a_good_import(world: World, tmp_path: Path) -> None:
    ref = Ref("tiny", "v1")
    world.manager.import_version(ref, folder(tmp_path, "one", TINY_V1))
    (world.store.version_dir(ref) / "model.bin").unlink()
    assert world.store.is_installed(ref) and not world.store.is_intact(ref)
    world.manager.import_version(ref, folder(tmp_path, "two", TINY_V1))
    assert world.store.is_intact(ref) and world.manager.verify(ref).ok


def test_leftovers_of_an_interrupted_download_do_not_end_up_in_the_install(
    world: World, tmp_path: Path
) -> None:
    ref = Ref("tiny", "v1")
    staging = world.store.staging_dir(ref)
    staging.mkdir(parents=True)
    (staging / "model.bin").write_bytes(b"half a download")
    (staging / "stray.txt").write_text("x")
    world.manager.import_version(ref, folder(tmp_path, "s", TINY_V1))
    assert store_files(world, ref) == sorted([MARKER, "config.json", "model.bin"])
    assert world.manager.verify(ref).ok


# --- moving -----------------------------------------------------------------------------------------


def test_move_takes_the_files_out_of_the_folder(world: World, tmp_path: Path) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    ref = Ref("tiny", "v1")
    result = world.manager.import_version(ref, source, move=True)
    assert result.moved and list(source.iterdir()) == []
    assert world.manager.verify(ref).ok


def test_a_failed_move_gives_the_files_back(world: World, tmp_path: Path, monkeypatch) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    ref = Ref("tiny", "v1")

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(world.store, "commit", fail)  # after every file was renamed into staging
    with pytest.raises(OSError, match="disk full"):
        world.manager.import_version(ref, source, move=True)
    assert nothing_installed(world, ref)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == REPOS[TINY_V1]


def test_move_across_file_systems_copies_and_deletes_only_after_the_commit(
    world: World, tmp_path: Path, monkeypatch
) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    real_replace = os.replace

    def cross_device(src, dst):
        if Path(src).parent == source:
            raise OSError(18, "Invalid cross-device link")
        real_replace(src, dst)

    monkeypatch.setattr("hibiki_asr.models.adopt.os.replace", cross_device)
    seen_at_commit: list[list[str]] = []
    commit = world.store.commit

    def spying_commit(ref, spec):
        seen_at_commit.append(sorted(p.name for p in source.iterdir()))
        commit(ref, spec)

    monkeypatch.setattr(world.store, "commit", spying_commit)
    ref = Ref("tiny", "v1")
    world.manager.import_version(ref, source, move=True)
    assert seen_at_commit == [
        ["config.json", "model.bin"]
    ]  # the originals were still there at the commit ...
    assert list(source.iterdir()) == []  # ... and gone after it
    assert world.manager.verify(ref).ok


def test_a_failed_copy_during_a_move_keeps_the_originals(world: World, tmp_path: Path, monkeypatch) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    ref = Ref("tiny", "v1")

    def no_rename_no_copy(src, dst):
        raise OSError(18, "Invalid cross-device link")

    def no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("hibiki_asr.models.adopt.os.replace", no_rename_no_copy)
    monkeypatch.setattr("hibiki_asr.models.adopt.shutil.copy2", no_space)
    with pytest.raises(OSError, match="No space"):
        world.manager.import_version(ref, source, move=True)
    assert nothing_installed(world, ref)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == REPOS[TINY_V1]


def test_staging_rolls_back_renames_when_a_later_file_fails(
    world: World, tmp_path: Path, monkeypatch
) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    spec = world.manager.catalog.resolve(Ref("tiny", "v1"))[1]
    staging = tmp_path / "staging"
    staging.mkdir()
    real_replace = os.replace
    calls = []

    def second_rename_fails(src, dst):
        calls.append(src)
        if len(calls) == 2 and Path(src).parent == source:
            raise OSError(13, "permission denied")  # falls back to copy, which then fails too
        real_replace(src, dst)

    def copy_fails(*_a, **_k):
        raise OSError(13, "permission denied")

    monkeypatch.setattr("hibiki_asr.models.adopt.os.replace", second_rename_fails)
    monkeypatch.setattr("hibiki_asr.models.adopt.shutil.copy2", copy_fails)
    with pytest.raises(OSError, match="permission denied"):
        stage_files(spec, source, staging, move=True)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == REPOS[TINY_V1]  # the first rename was undone


# --- the user's own catalog -------------------------------------------------------------------------


def test_files_without_checksums_and_nested_paths(world: World, tmp_path: Path) -> None:
    local = parse_local_toml(
        """
schema = 1
[[entry]]
id = "mine"
kind = "model"
display_name = "Mine"
task = "transcribe"
source_languages = ["ja"]
output_languages = ["ja"]
[[entry.versions]]
version = "1"
repo = "me/mine"
revision = "main"
files = [{ path = "config.json" }, { path = "onnx/model.bin" }]
"""
    )
    manager = ModelManager(
        world.settings,
        merge_entries(world.manager.catalog.entries.values(), local),
        world.store,
        client_factory=world.hub.client,
    )
    source = tmp_path / "mine"
    (source / "onnx").mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "onnx" / "model.bin").write_bytes(b"anything")
    result = manager.import_version(Ref("mine", "1"), source)
    assert result.files == 2 and result.size_bytes == 0  # sizes are unknown, so unchecked
    assert store_files(world, Ref("mine", "1")) == sorted([MARKER, "config.json", "onnx/model.bin"])
    assert world.store.active("mine") == "1"
    manager.shutdown()


# --- the command ------------------------------------------------------------------------------------


def test_models_import_command(env, tmp_path: Path, capsys) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    code, out, err = run(capsys, "models", "import", str(source), "--model", "tiny@v1")
    assert code == 0
    assert "imported tiny@v1: 2 files" in out and "copied into" in out
    assert "checking model.bin" in err and "checking config.json" in err
    assert "tiny@v1 also needs vad@1, fe@1" in out and "models download tiny@v1" in out
    assert env.hub.requests == []

    code, out, _ = run(capsys, "models", "list")
    assert "installed" in out and "<- active" in out
    assert run(capsys, "models", "verify", "tiny@v1")[1].strip() == "ok"

    # the components are then the only thing left to download
    assert run(capsys, "models", "download", "tiny@v1")[0] == 0
    assert "model.bin" not in {name for _, name, _ in env.hub.requests}


def test_models_import_of_a_component_prints_no_note(env, tmp_path: Path, capsys) -> None:
    code, out, _ = run(capsys, "models", "import", str(folder(tmp_path, "v", VAD)), "--model", "vad@1")
    assert code == 0 and "imported vad@1" in out and "also needs" not in out


def test_models_import_reports_a_wrong_file_and_exits_nonzero(env, tmp_path: Path, capsys) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    flip_a_byte(source / "model.bin")
    code, out, err = run(capsys, "models", "import", str(source), "--model", "tiny@v1")
    assert code == 1 and out == ""
    assert "model.bin: sha256 does not match the catalog" in err and "nothing was installed" in err
    assert not env.store.version_dir(Ref("tiny", "v1")).exists()
    assert not env.store.staging_dir(Ref("tiny", "v1")).exists()
    assert (source / "config.json").is_file()  # nothing was taken from the folder


def test_models_import_move(env, tmp_path: Path, capsys) -> None:
    source = folder(tmp_path, "upstream", TINY_V1)
    code, out, _ = run(capsys, "models", "import", str(source), "--model", "tiny@v1", "--move")
    assert code == 0 and "moved into" in out and list(source.iterdir()) == []


def test_models_import_needs_a_version(env, tmp_path: Path, capsys) -> None:
    code, _, err = run(capsys, "models", "import", str(tmp_path), "--model", "tiny")
    assert code == 1 and "ID@VERSION" in err
