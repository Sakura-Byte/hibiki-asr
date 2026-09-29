"""Catalog, store and manager: install, versions, dependencies, downloads, sources, refresh."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from fakehub import FakeHub

from hibiki_asr.models.catalog import (
    CatalogError,
    Ref,
    load_bundled,
    merge_entries,
    parse_catalog,
    parse_local_toml,
)
from hibiki_asr.models.manager import (
    Conflict,
    ModelManager,
    ModelNotInstalled,
    NotFound,
    load_catalog,
)
from hibiki_asr.models.schema import DownloadRequest, DownloadState, VersionStatus
from hibiki_asr.models.store import ModelStore
from hibiki_asr.settings import Settings

HF = "huggingface.co"
MIRROR = "hf-mirror.com"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_entry(path: str, data: bytes) -> dict:
    return {"path": path, "size": len(data), "sha256": sha(data)}


REPOS = {
    ("acme/tiny", "a" * 40): {"model.bin": b"weights-v1" * 100, "config.json": b'{"v": 1}'},
    ("acme/tiny", "b" * 40): {"model.bin": b"weights-v2" * 100, "config.json": b'{"v": 2}'},
    ("acme/vad", "c" * 40): {"model.onnx": b"onnx" * 200, "model_metadata.json": b"{}"},
    ("acme/fe", "d" * 40): {"preprocessor_config.json": b'{"feature_size": 80}'},
}


def version(label: str, repo: str, rev: str) -> dict:
    return {
        "version": label,
        "repo": repo,
        "revision": rev,
        "files": [file_entry(p, d) for p, d in REPOS[(repo, rev)].items()],
    }


def catalog_data(*, with_v2: bool = True) -> dict:
    versions = [version("v1", "acme/tiny", "a" * 40)]
    if with_v2:
        versions.append(version("v2", "acme/tiny", "b" * 40))
    return {
        "schema": 1,
        "entries": [
            {
                "id": "tiny",
                "kind": "model",
                "display_name": "Tiny",
                "task": "translate",
                "source_languages": ["ja"],
                "output_languages": ["zh"],
                "requires": ["vad@1", "fe@1"],
                "versions": versions,
            },
            {"id": "vad", "kind": "component", "versions": [version("1", "acme/vad", "c" * 40)]},
            {"id": "fe", "kind": "component", "versions": [version("1", "acme/fe", "d" * 40)]},
        ],
    }


class World:
    def __init__(self, tmp_path: Path, *, with_v2: bool = True, **settings) -> None:
        self.hub = FakeHub({})
        self.hub.repos.clear()
        for (repo, rev), files in REPOS.items():
            self.hub.add_repo(repo, rev, files)
        self.settings = Settings(data_dir=tmp_path / "data", hf_mirrors=[f"https://{MIRROR}"], **settings)
        self.store = ModelStore(self.settings.resolved_models_dir)
        catalog = merge_entries(parse_catalog(catalog_data(with_v2=with_v2), require_hashes=True))
        self.manager = ModelManager(
            self.settings, catalog, self.store, client_factory=self.hub.client, sleep=lambda _s: None
        )

    def wait(self, download_id: str, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.manager.get_download(download_id)
            if status.state not in (DownloadState.queued, DownloadState.running):
                return status
            time.sleep(0.01)
        raise AssertionError("download did not finish in time")

    def install(self, ref: Ref, request: DownloadRequest | None = None):
        return self.wait(self.manager.start_download(ref, request).id)


@pytest.fixture
def world(tmp_path: Path):
    w = World(tmp_path)
    yield w
    w.manager.shutdown()


# --- catalog ----------------------------------------------------------------------------------


def test_the_bundled_catalog_is_valid_and_complete() -> None:
    catalog = merge_entries(load_bundled())
    models = {e.id: e for e in catalog.models()}
    assert set(models) == {"chickenrice", "whisper-ja"}
    assert (models["chickenrice"].task, models["whisper-ja"].task) == ("translate", "transcribe")
    for entry in catalog.entries.values():
        for spec in entry.versions:
            assert len(spec.revision) == 40 and all(f.sha256 and f.size for f in spec.files)
    assert {str(r) for r in models["chickenrice"].requires} == {"vad-asr@1", "whisper-base-fe@1"}


def test_catalog_rejects_unsafe_paths_and_missing_hashes() -> None:
    bad = catalog_data()
    bad["entries"][0]["versions"][0]["files"][0]["path"] = "../../etc/passwd"
    with pytest.raises(CatalogError, match="unsafe"):
        parse_catalog(bad, require_hashes=True)

    missing = catalog_data()
    del missing["entries"][0]["versions"][0]["files"][0]["sha256"]
    with pytest.raises(CatalogError, match="required"):
        parse_catalog(missing, require_hashes=True)
    parse_catalog(missing, require_hashes=False)  # the user's own local catalog may omit them

    floating = catalog_data()
    floating["entries"][0]["versions"][0]["revision"] = "main"
    with pytest.raises(CatalogError, match="commit revision"):
        parse_catalog(floating, require_hashes=True)


def test_catalog_validates_structure() -> None:
    for mutate, message in [
        (lambda d: d["entries"][0].update(task="dance"), "task"),
        (lambda d: d["entries"][0].update(id="Bad ID"), "invalid entry id"),
        (lambda d: d["entries"][0]["versions"].append(d["entries"][0]["versions"][0]), "duplicate"),
        (lambda d: d["entries"][0]["versions"][0].update(repo="nope"), "invalid repo"),
        (lambda d: d.update(schema=2), "schema"),
    ]:
        data = catalog_data()
        mutate(data)
        with pytest.raises(CatalogError, match=message):
            parse_catalog(data, require_hashes=True)


def test_catalog_rejects_unknown_requirements() -> None:
    data = catalog_data()
    data["entries"][0]["requires"] = ["ghost@1"]
    with pytest.raises(CatalogError, match="ghost@1"):
        merge_entries(parse_catalog(data, require_hashes=True))


def test_merge_unions_versions_and_later_sources_win() -> None:
    base = parse_catalog(catalog_data(with_v2=False), require_hashes=True)
    overlay = parse_catalog(catalog_data(with_v2=True), require_hashes=True)
    merged = merge_entries(base, overlay)
    assert [v.version for v in merged.get("tiny").versions] == ["v1", "v2"]


def test_local_toml_can_add_a_custom_model(tmp_path: Path) -> None:
    text = """
schema = 1
[[entry]]
id = "my-finetune"
kind = "model"
display_name = "My fine-tune"
task = "transcribe"
source_languages = ["ja"]
output_languages = ["ja"]
[[entry.versions]]
version = "1"
repo = "me/my-finetune"
revision = "main"
files = [{ path = "model.bin" }, { path = "config.json" }]
"""
    entries = parse_local_toml(text)
    assert entries[0].id == "my-finetune" and entries[0].versions[0].files[0].sha256 is None
    with pytest.raises(CatalogError, match="invalid TOML"):
        parse_local_toml("[[entry")


def test_load_catalog_layers_cache_and_local_and_survives_bad_layers(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.data_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    catalog, warnings = load_catalog(settings, config_dir)
    assert warnings == [] and {e.id for e in catalog.models()} == {"chickenrice", "whisper-ja"}

    (settings.data_dir / "catalog.cache.json").write_text("{not json")
    (config_dir / "catalog.local.toml").write_text('schema = 1\n[[entry]]\nid = "x"\n')
    catalog, warnings = load_catalog(settings, config_dir)
    assert len(warnings) == 2 and "cached catalog" in warnings[0] and "catalog.local.toml" in warnings[1]
    assert {e.id for e in catalog.models()} == {"chickenrice", "whisper-ja"}  # the built-in catalog still works


# --- listing / install --------------------------------------------------------------------------


def test_nothing_is_installed_at_first(world: World) -> None:
    (info,) = world.manager.list_models()
    assert info.id == "tiny" and info.active_version is None and info.update_available is False
    assert [v.status for v in info.versions] == [VersionStatus.not_installed] * 2
    assert info.latest_version == "v2"
    assert [(r.id, r.installed) for r in info.requires] == [("vad", False), ("fe", False)]
    with pytest.raises(ModelNotInstalled) as error:
        world.manager.resolve("tiny")
    assert error.value.code == "MODEL_NOT_INSTALLED"


def test_download_installs_the_model_and_its_components(world: World) -> None:
    started = world.manager.start_download(Ref("tiny", "v1"))
    assert started.state in (DownloadState.queued, DownloadState.running) and started.bytes_total is not None

    done = world.wait(started.id)

    assert done.state is DownloadState.succeeded and done.error is None
    assert done.bytes_done == done.bytes_total
    info = world.manager.get_model("tiny")
    assert info.versions[0].status is VersionStatus.installed
    assert info.active_version == "v1"  # the first installed version becomes the active one
    assert all(r.installed for r in info.requires)

    resolved = world.manager.resolve("tiny")
    assert resolved.ref == Ref("tiny", "v1")
    assert (resolved.model_dir / "model.bin").read_bytes() == REPOS[("acme/tiny", "a" * 40)]["model.bin"]
    assert (resolved.components["vad"] / "model.onnx").is_file()
    assert (resolved.components["fe"] / "preprocessor_config.json").is_file()


def test_a_second_download_of_the_same_version_joins_the_running_one(world: World) -> None:
    gate = threading.Event()
    world.hub.on_request = lambda _r: gate.wait(5)
    first = world.manager.start_download(Ref("tiny", "v1"))
    second = world.manager.start_download(Ref("tiny", "v1"))
    assert second.id == first.id
    gate.set()
    assert world.wait(first.id).state is DownloadState.succeeded


def test_installing_another_version_keeps_the_active_one_until_told_otherwise(world: World) -> None:
    world.install(Ref("tiny", "v1"))
    assert world.manager.get_model("tiny").update_available is True  # v2 exists and is not installed

    world.install(Ref("tiny", "v2"))
    info = world.manager.get_model("tiny")
    assert info.active_version == "v1" and info.update_available is False
    assert world.manager.resolve("tiny").ref.version == "v1"
    assert world.manager.resolve("tiny@v2").ref.version == "v2"  # pinning a version per job

    info = world.manager.set_active("tiny", "v2")
    assert info.active_version == "v2" and world.manager.resolve("tiny").ref.version == "v2"


def test_set_active_validates(world: World) -> None:
    with pytest.raises(NotFound):
        world.manager.set_active("ghost", "v1")
    with pytest.raises(NotFound):
        world.manager.set_active("tiny", "v9")
    with pytest.raises(Conflict, match="not installed"):
        world.manager.set_active("tiny", "v1")


def test_resolving_an_unknown_model_or_version(world: World) -> None:
    world.install(Ref("tiny", "v1"))
    with pytest.raises(NotFound):
        world.manager.resolve("ghost")
    with pytest.raises(NotFound):
        world.manager.resolve("tiny@v9")
    with pytest.raises(ModelNotInstalled):
        world.manager.resolve("tiny@v2")


def test_components_cannot_be_downloaded_directly(world: World) -> None:
    with pytest.raises(NotFound, match="component"):
        world.manager.start_download(Ref("vad", "1"))
    with pytest.raises(NotFound):
        world.manager.start_download(Ref("tiny", "v9"))


# --- delete / verify / corruption ---------------------------------------------------------------------


def test_delete_rules(world: World) -> None:
    world.install(Ref("tiny", "v1"))
    world.install(Ref("tiny", "v2"))

    with pytest.raises(Conflict) as in_use:  # a component a model needs
        world.manager.delete(Ref("vad", "1"))
    assert in_use.value.code == "IN_USE"

    with pytest.raises(Conflict) as active:  # the active version while another one is installed
        world.manager.delete(Ref("tiny", "v1"))
    assert active.value.code == "ACTIVE_VERSION"

    world.manager.delete(Ref("tiny", "v2"))  # not active: fine
    assert world.manager.get_model("tiny").versions[1].status is VersionStatus.not_installed

    world.manager.delete(Ref("tiny", "v1"))  # the only one left: allowed, and nothing stays active
    assert world.manager.get_model("tiny").active_version is None
    world.manager.delete(Ref("vad", "1"))  # nothing needs it any more
    with pytest.raises(NotFound):
        world.manager.delete(Ref("tiny", "v9"))


def test_verify_detects_damage_and_resolve_refuses_a_broken_install(world: World) -> None:
    world.install(Ref("tiny", "v1"))
    assert world.manager.verify(Ref("tiny", "v1")).ok

    model = world.store.version_dir(Ref("tiny", "v1")) / "model.bin"
    data = bytearray(model.read_bytes())
    data[0] ^= 0xFF
    model.write_bytes(bytes(data))  # same size, different content: only hashing notices
    result = world.manager.verify(Ref("tiny", "v1"))
    assert not result.ok and "checksum mismatch" in result.problems[0]

    model.unlink()  # a missing file is noticed by the cheap check too
    assert world.manager.get_model("tiny").versions[0].status is VersionStatus.corrupt
    with pytest.raises(ModelNotInstalled) as error:
        world.manager.resolve("tiny")
    assert error.value.code == "MODEL_CORRUPT"
    with pytest.raises(Conflict):
        world.manager.verify(Ref("tiny", "v2"))

    # Downloading a corrupt version repairs it.
    assert world.install(Ref("tiny", "v1")).state is DownloadState.succeeded
    assert world.manager.verify(Ref("tiny", "v1")).ok


# --- cancel / resume ---------------------------------------------------------------------------------------


def test_cancel_then_resume_finishes_the_install(world: World) -> None:
    gate = threading.Event()
    seen = threading.Event()

    def hold(_request) -> None:
        seen.set()
        gate.wait(5)

    world.hub.on_request = hold
    started = world.manager.start_download(Ref("tiny", "v1"))
    assert seen.wait(5)
    world.manager.cancel_download(started.id)
    world.manager.cancel_download(started.id)  # idempotent
    gate.set()

    cancelled = world.wait(started.id)
    assert cancelled.state is DownloadState.cancelled
    assert world.manager.get_model("tiny").versions[0].status is VersionStatus.not_installed

    world.hub.on_request = None
    assert world.install(Ref("tiny", "v1")).state is DownloadState.succeeded


def test_unknown_download_id(world: World) -> None:
    with pytest.raises(NotFound):
        world.manager.get_download("nope")
    with pytest.raises(NotFound):
        world.manager.cancel_download("nope")


def test_deleting_a_version_that_is_downloading_is_refused(world: World) -> None:
    gate = threading.Event()
    world.hub.on_request = lambda _r: gate.wait(5)
    started = world.manager.start_download(Ref("tiny", "v1"))
    with pytest.raises(Conflict) as error:
        world.manager.delete(Ref("tiny", "v1"))
    assert error.value.code == "DOWNLOAD_IN_PROGRESS"
    assert world.manager.get_model("tiny").versions[0].status is VersionStatus.downloading
    gate.set()
    world.wait(started.id)


# --- mirrors -------------------------------------------------------------------------------------------------


def test_a_chosen_endpoint_is_used_first_and_reported(world: World) -> None:
    done = world.install(Ref("tiny", "v1"), DownloadRequest(endpoint="https://custom.example/"))
    assert done.state is DownloadState.succeeded
    assert world.hub.requests[0][0] == "custom.example"
    assert done.source == "https://custom.example"


def test_a_chosen_endpoint_falls_back_to_the_configured_ones(world: World) -> None:
    world.hub.down_hosts = {"custom.example"}
    done = world.install(Ref("tiny", "v1"), DownloadRequest(endpoint="https://custom.example"))
    assert done.state is DownloadState.succeeded and done.source == f"https://{HF}"


def test_fallback_can_be_switched_off(world: World) -> None:
    world.hub.down_hosts = {"custom.example"}
    done = world.install(Ref("tiny", "v1"), DownloadRequest(endpoint="https://custom.example", fallback=False))
    assert done.state is DownloadState.failed
    assert "custom.example" not in (done.error or "") or "HTTP 503" in (done.error or "")
    assert world.hub.hosts_used() == {"custom.example"}  # never touched Hugging Face or the mirror


def test_the_configured_endpoint_can_be_a_mirror(tmp_path: Path) -> None:
    w = World(tmp_path, hf_endpoint=f"https://{MIRROR}")
    try:
        w.install(Ref("tiny", "v1"))
        assert w.hub.requests[0][0] == MIRROR
    finally:
        w.manager.shutdown()


def test_download_failure_is_reported_not_raised(world: World) -> None:
    world.hub.down_hosts = {HF, MIRROR}
    done = world.install(Ref("tiny", "v1"))
    assert done.state is DownloadState.failed and "HTTP 503" in done.error and "hf_endpoint" in done.error


def test_sources_are_cached_and_can_be_refreshed(world: World) -> None:
    first = world.manager.sources()
    calls = len(world.hub.requests)
    assert world.manager.sources() is first and len(world.hub.requests) == calls  # served from the cache

    world.manager.sources(refresh=True)
    assert len(world.hub.requests) > calls

    extra = world.manager.sources(extra=["https://custom.example"])
    assert "https://custom.example" in {s.endpoint for s in extra.sources}
    assert world.manager.sources() is not extra  # an ad-hoc probe does not replace the cached result


def test_the_probe_reports_a_blocked_huggingface(world: World) -> None:
    world.hub.down_hosts = {HF}
    result = world.manager.sources(refresh=True)
    assert result.huggingface_reachable is False and result.recommended_endpoint == f"https://{MIRROR}"


# --- catalog refresh ------------------------------------------------------------------------------------------


def refreshing_world(tmp_path: Path, payload: dict | Exception | None):
    w = World(tmp_path, with_v2=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "catalog.example":
            if isinstance(payload, Exception):
                raise payload
            return httpx.Response(200, content=json.dumps(payload).encode())
        return w.hub._handle(request)

    w.manager._client_factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))
    w.settings.catalog_url = "https://catalog.example/catalog.json"
    return w


def test_refresh_brings_in_a_new_model_version(tmp_path: Path) -> None:
    w = refreshing_world(tmp_path, catalog_data(with_v2=True))
    try:
        assert [v.version for v in w.manager.get_model("tiny").versions] == ["v1"]
        result = w.manager.refresh_catalog()
        assert result.refreshed and result.models == 1
        assert [v.version for v in w.manager.get_model("tiny").versions] == ["v1", "v2"]
        assert (w.settings.data_dir / "catalog.cache.json").is_file()  # survives a restart
    finally:
        w.manager.shutdown()


def test_refresh_failures_keep_the_current_catalog(tmp_path: Path) -> None:
    unhashed = catalog_data()
    del unhashed["entries"][0]["versions"][0]["files"][0]["sha256"]
    for payload in (httpx.ConnectError("no route"), unhashed, {"schema": 9, "entries": []}):
        w = refreshing_world(tmp_path / str(id(payload)), payload)
        try:
            result = w.manager.refresh_catalog()
            assert not result.refreshed and "kept the current catalog" in result.message
            assert [v.version for v in w.manager.get_model("tiny").versions] == ["v1"]
        finally:
            w.manager.shutdown()


def test_refresh_requires_https(tmp_path: Path) -> None:
    w = refreshing_world(tmp_path, catalog_data())
    try:
        w.settings.catalog_url = "http://catalog.example/catalog.json"
        result = w.manager.refresh_catalog()
        assert not result.refreshed and "https" in result.message
    finally:
        w.manager.shutdown()


# --- store ------------------------------------------------------------------------------------------------------


def test_state_file_survives_a_corrupt_write(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    assert store.active("tiny") is None
    store.set_active("tiny", "v1")
    assert ModelStore(tmp_path).active("tiny") == "v1"
    (tmp_path / "state.json").write_text("{broken")
    assert store.active("tiny") is None  # a damaged pointer falls back to "newest installed" in the manager
    store.set_active("tiny", "v2")
    assert store.active("tiny") == "v2"
