"""The HTTP API end to end: auth, errors, models, downloads with a chosen mirror, jobs, cancel, diagnostics."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from helpers import RTX4090, hw, rt
from hibiki_asr import API_VERSION
from hibiki_asr.api.app import CAPABILITIES, create_app
from hibiki_asr.engine import Engine
from hibiki_asr.jobs.manager import JobManager
from hibiki_asr.worker.protocol import Progress, RunFailed
from test_jobs import FakeWorker, cooperative, ok, result, stubborn
from test_models import World

TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


class Api:
    def __init__(
        self,
        tmp_path: Path,
        behaviors=(ok,),
        *,
        token: str | None = TOKEN,
        hardware=None,
        runtime=None,
        **settings,
    ):
        self.world = World(tmp_path, token=token, **settings)
        self.workers: list[FakeWorker] = []
        it = iter(behaviors)
        last = behaviors[-1]

        def factory() -> FakeWorker:
            worker = FakeWorker(next(it, last))
            self.workers.append(worker)
            return worker

        self.engine = Engine(
            self.world.settings,
            models=self.world.manager,
            jobs=JobManager(self.world.settings, factory),
            hardware_probe=lambda: hardware or hw(),
            runtime_probe=lambda: runtime or rt(0),
        )
        self.client = TestClient(create_app(self.engine))
        self.headers = AUTH if token else {}

    def get(self, path, **kw):
        return self.client.get(path, headers=self.headers, **kw)

    def post(self, path, **kw):
        return self.client.post(path, headers=self.headers, **kw)

    def delete(self, path, **kw):
        return self.client.delete(path, headers=self.headers, **kw)

    def put(self, path, **kw):
        return self.client.put(path, headers=self.headers, **kw)

    def install(self, version: str = "v1", **body):
        started = self.post(f"/v1/models/tiny/versions/{version}/download", json=body or None).json()
        self.world.wait(started["id"])

    def submit(self, params: dict | None = None, audio: bytes = WAV, name: str = "a.wav"):
        params = params if params is not None else {"model_id": "tiny"}
        return self.post(
            "/v1/jobs", data={"params": json.dumps(params)}, files={"audio": (name, audio, "audio/wav")}
        )

    def wait_job(self, job_id: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.get(f"/v1/jobs/{job_id}").json()
            if body["state"] in ("succeeded", "failed", "cancelled"):
                return body
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    def close(self) -> None:
        self.engine.shutdown()


@pytest.fixture
def api(tmp_path: Path):
    made: list[Api] = []

    def build(*behaviors, **kw) -> Api:
        a = Api(tmp_path / f"api{len(made)}", behaviors or (ok,), **kw)
        made.append(a)
        return a

    yield build
    for a in made:
        a.close()


# --- auth / meta ------------------------------------------------------------------------------------------


def test_healthz_needs_no_token(api) -> None:
    a = api()
    response = a.client.get("/v1/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok", "api_version": API_VERSION}


@pytest.mark.parametrize(
    "headers,status,code",
    [
        ({}, 401, "UNAUTHORIZED"),
        ({"Authorization": "Basic abc"}, 401, "UNAUTHORIZED"),
        ({"Authorization": "Bearer nope"}, 403, "FORBIDDEN"),
    ],
)
def test_every_other_route_requires_the_token(api, headers, status, code) -> None:
    a = api()
    for path in ("/v1/version", "/v1/models", "/v1/diagnostics", "/v1/jobs/x"):
        response = a.client.get(path, headers=headers)
        assert response.status_code == status and response.json()["error"]["code"] == code
    assert a.client.get("/v1/version", headers={}).headers.get("www-authenticate") == "Bearer"


def test_without_a_configured_token_the_engine_is_open(api) -> None:
    a = api(token=None)
    assert a.client.get("/v1/version").status_code == 200


def test_version_lists_capabilities(api) -> None:
    body = api().get("/v1/version").json()
    assert body["api_version"] == API_VERSION and body["capabilities"] == CAPABILITIES
    assert {"job-cancel", "model-download-sources"} <= set(body["capabilities"])


def test_validation_errors_use_the_common_error_shape(api) -> None:
    response = api().put("/v1/models/tiny/active", json={})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_REQUEST"


# --- diagnostics --------------------------------------------------------------------------------------------


def test_diagnostics_on_a_cpu_machine(api) -> None:
    body = api().get("/v1/diagnostics").json()
    assert body["selection"]["device"] == "cpu" and body["selection"]["degraded"] is False
    assert [f["code"] for f in body["findings"]] == ["CPU_ONLY"]
    assert (
        "token" not in body["settings"] and "hf_token" not in body["settings"]
    )  # secrets never leave the engine
    assert body["settings"]["device"] == "auto" and body["settings"]["models_dir"]


def test_diagnostics_explain_a_cpu_fallback(api) -> None:
    a = api(hardware=hw(RTX4090, container=True), runtime=rt(0))
    body = a.get("/v1/diagnostics").json()
    assert body["selection"]["degraded"] is True
    finding = next(f for f in body["findings"] if f["code"] == "CT2_NO_GPU_SUPPORT")
    assert "gpus: all" in finding["hint"]
    assert body["findings"][-1]["code"] == "DEGRADED_TO_CPU"


def test_diagnostics_are_cached_until_refreshed(api) -> None:
    calls = {"n": 0}
    a = api()

    def counting() -> object:
        calls["n"] += 1
        return hw()

    a.engine._hardware_probe = counting  # type: ignore[assignment]
    a.get("/v1/diagnostics")
    a.get("/v1/diagnostics")
    assert calls["n"] == 1
    a.get("/v1/diagnostics", params={"refresh": "true"})
    assert calls["n"] == 2


def test_a_broken_local_catalog_is_surfaced_as_a_finding(api) -> None:
    a = api()
    a.world.manager.catalog_warnings.append("ignoring /x/catalog.local.toml: boom")
    finding = next(
        f for f in a.get("/v1/diagnostics").json()["findings"] if f["code"] == "CATALOG_SOURCE_IGNORED"
    )
    assert "boom" in finding["message"]


# --- models -------------------------------------------------------------------------------------------------


def test_models_listing_download_and_versions(api) -> None:
    a = api()
    (model,) = a.get("/v1/models").json()
    assert model["id"] == "tiny" and model["active_version"] is None and model["latest_version"] == "v2"
    assert a.get("/v1/models/tiny").json()["id"] == "tiny"

    started = a.post("/v1/models/tiny/versions/v1/download")
    assert started.status_code == 202 and started.json()["state"] in ("queued", "running", "succeeded")
    download_id = started.json()["id"]
    a.world.wait(download_id)
    assert a.get(f"/v1/models/downloads/{download_id}").json()["state"] == "succeeded"
    assert [d["id"] for d in a.get("/v1/models/downloads").json()] == [download_id]

    a.install("v2")
    assert a.get("/v1/models/tiny").json()["active_version"] == "v1"
    switched = a.put("/v1/models/tiny/active", json={"version": "v2"})
    assert switched.status_code == 200 and switched.json()["active_version"] == "v2"

    assert a.post("/v1/models/tiny/versions/v1/verify").json() == {
        "ok": True,
        "checked_files": 2,
        "problems": [],
    }
    assert (
        a.delete("/v1/models/tiny/versions/v2").json()["error"]["code"] == "ACTIVE_VERSION"
    )  # active, and v1 exists
    assert a.delete("/v1/models/tiny/versions/v1").status_code == 204


def test_model_errors_carry_codes_and_statuses(api) -> None:
    a = api()
    assert a.get("/v1/models/ghost").status_code == 404
    assert a.get("/v1/models/ghost").json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert a.get("/v1/models/downloads/nope").json()["error"]["code"] == "DOWNLOAD_NOT_FOUND"
    assert a.put("/v1/models/tiny/active", json={"version": "v1"}).status_code == 409  # not installed
    assert a.delete("/v1/models/tiny/versions/v9").status_code == 404


def test_the_download_source_can_be_chosen_per_request(api) -> None:
    a = api()
    started = a.post(
        "/v1/models/tiny/versions/v1/download",
        json={"endpoint": "https://hf-mirror.com", "threads": 2, "fallback": False},
    )
    assert started.status_code == 202
    done = a.world.wait(started.json()["id"])
    assert done.source == "https://hf-mirror.com" and a.world.hub.hosts_used() == {"hf-mirror.com"}


def test_a_bad_download_endpoint_is_rejected(api) -> None:
    a = api()
    response = a.post("/v1/models/tiny/versions/v1/download", json={"endpoint": "ftp://nope"})
    assert response.status_code == 422 and "http(s) URL" in response.json()["error"]["message"]


def test_a_download_can_be_cancelled_over_http(api) -> None:
    a = api()
    gate, seen = threading.Event(), threading.Event()
    a.world.hub.on_request = lambda _r: (seen.set(), gate.wait(5))
    download_id = a.post("/v1/models/tiny/versions/v1/download").json()["id"]
    assert seen.wait(5)
    cancel = a.delete(f"/v1/models/downloads/{download_id}")
    assert cancel.status_code == 202
    gate.set()
    assert a.world.wait(download_id).state.value == "cancelled"


def test_sources_tell_the_client_whether_huggingface_is_reachable(api) -> None:
    a = api()
    body = a.get("/v1/models/sources").json()
    assert body["huggingface_reachable"] is True and body["default_endpoint"] == "https://huggingface.co"
    assert {s["endpoint"] for s in body["sources"]} == {"https://huggingface.co", "https://hf-mirror.com"}

    a.world.hub.down_hosts = {"huggingface.co"}
    blocked = a.get("/v1/models/sources", params={"refresh": "true"}).json()
    assert (
        blocked["huggingface_reachable"] is False
        and blocked["recommended_endpoint"] == "https://hf-mirror.com"
    )

    custom = a.post("/v1/models/sources/probe", json={"endpoints": ["https://my-mirror.example/"]}).json()
    entry = next(s for s in custom["sources"] if s["kind"] == "custom")
    assert entry["endpoint"] == "https://my-mirror.example" and entry["reachable"] is True
    assert a.post("/v1/models/sources/probe", json={"endpoints": ["nonsense"]}).status_code == 422


def test_catalog_refresh_reports_failure_without_an_error_status(api) -> None:
    a = api()
    a.world.settings.catalog_url = "http://insecure.example/catalog.json"
    response = a.post("/v1/models/refresh")
    assert response.status_code == 200 and response.json()["refreshed"] is False


# --- jobs ---------------------------------------------------------------------------------------------------------


def test_a_job_needs_an_installed_model(api) -> None:
    a = api()
    response = a.submit()
    assert response.status_code == 409 and response.json()["error"]["code"] == "MODEL_NOT_INSTALLED"
    assert a.submit({"model_id": "ghost"}).status_code == 404
    assert not (a.world.settings.data_dir / "jobs").exists()  # nothing was stored for a refused job


def test_submit_poll_and_read_the_result(api) -> None:
    a = api()
    a.install()
    response = a.submit({"model_id": "tiny"})
    assert response.status_code == 202
    job = response.json()
    assert job["state"] in ("queued", "running") and job["id"]

    done = a.wait_job(job["id"])
    assert done["state"] == "succeeded" and done["progress"] == 1.0
    assert done["result"]["segments"][0] == {"start_ms": 0, "end_ms": 1500, "text": "こんにちは"}
    assert done["result"]["vtt"].startswith("WEBVTT")
    assert done["result"]["model"] == {"id": "tiny", "version": "v1"}
    assert done["result"]["runtime"]["degraded"] is False

    request = a.workers[0].requests[0]
    assert (request.task, request.language) == ("translate", "ja")  # defaults come from the model
    assert request.model_ref == "tiny@v1"
    assert not Path(request.audio_path).exists()  # the upload is removed once the job is done


def test_a_specific_model_version_can_be_pinned_per_job(api) -> None:
    a = api()
    a.install("v1")
    a.install("v2")
    job = a.submit({"model_id": "tiny@v2"}).json()
    a.wait_job(job["id"])
    assert a.workers[0].requests[0].model_ref == "tiny@v2"


def test_task_and_language_are_validated_against_the_model(api) -> None:
    a = api()
    a.install()
    task = a.submit({"model_id": "tiny", "task": "transcribe"})
    assert task.status_code == 422 and task.json()["error"]["code"] == "TASK_NOT_SUPPORTED"
    language = a.submit({"model_id": "tiny", "language": "en"})
    assert language.status_code == 422 and language.json()["error"]["code"] == "LANGUAGE_NOT_SUPPORTED"
    assert a.submit({"model_id": "tiny", "task": "translate", "language": "ja"}).status_code == 202


def test_bad_job_requests(api) -> None:
    a = api()
    a.install()
    assert (
        a.post("/v1/jobs", data={"params": "{not json"}, files={"audio": ("a.wav", WAV)}).json()["error"][
            "code"
        ]
        == "INVALID_REQUEST"
    )
    assert (
        a.post("/v1/jobs", data={"params": "{}"}, files={"audio": ("a.wav", WAV)}).json()["error"]["code"]
        == "INVALID_REQUEST"
    )
    assert a.submit(audio=b"").json()["error"]["code"] == "EMPTY_AUDIO"
    assert a.post("/v1/jobs", data={"params": '{"model_id": "tiny"}'}).status_code == 422  # no audio part


def test_an_oversized_upload_is_refused_and_not_kept(api) -> None:
    a = api(max_upload_mb=1)
    a.install()
    response = a.submit(audio=b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413 and response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    jobs_dir = a.world.settings.data_dir / "jobs"
    assert not jobs_dir.exists() or not any(jobs_dir.iterdir())


def test_upload_names_never_choose_where_the_file_goes(api) -> None:
    a = api()
    a.install()
    jobs_dir = (a.world.settings.data_dir / "jobs").resolve()

    hostile = a.submit(name="../../evil.sh").json()
    odd = a.submit(name="track.a$b").json()
    for job in (hostile, odd):
        a.wait_job(job["id"])

    paths = [Path(r.audio_path) for r in a.workers[0].requests]
    assert all(
        p.resolve().is_relative_to(jobs_dir) and p.parent.name in (hostile["id"], odd["id"]) for p in paths
    )
    assert [p.name for p in paths] == [
        "audio.sh",
        "audio",
    ]  # a plain short extension is kept; anything odd is dropped


def test_cancel_over_http(api) -> None:
    a = api(cooperative)
    a.install()
    job = a.submit().json()
    deadline = time.monotonic() + 5
    while a.get(f"/v1/jobs/{job['id']}").json()["progress"] == 0 and time.monotonic() < deadline:
        time.sleep(0.02)

    cancel = a.delete(f"/v1/jobs/{job['id']}")
    assert cancel.status_code == 202 and cancel.json()["state"] == "cancelling"
    assert a.wait_job(job["id"])["state"] == "cancelled"
    assert a.delete(f"/v1/jobs/{job['id']}").json()["state"] == "cancelled"  # idempotent


def test_a_stubborn_worker_is_killed_after_the_grace_period_over_http(api) -> None:
    a = api(stubborn, ok, cancel_grace_seconds=0.3)
    a.install()
    job = a.submit().json()
    deadline = time.monotonic() + 5
    while a.get(f"/v1/jobs/{job['id']}").json()["progress"] == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    a.delete(f"/v1/jobs/{job['id']}")
    assert a.wait_job(job["id"])["state"] == "cancelled" and a.workers[0].killed


def test_failures_and_fallbacks_are_reported_with_their_hints(api) -> None:
    a = api(lambda w, r: w.emit(RunFailed("GPU_LOAD_FAILED", "no memory", "close other GPU programs")))
    a.install()
    failed = a.wait_job(a.submit().json()["id"])
    assert failed["state"] == "failed" and failed["result"] is None
    assert failed["error"] == {
        "code": "GPU_LOAD_FAILED",
        "message": "no memory",
        "hint": "close other GPU programs",
        "findings": [],
    }

    b = api(lambda w, r: (w.emit(Progress("loading", 0.0, "x")), w.emit(result(r, degraded=True))))
    b.install()
    done = b.wait_job(b.submit().json()["id"])
    assert done["result"]["runtime"]["degraded"] is True and done["result"]["runtime"]["device"] == "cpu"


def test_unknown_jobs_are_404(api) -> None:
    a = api()
    assert a.get("/v1/jobs/nope").json()["error"]["code"] == "JOB_NOT_FOUND"
    assert a.delete("/v1/jobs/nope").status_code == 404
