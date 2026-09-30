"""The downloader against a fake hub: parallel blocks, resume, mirrors, cancel, verification."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from fakehub import FakeHub, make_version
from hibiki_asr.models import download as dl
from hibiki_asr.models.download import (
    ChecksumMismatch,
    DownloadCancelled,
    Downloader,
    DownloadError,
    DownloadProgress,
    InsufficientDiskSpace,
)

HF = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"
MIB = 1024 * 1024


def payload(size: int, seed: int = 7) -> bytes:
    """Deterministic, non-repeating bytes so a misplaced block changes the checksum."""
    out = bytearray()
    x = seed
    while len(out) < size:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out += x.to_bytes(4, "little")
    return bytes(out[:size])


def downloader(hub: FakeHub, endpoints=(HF,), **kwargs) -> Downloader:
    kwargs.setdefault("sleep", lambda _s: None)
    return Downloader(hub.client(), endpoints, **kwargs)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "dest", tmp_path / "partial"


def test_small_files_are_downloaded_and_verified(dirs) -> None:
    dest, partial = dirs
    files = {"config.json": b'{"a": 1}', "sub/vocab.txt": payload(1000)}
    hub = FakeHub(files)

    downloader(hub).fetch(make_version(files), dest, partial)

    assert (dest / "config.json").read_bytes() == files["config.json"]
    assert (dest / "sub" / "vocab.txt").read_bytes() == files["sub/vocab.txt"]
    assert not partial.exists()  # no half-downloaded files and no empty scaffolding left behind


def test_large_file_is_fetched_in_parallel_blocks(dirs) -> None:
    dest, partial = dirs
    data = payload(5 * MIB + 123)
    hub = FakeHub({"model.bin": data})
    seen: list[DownloadProgress] = []

    downloader(hub, threads=4, block_size=MIB, parallel_min_size=2 * MIB).fetch(
        make_version({"model.bin": data}), dest, partial, on_progress=seen.append
    )

    assert (dest / "model.bin").read_bytes() == data
    ranges = [r for _, _, r in hub.requests if r]
    assert len(ranges) == 6  # 5 full blocks + the short tail
    assert len(set(ranges)) == 6  # every block exactly once
    # progress never goes backwards and ends at the full size
    done = [p.bytes_done for p in seen if p.stage == "downloading"]
    assert done == sorted(done) and done[-1] == len(data)
    assert seen[-1].stage == "verifying" and seen[-1].bytes_done == len(data)
    # nothing is known before the first response; from then on the source is reported
    assert {p.endpoint for p in seen} <= {None, HF} and seen[-1].endpoint == HF


def test_blocks_really_download_concurrently(dirs) -> None:
    dest, partial = dirs
    data = payload(4 * MIB)
    hub = FakeHub({"model.bin": data})
    barrier = threading.Barrier(3, timeout=5)  # the first three requests must be in flight at the same time
    arrivals = iter(range(3))
    lock = threading.Lock()

    def rendezvous(_request) -> None:
        with lock:
            first_wave = next(arrivals, None) is not None
        if first_wave:
            barrier.wait()  # the fourth block is fetched by whichever thread frees up first

    hub.on_request = rendezvous
    downloader(hub, threads=3, block_size=MIB, parallel_min_size=MIB).fetch(
        make_version({"model.bin": data}), dest, partial
    )
    assert (dest / "model.bin").read_bytes() == data


def test_interrupted_block_download_resumes_only_the_missing_blocks(dirs) -> None:
    dest, partial = dirs
    block = MIB
    data = payload(4 * block)
    version = make_version({"model.bin": data})

    # A previous run finished blocks 0 and 2 and was killed.
    part_dir = partial / f"{version.repo.replace('/', '__')}@{version.revision}"
    part_dir.mkdir(parents=True)
    blocks = part_dir / "model.bin.blocks"
    buffer = bytearray(len(data))
    for index in (0, 2):
        buffer[index * block : (index + 1) * block] = data[index * block : (index + 1) * block]
    blocks.write_bytes(bytes(buffer))
    (part_dir / "model.bin.blocks.done").write_text("0\n2\n")

    hub = FakeHub({"model.bin": data})
    downloader(hub, threads=2, block_size=block, parallel_min_size=block).fetch(version, dest, partial)

    assert (dest / "model.bin").read_bytes() == data
    fetched = sorted(r for _, _, r in hub.requests)
    assert fetched == [f"bytes={block}-{2 * block - 1}", f"bytes={3 * block}-{4 * block - 1}"]


def test_single_stream_resumes_from_a_partial_file(dirs) -> None:
    dest, partial = dirs
    data = payload(300_000)
    version = make_version({"vocab.bin": data})
    part_dir = partial / f"{version.repo.replace('/', '__')}@{version.revision}"
    part_dir.mkdir(parents=True)
    (part_dir / "vocab.bin.part").write_bytes(data[:100_000])

    hub = FakeHub({"vocab.bin": data})
    downloader(hub).fetch(version, dest, partial)

    assert (dest / "vocab.bin").read_bytes() == data
    assert [r for _, _, r in hub.requests] == ["bytes=100000-"]


def test_falls_back_to_a_mirror_when_the_first_endpoint_is_down(dirs) -> None:
    dest, partial = dirs
    files = {"config.json": b"{}"}
    hub = FakeHub(files)
    hub.down_hosts = {"huggingface.co"}
    seen: list[DownloadProgress] = []

    downloader(hub, (HF, MIRROR)).fetch(make_version(files), dest, partial, on_progress=seen.append)

    assert (dest / "config.json").read_bytes() == b"{}"
    assert hub.hosts_used() == {"huggingface.co", "hf-mirror.com"}
    assert seen[-1].endpoint == MIRROR


def test_a_missing_file_on_the_first_endpoint_moves_on_without_retrying_it(dirs) -> None:
    dest, partial = dirs
    hub = FakeHub({"config.json": b"{}"})
    hub.missing_hosts = {"huggingface.co"}

    downloader(hub, (HF, MIRROR)).fetch(make_version({"config.json": b"{}"}), dest, partial)

    assert [h for h, _, _ in hub.requests].count("huggingface.co") == 1  # 404 is not retried


def test_transient_failures_are_retried_on_the_same_endpoint(dirs) -> None:
    dest, partial = dirs
    hub = FakeHub({"config.json": b"{}"})
    hub.fail_first["huggingface.co"] = 2
    sleeps: list[float] = []

    downloader(hub, sleep=sleeps.append).fetch(make_version({"config.json": b"{}"}), dest, partial)

    assert (dest / "config.json").read_bytes() == b"{}"
    assert sleeps == [1, 2]  # exponential backoff


def test_all_endpoints_failing_gives_an_actionable_error(dirs) -> None:
    dest, partial = dirs
    hub = FakeHub({"config.json": b"{}"})
    hub.down_hosts = {"huggingface.co", "hf-mirror.com"}

    with pytest.raises(DownloadError) as info:
        downloader(hub, (HF, MIRROR)).fetch(make_version({"config.json": b"{}"}), dest, partial)

    message = str(info.value)
    assert "config.json" in message and "HTTP 503" in message and "hf_endpoint" in message


def test_corrupted_download_is_rejected_and_discarded(dirs) -> None:
    dest, partial = dirs
    files = {"model.bin": payload(5000)}
    hub = FakeHub(files)
    hub.corrupt = {"model.bin"}

    with pytest.raises(ChecksumMismatch, match="sha256"):
        downloader(hub).fetch(make_version(files), dest, partial)

    assert not (dest / "model.bin").exists()
    assert not list(partial.rglob("*.part"))  # the bad bytes are not kept for the next resume


def test_server_without_range_support_falls_back_to_a_single_stream(dirs) -> None:
    dest, partial = dirs
    data = payload(3 * MIB)
    hub = FakeHub({"model.bin": data})
    hub.no_range_hosts = {"huggingface.co"}

    downloader(hub, threads=4, block_size=MIB, parallel_min_size=MIB).fetch(
        make_version({"model.bin": data}), dest, partial
    )

    assert (dest / "model.bin").read_bytes() == data
    assert not list(partial.rglob("*.blocks*"))


def test_cancel_stops_the_download_and_keeps_progress_for_resuming(dirs) -> None:
    dest, partial = dirs
    data = payload(6 * MIB)
    hub = FakeHub({"model.bin": data})
    version = make_version({"model.bin": data})
    state = {"chunks": 0}

    def cancelled() -> bool:
        return state["chunks"] >= 2

    def progress(_p: DownloadProgress) -> None:
        state["chunks"] += 1

    with pytest.raises(DownloadCancelled):
        downloader(hub, threads=2, block_size=MIB, parallel_min_size=MIB).fetch(
            version, dest, partial, cancelled=cancelled, on_progress=progress
        )
    assert not (dest / "model.bin").exists()

    # Resuming finishes the job and does not refetch everything.
    hub.requests.clear()
    downloader(hub, threads=2, block_size=MIB, parallel_min_size=MIB).fetch(version, dest, partial)
    assert (dest / "model.bin").read_bytes() == data
    assert len(hub.requests) < 6


def test_files_already_installed_are_skipped(dirs) -> None:
    dest, partial = dirs
    files = {"a.json": b"aaa", "b.json": b"bbb"}
    dest.mkdir()
    (dest / "a.json").write_bytes(b"aaa")
    hub = FakeHub(files)

    downloader(hub).fetch(make_version(files), dest, partial)

    assert [f for _, f, _ in hub.requests] == ["b.json"]


def test_not_enough_disk_space_is_reported_before_downloading(dirs, monkeypatch) -> None:
    dest, partial = dirs
    hub = FakeHub({"model.bin": payload(1000)})
    monkeypatch.setattr(
        dl.shutil, "disk_usage", lambda _p: os.stat_result((0,) * 10) and type("U", (), {"free": 1000})()
    )

    with pytest.raises(InsufficientDiskSpace, match="models_dir"):
        downloader(hub).fetch(make_version({"model.bin": payload(1000)}), dest, partial)
    assert hub.requests == []


def test_token_is_only_sent_to_the_first_endpoint(dirs) -> None:
    dest, partial = dirs
    hub = FakeHub({"config.json": b"{}"})
    auth: list[tuple[str, str | None]] = []
    hub.on_request = lambda r: auth.append((r.url.host, r.headers.get("Authorization")))
    hub.down_hosts = {"huggingface.co"}

    downloader(hub, (HF, MIRROR), token="secret").fetch(make_version({"config.json": b"{}"}), dest, partial)

    by_host = dict(auth)
    assert by_host["huggingface.co"] == "Bearer secret"
    assert by_host["hf-mirror.com"] is None  # never leak a Hugging Face token to a third-party mirror


def test_constructor_validation() -> None:
    hub = FakeHub({})
    with pytest.raises(ValueError):
        Downloader(hub.client(), [])
    with pytest.raises(ValueError):
        Downloader(hub.client(), [HF], threads=0)


def test_a_dead_endpoint_is_not_retried_for_every_file(dirs) -> None:
    dest, partial = dirs
    files = {f"f{i}.json": payload(200, seed=i) for i in range(4)}
    hub = FakeHub(files)
    hub.down_hosts = {"huggingface.co"}

    downloader(hub, (HF, MIRROR)).fetch(make_version(files), dest, partial)

    assert all((dest / name).read_bytes() == data for name, data in files.items())
    # Three attempts on the first file, then the mirror is tried first for the remaining three files.
    assert [h for h, _, _ in hub.requests].count("huggingface.co") == dl.ATTEMPTS_PER_ENDPOINT


def test_an_unreachable_host_moves_on_immediately(dirs) -> None:
    dest, partial = dirs
    hub = FakeHub({"config.json": b"{}"})
    hub.unreachable_hosts = {"huggingface.co"}
    sleeps: list[float] = []

    downloader(hub, (HF, MIRROR), sleep=sleeps.append).fetch(
        make_version({"config.json": b"{}"}), dest, partial
    )

    assert (dest / "config.json").read_bytes() == b"{}"
    assert sleeps == []  # no backoff for a host we cannot even connect to
    assert hub.hosts_used() == {"huggingface.co", "hf-mirror.com"}
