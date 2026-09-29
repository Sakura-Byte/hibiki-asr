"""Source probing: can this machine reach Hugging Face directly, and which mirror is best?"""

from __future__ import annotations

import time

import pytest
from fakehub import REPO, REVISION, FakeHub

from hibiki_asr.models.catalog import merge_entries, parse_catalog
from hibiki_asr.models.schema import ProbeSourcesRequest, SourceKind
from hibiki_asr.models.sources import OFFICIAL_ENDPOINT, probe_sources, smallest_pinned_file

import hashlib

SMALL = b'{"tiny": true}'
BIG = b"x" * 5000


def catalog_for_hub():
    def file(path, data):
        return {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    data = {
        "schema": 1,
        "entries": [
            {
                "id": "comp",
                "kind": "component",
                "versions": [{"version": "1", "repo": REPO, "revision": REVISION, "files": [file("big.bin", BIG), file("meta.json", SMALL)]}],
            }
        ],
    }
    return merge_entries(parse_catalog(data, require_hashes=True))


def hub() -> FakeHub:
    return FakeHub({"big.bin": BIG, "meta.json": SMALL})


def probe(h: FakeHub, **kwargs):
    return probe_sources(h.client(), catalog_for_hub(), configured="https://huggingface.co", mirrors=["https://hf-mirror.com"], **kwargs)


def test_probes_with_the_smallest_pinned_file() -> None:
    version, spec = smallest_pinned_file(catalog_for_hub())
    assert spec.path == "meta.json" and version.repo == REPO


def test_everything_reachable_reports_all_sources() -> None:
    h = hub()
    result = probe(h)

    assert result.huggingface_reachable is True
    assert result.default_endpoint == "https://huggingface.co"
    by_endpoint = {s.endpoint: s for s in result.sources}
    assert set(by_endpoint) == {"https://huggingface.co", "https://hf-mirror.com"}
    assert by_endpoint["https://huggingface.co"].kind is SourceKind.official  # the official site keeps its own kind
    assert by_endpoint["https://hf-mirror.com"].kind is SourceKind.mirror
    assert all(s.reachable and s.latency_ms is not None and s.error is None for s in result.sources)
    assert result.recommended_endpoint in by_endpoint
    assert {f for _, f, _ in h.requests} == {"meta.json"}  # only the tiny file is fetched


def test_blocked_huggingface_recommends_the_mirror() -> None:
    h = hub()
    h.down_hosts = {"huggingface.co"}
    result = probe(h)

    assert result.huggingface_reachable is False
    assert result.recommended_endpoint == "https://hf-mirror.com"
    official = next(s for s in result.sources if s.kind is SourceKind.official)
    assert official.reachable is False and official.error == "HTTP 503"


def test_nothing_reachable_has_no_recommendation() -> None:
    h = hub()
    h.down_hosts = {"huggingface.co", "hf-mirror.com"}
    result = probe(h)
    assert result.recommended_endpoint is None and not any(s.reachable for s in result.sources)


def test_a_source_that_returns_the_wrong_content_is_not_reachable() -> None:
    h = hub()
    h.corrupt = {"meta.json"}  # e.g. a captive portal answering 200 with its own page
    result = probe(h)
    assert not any(s.reachable for s in result.sources)
    assert "different content" in (result.sources[0].error or "")


def test_extra_endpoints_are_tested_as_custom_sources() -> None:
    h = hub()
    result = probe(h, extra=["https://my-mirror.example/"])
    custom = next(s for s in result.sources if s.kind is SourceKind.custom)
    assert custom.endpoint == "https://my-mirror.example" and custom.reachable


def test_the_configured_endpoint_is_labelled_when_it_is_not_the_official_one() -> None:
    h = hub()
    result = probe_sources(
        h.client(), catalog_for_hub(), configured="https://hf-mirror.com", mirrors=["https://other.example"]
    )
    kinds = {s.endpoint: s.kind for s in result.sources}
    assert kinds == {
        OFFICIAL_ENDPOINT: SourceKind.official,
        "https://hf-mirror.com": SourceKind.configured,
        "https://other.example": SourceKind.mirror,
    }
    assert result.default_endpoint == "https://hf-mirror.com"


def test_fastest_reachable_source_is_recommended() -> None:
    h = hub()
    delays = {"huggingface.co": 0.15, "hf-mirror.com": 0.0}
    h.on_request = lambda r: time.sleep(delays[r.url.host])
    assert probe(h).recommended_endpoint == "https://hf-mirror.com"


def test_probe_needs_a_checksummed_file() -> None:
    empty = merge_entries([])
    with pytest.raises(ValueError, match="checksum"):
        probe_sources(hub().client(), empty, configured="https://huggingface.co", mirrors=[])


def test_probe_request_validates_endpoints() -> None:
    assert ProbeSourcesRequest(endpoints=["https://a.example/"]).endpoints == ["https://a.example"]
    with pytest.raises(ValueError):
        ProbeSourcesRequest(endpoints=["not a url"])
