"""Which Hugging Face endpoints can this machine actually download from?

A source only counts as reachable when a tiny pinned file (the smallest one in the catalog) comes back
with the right sha256. That is stricter than "the TCP connection worked": it also catches captive
portals, broken mirrors and proxies that answer with an error page.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

from .catalog import Catalog, FileSpec, VersionSpec
from .schema import SourceKind, SourcesResponse, SourceStatus

OFFICIAL_ENDPOINT = "https://huggingface.co"
PROBE_TIMEOUT_S = 8.0


def smallest_pinned_file(catalog: Catalog) -> tuple[VersionSpec, FileSpec] | None:
    """The smallest file that has both a size and a checksum, or None for a catalog without any."""
    candidates = [
        (version, f)
        for entry in catalog.entries.values()
        for version in entry.versions
        for f in version.files
        if f.size is not None and f.sha256 is not None
    ]
    return min(candidates, key=lambda c: c[1].size or 0, default=None)


def _probe_one(
    client: httpx.Client,
    endpoint: str,
    kind: SourceKind,
    target: tuple[VersionSpec, FileSpec],
    clock: Callable[[], float],
) -> SourceStatus:
    version, spec = target
    url = f"{endpoint}/{version.repo}/resolve/{version.revision}/{quote(spec.path)}"
    started = clock()
    try:
        response = client.get(url, follow_redirects=True, timeout=PROBE_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return SourceStatus(
            endpoint=endpoint, kind=kind, reachable=False, error=f"HTTP {exc.response.status_code}"
        )
    except httpx.HTTPError as exc:
        return SourceStatus(
            endpoint=endpoint, kind=kind, reachable=False, error=f"{type(exc).__name__}: {exc}"
        )

    latency_ms = max(0, int((clock() - started) * 1000))
    if hashlib.sha256(response.content).hexdigest() != spec.sha256:
        return SourceStatus(
            endpoint=endpoint,
            kind=kind,
            reachable=False,
            latency_ms=latency_ms,
            error="reachable, but it returned different content than the catalog pins (a proxy or portal page?)",
        )
    return SourceStatus(endpoint=endpoint, kind=kind, reachable=True, latency_ms=latency_ms)


def probe_sources(
    client: httpx.Client,
    catalog: Catalog,
    *,
    configured: str,
    mirrors: Sequence[str],
    extra: Iterable[str] = (),
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], float] = time.time,
) -> SourcesResponse:
    """Test the official endpoint, the configured one, the mirrors and any extra candidates, concurrently."""
    plan: dict[str, SourceKind] = {OFFICIAL_ENDPOINT: SourceKind.official}
    for endpoint, kind in [(configured, SourceKind.configured), *((m, SourceKind.mirror) for m in mirrors)]:
        plan.setdefault(endpoint.rstrip("/"), kind)
    for endpoint in extra:
        plan.setdefault(endpoint.rstrip("/"), SourceKind.custom)

    target = smallest_pinned_file(catalog)
    if target is None:
        raise ValueError("the catalog has no file with a checksum to probe with")

    with ThreadPoolExecutor(max_workers=len(plan), thread_name_prefix="probe") as pool:
        futures = [
            pool.submit(_probe_one, client, endpoint, kind, target, clock) for endpoint, kind in plan.items()
        ]
        statuses = [f.result() for f in futures]

    working = sorted((s for s in statuses if s.reachable), key=lambda s: s.latency_ms or 0)
    official = next(s for s in statuses if s.endpoint == OFFICIAL_ENDPOINT)
    return SourcesResponse(
        huggingface_reachable=official.reachable,
        recommended_endpoint=working[0].endpoint if working else None,
        default_endpoint=configured.rstrip("/"),
        sources=statuses,
        checked_at=now(),
    )
