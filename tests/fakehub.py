"""A Hugging Face look-alike for download tests: /{repo}/resolve/{revision}/{file}, with Range support."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from urllib.parse import unquote

import httpx

from hibiki_asr.models.catalog import FileSpec, VersionSpec

REPO = "acme/tiny-model"
REVISION = "a" * 40


def make_version(files: dict[str, bytes], *, repo: str = REPO, revision: str = REVISION) -> VersionSpec:
    return VersionSpec(
        version="v1",
        repo=repo,
        revision=revision,
        files=tuple(
            FileSpec(path, len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()
        ),
    )


class FakeHub:
    """Serves ``files`` on any host. Per-host behaviour is switched with the attributes below."""

    def __init__(self, files: dict[str, bytes], *, repo: str = REPO, revision: str = REVISION) -> None:
        self.repos: dict[tuple[str, str], dict[str, bytes]] = {(repo, revision): dict(files)}
        self.down_hosts: set[str] = set()  # answer 503
        self.unreachable_hosts: set[str] = set()  # the connection itself fails (blocked, no route)
        self.missing_hosts: set[str] = set()  # answer 404
        self.no_range_hosts: set[str] = set()  # ignore Range and send the whole file
        self.corrupt: set[str] = set()  # serve altered bytes for these paths
        self.fail_first: dict[str, int] = defaultdict(
            int
        )  # host -> number of requests to fail with 503 first
        self.requests: list[tuple[str, str, str | None]] = []  # (host, file, Range header)
        self.on_request = None  # optional hook(request) called before answering
        self._lock = threading.Lock()

    @property
    def files(self) -> dict[str, bytes]:
        """The files of the first (default) repository."""
        return next(iter(self.repos.values()))

    def add_repo(self, repo: str, revision: str, files: dict[str, bytes]) -> None:
        self.repos[(repo, revision)] = dict(files)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport())

    def hosts_used(self) -> set[str]:
        return {host for host, _, _ in self.requests}

    def _handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = unquote(request.url.path)
        match = re.fullmatch(r"/([^/]+/[^/]+)/resolve/([^/]+)/(.+)", path)
        rng = request.headers.get("Range")
        with self._lock:
            self.requests.append((host, match.group(3) if match else path, rng))
            if self.fail_first[host] > 0:
                self.fail_first[host] -= 1
                return httpx.Response(503)
        if host in self.unreachable_hosts:
            raise httpx.ConnectError("no route to host", request=request)
        if self.on_request:
            self.on_request(request)
        if host in self.down_hosts:
            return httpx.Response(503)
        repo_files = self.repos.get((match.group(1), match.group(2))) if match else None
        if host in self.missing_hosts or repo_files is None or match.group(3) not in repo_files:
            return httpx.Response(404)

        name = match.group(3)
        data = repo_files[name]
        if name in self.corrupt:
            data = bytes(b ^ 0xFF for b in data)

        if rng and host not in self.no_range_hosts:
            start_text, _, end_text = rng.removeprefix("bytes=").partition("-")
            start = int(start_text)
            if start >= len(data):
                return httpx.Response(416)
            end = int(end_text) if end_text else len(data) - 1
            end = min(end, len(data) - 1)
            return httpx.Response(
                206,
                content=data[start : end + 1],
                headers={"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            )
        return httpx.Response(200, content=data)
