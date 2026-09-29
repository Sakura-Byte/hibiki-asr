"""The model catalog: what can be installed, at which pinned revision, with which checksums.

Three sources are merged, later ones winning:
  1. the catalog shipped inside the package,
  2. a cached copy fetched from ``catalog_url`` (new model versions without a new engine release),
  3. ``catalog.local.toml`` in the config directory (the user's own models).

Only sources 1 and 2 must carry a sha256 for every file. The local file is the user's own machine
and their own choice, so it may omit them.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import PurePosixPath
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

SCHEMA = 1
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{7,64}$|^main$")
TASKS = ("transcribe", "translate")


class CatalogError(ValueError):
    """The catalog (or one of its sources) is malformed."""


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class VersionSpec:
    version: str
    repo: str
    revision: str
    files: tuple[FileSpec, ...]
    notes: str = ""

    @property
    def total_size(self) -> int | None:
        sizes = [f.size for f in self.files]
        return None if any(s is None for s in sizes) else sum(s for s in sizes if s is not None)


@dataclass(frozen=True)
class Ref:
    id: str
    version: str

    def __str__(self) -> str:
        return f"{self.id}@{self.version}"

    @classmethod
    def parse(cls, text: str) -> Ref:
        ident, sep, version = text.partition("@")
        if not sep or not ident or not version:
            raise CatalogError(f"expected 'id@version', got {text!r}")
        return cls(ident, version)


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str  # "model" | "component"
    display_name: str
    versions: tuple[VersionSpec, ...]
    license_note: str = ""
    task: str | None = None
    source_languages: tuple[str, ...] = ()
    output_languages: tuple[str, ...] = ()
    requires: tuple[Ref, ...] = ()

    def version(self, version: str) -> VersionSpec | None:
        return next((v for v in self.versions if v.version == version), None)

    @property
    def latest(self) -> VersionSpec:
        """The version listed last is the newest."""
        return self.versions[-1]


@dataclass(frozen=True)
class Catalog:
    entries: Mapping[str, Entry] = field(default_factory=dict)

    def get(self, entry_id: str) -> Entry | None:
        return self.entries.get(entry_id)

    def models(self) -> list[Entry]:
        return [e for e in self.entries.values() if e.kind == "model"]

    def resolve(self, ref: Ref) -> tuple[Entry, VersionSpec]:
        entry = self.entries.get(ref.id)
        version = entry.version(ref.version) if entry else None
        if entry is None or version is None:
            raise KeyError(str(ref))
        return entry, version

    def dependents_of(self, ref: Ref) -> list[Ref]:
        """Every entry version that lists ``ref`` in ``requires``."""
        return [Ref(e.id, v.version) for e in self.entries.values() if ref in e.requires for v in e.versions]


def _fail(where: str, message: str) -> CatalogError:
    return CatalogError(f"{where}: {message}")


def _parse_file(raw: Mapping[str, Any], where: str, *, require_hashes: bool) -> FileSpec:
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise _fail(where, "file without a path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise _fail(where, f"unsafe file path {path!r}")
    sha256, size = raw.get("sha256"), raw.get("size")
    if sha256 is not None and not (isinstance(sha256, str) and _SHA256.match(sha256)):
        raise _fail(where, f"{path}: sha256 must be 64 lowercase hex characters")
    if size is not None and not (isinstance(size, int) and size >= 0):
        raise _fail(where, f"{path}: size must be a non-negative integer")
    if require_hashes and (sha256 is None or size is None):
        raise _fail(where, f"{path}: size and sha256 are required")
    return FileSpec(path, size, sha256)


def _parse_version(raw: Mapping[str, Any], where: str, *, require_hashes: bool) -> VersionSpec:
    version = raw.get("version")
    if not isinstance(version, str) or not _VERSION.match(version):
        raise _fail(where, f"invalid version {version!r}")
    where = f"{where}@{version}"
    repo, revision = raw.get("repo"), raw.get("revision", "main")
    if not isinstance(repo, str) or not _REPO.match(repo):
        raise _fail(where, f"invalid repo {repo!r}; expected 'owner/name'")
    if not isinstance(revision, str) or not _REVISION.match(revision):
        raise _fail(where, f"invalid revision {revision!r}")
    if require_hashes and revision == "main":
        raise _fail(where, "a commit revision is required, not 'main'")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise _fail(where, "no files")
    return VersionSpec(
        version=version,
        repo=repo,
        revision=revision,
        notes=str(raw.get("notes", "")),
        files=tuple(_parse_file(f, where, require_hashes=require_hashes) for f in files),
    )


def _parse_entry(raw: Mapping[str, Any], *, require_hashes: bool) -> Entry:
    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not _ID.match(entry_id):
        raise CatalogError(f"invalid entry id {entry_id!r}")
    kind = raw.get("kind", "model")
    if kind not in ("model", "component"):
        raise _fail(entry_id, f"kind must be 'model' or 'component', got {kind!r}")

    task = raw.get("task")
    sources, outputs = tuple(raw.get("source_languages", ())), tuple(raw.get("output_languages", ()))
    if kind == "model":
        if task not in TASKS:
            raise _fail(entry_id, f"task must be one of {', '.join(TASKS)}")
        if not sources or not outputs:
            raise _fail(entry_id, "a model needs source_languages and output_languages")

    versions = raw.get("versions")
    if not isinstance(versions, list) or not versions:
        raise _fail(entry_id, "no versions")
    parsed = tuple(_parse_version(v, entry_id, require_hashes=require_hashes) for v in versions)
    if len({v.version for v in parsed}) != len(parsed):
        raise _fail(entry_id, "duplicate version labels")

    return Entry(
        id=entry_id,
        kind=kind,
        display_name=str(raw.get("display_name", entry_id)),
        versions=parsed,
        license_note=str(raw.get("license_note", "")),
        task=task,
        source_languages=sources,
        output_languages=outputs,
        requires=tuple(Ref.parse(r) for r in raw.get("requires", ())),
    )


def parse_catalog(data: Mapping[str, Any], *, require_hashes: bool) -> list[Entry]:
    """Parse one source. The entries live under ``entries`` (JSON) or ``entry`` (TOML array of tables)."""
    schema = data.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise CatalogError(f"unsupported catalog schema {schema!r}; this engine understands {SCHEMA}")
    raw_entries = data.get("entries", data.get("entry", []))
    if not isinstance(raw_entries, list):
        raise CatalogError("entries must be a list")
    return [_parse_entry(e, require_hashes=require_hashes) for e in raw_entries]


def merge_entries(base: Iterable[Entry], *overlays: Iterable[Entry]) -> Catalog:
    """Later sources win per entry id; versions are unioned, the later source winning on a shared label."""
    merged: dict[str, Entry] = {e.id: e for e in base}
    for overlay in overlays:
        for entry in overlay:
            existing = merged.get(entry.id)
            if existing is None:
                merged[entry.id] = entry
                continue
            versions = {v.version: v for v in existing.versions}
            versions.update({v.version: v for v in entry.versions})
            merged[entry.id] = replace(entry, versions=tuple(versions.values()))
    catalog = Catalog(merged)
    _check_requirements(catalog)
    return catalog


def _check_requirements(catalog: Catalog) -> None:
    for entry in catalog.entries.values():
        for ref in entry.requires:
            component = catalog.get(ref.id)
            if component is None or component.kind != "component" or component.version(ref.version) is None:
                raise CatalogError(f"{entry.id} requires {ref}, which is not a component in the catalog")


def load_bundled() -> list[Entry]:
    text = resources.files("hibiki_asr.models").joinpath("catalog.json").read_text(encoding="utf-8")
    return parse_catalog(json.loads(text), require_hashes=True)


def parse_local_toml(text: str) -> list[Entry]:
    try:
        return parse_catalog(tomllib.loads(text), require_hashes=False)
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"invalid TOML: {exc}") from exc
