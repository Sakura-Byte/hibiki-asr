"""On-disk layout of installed models.

    <models_dir>/
      state.json                     which version of each model is active
      <id>/<version>/                an installed version; the marker file says it is complete
      <id>/<version>.installing/     a download in progress (finished files are kept for resuming)
      .partial/                      half-downloaded files

A version becomes visible only when its ``.installing`` directory is renamed, so a crash never leaves a
half-installed version that looks usable.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from .catalog import Ref, VersionSpec

MARKER = ".hibiki-asr-installed.json"


class ModelStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    # -- paths ----------------------------------------------------------------------------------

    def version_dir(self, ref: Ref) -> Path:
        return self.root / ref.id / ref.version

    def staging_dir(self, ref: Ref) -> Path:
        return self.root / ref.id / f"{ref.version}.installing"

    @property
    def partial_root(self) -> Path:
        return self.root / ".partial"

    # -- installed versions -----------------------------------------------------------------------

    def read_marker(self, ref: Ref) -> dict | None:
        try:
            return json.loads((self.version_dir(ref) / MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def is_installed(self, ref: Ref) -> bool:
        return self.read_marker(ref) is not None

    def installed_versions(self, entry_id: str) -> list[str]:
        base = self.root / entry_id
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / MARKER).is_file())

    def is_intact(self, ref: Ref) -> bool:
        """Cheap check that every file listed in the marker is still there with its size."""
        marker = self.read_marker(ref)
        if marker is None:
            return False
        root = self.version_dir(ref)
        for f in marker.get("files", []):
            target = root / f["path"]
            if not target.is_file() or (f.get("size") is not None and target.stat().st_size != f["size"]):
                return False
        return True

    def commit(self, ref: Ref, spec: VersionSpec) -> None:
        """Turn a finished staging directory into an installed version."""
        staging = self.staging_dir(ref)
        marker = {
            "id": ref.id,
            "version": ref.version,
            "repo": spec.repo,
            "revision": spec.revision,
            "installed_at": time.time(),
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in spec.files],
        }
        (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        final = self.version_dir(ref)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)

    def delete(self, ref: Ref) -> None:
        shutil.rmtree(self.version_dir(ref), ignore_errors=True)
        shutil.rmtree(self.staging_dir(ref), ignore_errors=True)
        parent = self.root / ref.id
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def size_on_disk(self, ref: Ref) -> int:
        root = self.version_dir(ref)
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0

    # -- active version ---------------------------------------------------------------------------

    def _state(self) -> dict:
        try:
            state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
            return state if isinstance(state, dict) else {}
        except (OSError, ValueError):
            return {}

    def active(self, entry_id: str) -> str | None:
        return self._state().get("active", {}).get(entry_id)

    def set_active(self, entry_id: str, version: str | None) -> None:
        state = self._state()
        active = state.setdefault("active", {})
        if version is None:
            active.pop(entry_id, None)
        else:
            active[entry_id] = version
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".state-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(tmp, self.root / "state.json")  # atomic: a crash never leaves a truncated state file
