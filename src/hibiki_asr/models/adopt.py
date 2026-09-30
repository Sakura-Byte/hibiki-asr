"""Adopting model files that are already on disk: check them against the catalog, then stage them.

The manager decides what may be imported and commits the result; this module only knows how to verify a
directory against a ``VersionSpec`` and how to move or copy its files without ever losing the user's copy.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import VersionSpec
from .download import sha256_of


def find_problems(spec: VersionSpec, source: Path, on_file: Callable[[str], None] | None = None) -> list[str]:
    """Every way the files in ``source`` differ from what the catalog pins: missing, wrong size, wrong sha256.

    Size is compared first, so a truncated file is reported without hashing it. Files the catalog has no
    checksum for (a user's own ``catalog.local.toml``) are accepted as they are.
    """
    problems: list[str] = []
    for f in spec.files:
        target = source / f.path
        if not target.is_file():
            problems.append(f"{f.path}: missing")
            continue
        size = target.stat().st_size
        if f.size is not None and size != f.size:
            problems.append(f"{f.path}: expected {f.size} bytes, found {size}")
            continue
        if f.sha256 is not None:
            if on_file:
                on_file(f.path)
            if sha256_of(target) != f.sha256:
                problems.append(f"{f.path}: sha256 does not match the catalog")
    return problems


@dataclass
class Staged:
    """Files placed in a staging directory, and what it takes to finish or take back the transfer."""

    renamed: list[tuple[Path, Path]] = field(default_factory=list)  # (where it is now, where it came from)
    originals: list[Path] = field(default_factory=list)  # copies whose source is deleted once committed

    def undo(self) -> None:
        """Give every renamed file back to where it came from."""
        for staged, origin in self.renamed:
            if staged.exists():
                os.replace(staged, origin)
        self.renamed.clear()

    def finish(self) -> None:
        """The install is committed: remove the originals that were copied instead of renamed."""
        for original in self.originals:
            original.unlink(missing_ok=True)
        self.originals.clear()


def stage_files(spec: VersionSpec, source: Path, staging: Path, *, move: bool) -> Staged:
    """Copy the files into ``staging``, or with ``move`` rename them (copying across file systems).

    A copied file's original is only deleted by ``Staged.finish()``, after the commit, so a failure at any
    point leaves the user with every file: this function undoes its own renames before it re-raises.
    """
    staged = Staged()
    try:
        for f in spec.files:
            src, dest = source / f.path, staging / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if move:
                try:
                    os.replace(src, dest)
                except OSError:
                    pass  # another file system: fall through to a copy
                else:
                    staged.renamed.append((dest, src))
                    continue
            shutil.copy2(src, dest)
            if move:
                staged.originals.append(src)
    except BaseException:
        staged.undo()
        raise
    return staged
