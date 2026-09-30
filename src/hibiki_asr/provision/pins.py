"""Access to the pinned runtime lockfiles shipped inside the package (`provision/lockfiles/<variant>.txt`)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from .variants import Variant

if TYPE_CHECKING:
    from importlib.abc import Traversable

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", re.MULTILINE)


def _resource(variant: Variant) -> Traversable:
    if not variant.installable:
        raise LookupError(f"variant {variant.id!r} has no lockfile")
    return resources.files("hibiki_asr.provision").joinpath("lockfiles").joinpath(variant.lockfile)


def lockfile_text(variant: Variant) -> str:
    return _resource(variant).read_text(encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lockfile_sha256(variant: Variant) -> str:
    return sha256_bytes(_resource(variant).read_bytes())


@contextmanager
def lockfile_path(variant: Variant) -> Iterator[Path]:
    """A real file for the lockfile (extracted first if the package is not on the file system)."""
    with resources.as_file(_resource(variant)) as path:
        yield path


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_names(text: str) -> set[str]:
    """Normalized names of every ``name==version`` requirement in a lockfile."""
    return {normalize_name(match.group(1)) for match in _PIN.finditer(text)}
