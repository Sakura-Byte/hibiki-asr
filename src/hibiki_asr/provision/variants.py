"""Install variants and the rule that picks one for a machine."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

from ..diagnostics.schema import GpuInfo, HardwareProbe

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib


STATUSES = ("stable", "experimental")


@dataclass(frozen=True)
class Variant:
    id: str
    title: str
    os: tuple[str, ...]
    gpu_vendor: str  # "none" | "nvidia" | "amd"
    lockfile: str  # "" when no verified pins exist
    docker: bool
    min_driver: int | None = None
    min_cc: tuple[int, int] | None = None
    min_cc_max: tuple[int, int] | None = None
    gfx: tuple[str, ...] = field(default_factory=tuple)
    docker_tag: str = ""
    status: str = "stable"
    note: str = ""

    def supports_os(self, platform: str) -> bool:
        return any(platform.startswith(name) for name in self.os)

    @property
    def experimental(self) -> bool:
        return self.status == "experimental"

    @property
    def installable(self) -> bool:
        """Pinned runtime dependencies exist, so `hibiki-asr setup` can install this variant."""
        return bool(self.lockfile)


def parse_cc(value: str | None) -> tuple[int, int] | None:
    """``"8.9"`` -> ``(8, 9)``; anything unparsable -> None."""
    if not value:
        return None
    try:
        major, _, minor = value.strip().partition(".")
        return int(major), int(minor or 0)
    except ValueError:
        return None


def driver_major(driver: str | None) -> int | None:
    """``"555.42.06"`` -> 555."""
    if not driver:
        return None
    try:
        return int(driver.strip().split(".")[0])
    except ValueError:
        return None


@cache
def load_variants() -> dict[str, Variant]:
    text = resources.files("hibiki_asr.provision").joinpath("variants.toml").read_text(encoding="utf-8")
    variants: dict[str, Variant] = {}
    for row in tomllib.loads(text)["variant"]:
        if row.get("status", "stable") not in STATUSES:
            raise ValueError(f"variant {row['id']!r}: status must be one of {', '.join(STATUSES)}")
        variants[row["id"]] = Variant(
            id=row["id"],
            title=row["title"],
            os=tuple(row["os"]),
            gpu_vendor=row["gpu_vendor"],
            lockfile=row["lockfile"],
            docker=bool(row["docker"]),
            min_driver=row.get("min_driver"),
            min_cc=parse_cc(row.get("min_cc")),
            min_cc_max=parse_cc(row.get("min_cc_max")),
            gfx=tuple(row.get("gfx", ())),
            docker_tag=row.get("docker_tag", ""),
            status=row.get("status", "stable"),
            note=row.get("note", ""),
        )
    return variants


def get_variant(variant_id: str) -> Variant:
    try:
        return load_variants()[variant_id]
    except KeyError:
        known = ", ".join(sorted(load_variants()))
        raise KeyError(f"unknown variant {variant_id!r}; known variants: {known}") from None


def setup_command(variant: Variant) -> str:
    """The command that installs ``variant``, as findings and hints print it."""
    flag = " --allow-experimental" if variant.experimental and variant.installable else ""
    return f"hibiki-asr setup --variant {variant.id}{flag}"


def _nvidia_variant(gpu: GpuInfo, platform: str) -> Variant | None:
    """Best NVIDIA build for one card, or None when its driver is too old for every build."""
    cc = parse_cc(gpu.compute_capability)
    driver = driver_major(gpu.driver)
    candidates = [v for v in load_variants().values() if v.gpu_vendor == "nvidia" and v.supports_os(platform)]

    def fits(v: Variant) -> bool:
        if cc is not None:
            if v.min_cc is not None and cc < v.min_cc:
                return False
            if v.min_cc_max is not None and cc >= v.min_cc_max:
                return False
        return driver is None or v.min_driver is None or driver >= v.min_driver

    # Prefer the newest build the card and driver can run.
    for variant_id in ("cuda12-blackwell", "cuda12", "cuda11"):
        variant = next((v for v in candidates if v.id == variant_id), None)
        if variant is not None and fits(variant):
            return variant
    return None


def _amd_variant(gpu: GpuInfo, platform: str) -> Variant | None:
    if gpu.integrated or not gpu.gfx:
        return None
    for variant in load_variants().values():
        if variant.gpu_vendor == "amd" and variant.supports_os(platform) and gpu.gfx in variant.gfx:
            return variant
    return None


def recommend_variant(probe: HardwareProbe, platform: str | None = None) -> Variant:
    """The variant `setup --variant auto` installs. Falls back to CPU when no GPU build fits."""
    platform = platform or sys.platform
    for gpu in probe.gpus:
        if gpu.vendor == "nvidia":
            found = _nvidia_variant(gpu, platform)
        elif gpu.vendor == "amd":
            found = _amd_variant(gpu, platform)
        else:
            found = None
        if found is not None:
            return found
    return load_variants()["cpu"]
