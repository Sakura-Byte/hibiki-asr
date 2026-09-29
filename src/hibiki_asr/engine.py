"""The running engine: settings plus the model manager, the job manager and cached diagnostics."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from importlib import metadata

import httpx

from . import API_VERSION
from .diagnostics.findings import evaluate_findings
from .diagnostics.probe import probe_hardware
from .diagnostics.report import log_findings
from .diagnostics.runtime import probe_runtime
from .diagnostics.schema import Diagnostics, Finding, HardwareProbe, RuntimeFacts, Severity
from .diagnostics.selection import select_runtime
from .jobs.manager import JobManager
from .models.manager import ModelManager, load_catalog
from .models.store import ModelStore
from .provision.state import read_variant
from .settings import Settings, config_file_path
from .worker.process import WorkerFactory, default_factory

logger = logging.getLogger(__name__)

DIAGNOSTICS_CACHE_SECONDS = 30.0
_SECRETS = {"token", "hf_token"}


def engine_version() -> str:
    try:
        return metadata.version("hibiki-asr")
    except metadata.PackageNotFoundError:  # running from a source tree
        return "0.0.0+source"


class Engine:
    def __init__(
        self,
        settings: Settings,
        *,
        models: ModelManager | None = None,
        jobs: JobManager | None = None,
        worker_factory: WorkerFactory | None = None,
        hardware_probe: Callable[[], HardwareProbe] = probe_hardware,
        runtime_probe: Callable[[], RuntimeFacts] = probe_runtime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        if models is None:
            catalog, warnings = load_catalog(settings, config_file_path().parent)
            models = ModelManager(
                settings,
                catalog,
                ModelStore(settings.resolved_models_dir),
                client_factory=lambda: httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0, read=60.0)),
                warnings=warnings,
            )
        self.models = models
        self.jobs = jobs or JobManager(settings, worker_factory or default_factory(settings.log_level))
        self._hardware_probe, self._runtime_probe, self._clock = hardware_probe, runtime_probe, clock
        self._lock = threading.Lock()
        self._cached: tuple[float, Diagnostics] | None = None

    @property
    def variant(self) -> str | None:
        return read_variant(self.settings.data_dir)

    def hardware(self) -> HardwareProbe:
        """What the machine has, probed now (not cached)."""
        return self._hardware_probe()

    def public_settings(self) -> dict[str, object]:
        data = self.settings.model_dump(mode="json", exclude=_SECRETS)
        data["models_dir"] = str(self.settings.resolved_models_dir)
        return data

    def diagnostics(self, *, refresh: bool = False) -> Diagnostics:
        """Hardware, runtime, the device that will be used and the findings. Cached briefly: the probes are slow."""
        with self._lock:
            if self._cached and not refresh and self._clock() - self._cached[0] < DIAGNOSTICS_CACHE_SECONDS:
                return self._cached[1]

            hardware, runtime = self._hardware_probe(), self._runtime_probe()
            s = self.settings
            selection = select_runtime(hardware, runtime, s.device, s.compute_type, self.variant)
            findings = evaluate_findings(
                hardware, runtime, selection, variant=self.variant, requested_compute=s.compute_type
            )
            findings += [
                Finding(
                    code="CATALOG_SOURCE_IGNORED",
                    severity=Severity.warning,
                    message=w,
                    hint="Fix or remove that file.",
                )
                for w in self.models.catalog_warnings
            ]
            result = Diagnostics(
                api_version=API_VERSION,
                engine_version=engine_version(),
                variant=self.variant,
                hardware=hardware,
                runtime=runtime,
                selection=selection,
                findings=findings,
                settings=self.public_settings(),
            )
            self._cached = (self._clock(), result)
            return result

    def log_startup_report(self) -> None:
        """Say in the log which device is used and, if it is not the one expected, why and how to fix it."""
        try:
            log_findings(logger, self.diagnostics())
        except Exception:
            logger.exception("could not inspect the environment")

    def shutdown(self) -> None:
        self.jobs.stop()
        self.models.shutdown()
