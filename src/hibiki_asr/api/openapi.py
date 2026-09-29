"""The OpenAPI document that clients (Hibiki) build against. Committed as openapi.json and kept current by a test."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from ..engine import Engine
from ..settings import Settings
from .app import create_app


def build_openapi() -> dict[str, Any]:
    """Generate the spec from the app. Nothing is started, probed or written outside a throw-away directory."""

    def no_worker() -> NoReturn:
        raise RuntimeError("the spec generator never runs jobs")

    with tempfile.TemporaryDirectory() as directory:
        engine = Engine(Settings(data_dir=Path(directory)), worker_factory=no_worker)
        try:
            return create_app(engine).openapi()
        finally:
            engine.shutdown()


def render_openapi() -> str:
    return json.dumps(build_openapi(), indent=2, ensure_ascii=False) + "\n"
