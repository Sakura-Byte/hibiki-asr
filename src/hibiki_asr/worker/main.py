"""Entry point of the inference worker process.

The worker is a separate process on purpose. It can be killed to stop a job that will not stop, that also
returns all VRAM to the driver, and a native crash in the GPU runtime takes down only this process, not the API.
"""

from __future__ import annotations

import logging
import sys
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

from .protocol import Progress, RunRequest, Shutdown
from .session import WorkerSession


def worker_main(conn: Connection, cancel: Event, log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] worker: %(message)s",
        stream=sys.stderr,
    )
    from .loaders import RealLoaders

    session = WorkerSession(RealLoaders())
    while True:
        try:
            message = conn.recv()
        except EOFError:  # the parent went away
            return
        if isinstance(message, Shutdown):
            return
        if isinstance(message, RunRequest):
            reply = session.run(message, cancel.is_set, lambda progress: _send(conn, progress))
            _send(conn, reply)


def _send(conn: Connection, message: object) -> None:
    try:
        conn.send(message)
    except (BrokenPipeError, OSError):
        raise SystemExit(0) from None  # nobody is listening any more


__all__ = ["Progress", "worker_main"]
