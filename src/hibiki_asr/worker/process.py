"""The API side of the worker: start it, talk to it, and stop it (politely or not)."""

from __future__ import annotations

import contextlib
import multiprocessing
from collections.abc import Callable
from typing import Protocol

from .main import worker_main


class WorkerHandle(Protocol):
    def send(self, message: object) -> None: ...
    def poll(self, timeout: float) -> object | None:
        """The next message from the worker, or None if none arrived within ``timeout`` seconds."""

    def is_alive(self) -> bool: ...
    def exit_code(self) -> int | None: ...
    def request_cancel(self) -> None: ...
    def clear_cancel(self) -> None: ...
    def kill(self) -> None: ...
    def shutdown(self, timeout: float) -> None: ...


WorkerFactory = Callable[[], WorkerHandle]


class ProcessWorker:
    """A worker running in a child process (``spawn``, so CUDA state is never inherited)."""

    def __init__(self, log_level: str = "INFO") -> None:
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe(duplex=True)
        self._cancel = context.Event()
        self._process = context.Process(
            target=worker_main, args=(child, self._cancel, log_level), name="hibiki-asr-worker", daemon=True
        )
        self._process.start()
        child.close()

    def send(self, message: object) -> None:
        self._conn.send(message)

    def poll(self, timeout: float) -> object | None:
        try:
            if self._conn.poll(timeout):
                return self._conn.recv()
        except (EOFError, OSError):
            return None  # the process died; is_alive() / exit_code() tell the caller
        return None

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def exit_code(self) -> int | None:
        return self._process.exitcode

    def request_cancel(self) -> None:
        self._cancel.set()

    def clear_cancel(self) -> None:
        self._cancel.clear()

    def kill(self) -> None:
        if self._process.is_alive():
            self._process.kill()
        self._process.join(timeout=5)
        self._conn.close()

    def shutdown(self, timeout: float) -> None:
        from .protocol import Shutdown

        with contextlib.suppress(BrokenPipeError, OSError):  # already gone: nothing to tell it
            self._conn.send(Shutdown())
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self.kill()
        else:
            self._conn.close()


def default_factory(log_level: str = "INFO") -> WorkerFactory:
    return lambda: ProcessWorker(log_level)
