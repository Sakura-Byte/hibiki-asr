"""Cooperative cancellation shared by the pipeline stages."""

from __future__ import annotations

from collections.abc import Callable


class JobCancelled(Exception):  # noqa: N818 - reads better than JobCancelledError at raise sites
    """Raised from a cancellation checkpoint once the job has been asked to stop."""


CancelCheck = Callable[[], None]


def never_cancelled() -> None:
    """A checkpoint that never cancels; used when the caller does not care."""
