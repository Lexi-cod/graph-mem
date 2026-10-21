"""Time sources. Evaluation uses SimClock so no wall-clock time leaks into results."""

from datetime import datetime
from typing import Protocol

from .schema import utcnow


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    def now(self) -> datetime:
        return utcnow()


class SimClock:
    """Simulated time. Raises if read before being set, so a forgotten `set`
    fails loudly instead of silently falling back to the wall clock."""

    def __init__(self, t: datetime | None = None):
        self._t = t

    def set(self, t: datetime) -> None:
        if t.tzinfo is None:
            raise ValueError("SimClock requires timezone-aware datetimes")
        self._t = t

    def now(self) -> datetime:
        if self._t is None:
            raise RuntimeError("SimClock read before set()")
        return self._t
