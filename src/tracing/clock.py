"""Monotonic timing primitives.

Everything in the tracing harness is timestamped with time.monotonic(),
never wall-clock time, so spans stay immune to clock adjustments and are
directly subtractable across a session.
"""

import time
from dataclasses import dataclass, field


def now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class FrameTimestamp:
    index: int
    t: float          # absolute monotonic seconds
    t_offset: float    # seconds since the clock was created


@dataclass
class FrameClock:
    """Assigns a monotonic timestamp to each audio frame as it arrives.

    One instance per session/stream. Call `tick()` once per audio frame
    (e.g. each 10-30ms chunk pulled off the mic) to get its absolute and
    session-relative timestamp.
    """

    t0: float = field(default_factory=now)
    _frame_index: int = field(default=0, init=False)

    def tick(self) -> FrameTimestamp:
        t = now()
        ts = FrameTimestamp(index=self._frame_index, t=t, t_offset=t - self.t0)
        self._frame_index += 1
        return ts

    def elapsed(self) -> float:
        return now() - self.t0
