"""Session-level tracer: turns stage spans into one JSONL row per turn.

Usage:
    tracer = Tracer(session_id="sess-001")
    tracer.start_turn("turn-01")
    tracer.mark("t_speech_start")
    ...
    tracer.mark("t_audio_out")
    trace = tracer.end_turn()   # appended to traces/sess-001.jsonl
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .trace import STAGE_FIELDS, TurnTrace, read_jsonl

DEFAULT_TRACE_DIR = Path("traces")


@dataclass
class ReplaySpan:
    """One stage mark, re-emitted from a historical trace file. Mirrors the
    (stage, timestamp) shape of a live Tracer.mark() call, so downstream
    analysis (Phase 6 stats, the dashboard) can consume a replayed session
    the same way it would consume a live one."""

    turn_id: str
    stage: str
    t: float


class Tracer:
    def __init__(self, session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR):
        self.session_id = session_id
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{session_id}.jsonl"
        self._current: Optional[TurnTrace] = None

    def start_turn(self, turn_id: str) -> None:
        if self._current is not None:
            raise RuntimeError(
                f"turn '{self._current.turn_id}' still open; call end_turn() first"
            )
        self._current = TurnTrace(session_id=self.session_id, turn_id=turn_id)

    def mark(self, stage: str, t: Optional[float] = None) -> float:
        """Record a monotonic timestamp for one of the seven span fields."""
        if stage not in STAGE_FIELDS:
            raise ValueError(f"unknown stage '{stage}', expected one of {STAGE_FIELDS}")
        if self._current is None:
            raise RuntimeError("no open turn; call start_turn() first")
        t = time.monotonic() if t is None else t
        setattr(self._current, stage, t)
        return t

    def end_turn(self) -> TurnTrace:
        if self._current is None:
            raise RuntimeError("no open turn to end")
        trace = self._current
        self._current = None
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(trace.to_json_line() + "\n")
        return trace

    @staticmethod
    def replay_session(jsonl_path: Path) -> Iterator[ReplaySpan]:
        """Reads an existing session JSONL and re-emits its spans, in
        STAGE_FIELDS order per turn, as an ordered stream of ReplaySpan
        events -- the same (turn_id, stage, t) shape a live Tracer.mark()
        call produces. Read-only: does not touch this Tracer's own state or
        write anything.

        Needed for Phase 6: re-running significance/regression analysis
        against historical traces without re-running the pipeline that
        produced them, and for the dashboard to step through an old session
        the same way it would a live one.
        """
        for trace in read_jsonl(Path(jsonl_path)):
            for stage in STAGE_FIELDS:
                t = getattr(trace, stage)
                if t is not None:
                    yield ReplaySpan(turn_id=trace.turn_id, stage=stage, t=t)
