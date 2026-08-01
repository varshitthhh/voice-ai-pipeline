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
from pathlib import Path
from typing import Optional

from .trace import STAGE_FIELDS, TurnTrace

DEFAULT_TRACE_DIR = Path("traces")


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
