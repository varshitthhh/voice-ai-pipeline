"""Per-turn trace schema.

Field order mirrors the README Section 1 latency budget table, so each
JSONL row maps directly onto the four measured spans: endpointing,
ASR final, LLM TTFT, TTS first chunk.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator, Optional

STAGE_FIELDS = (
    "t_speech_start",
    "t_vad_trigger",
    "t_endpoint_decision",
    "t_asr_final",
    "t_llm_first_token",
    "t_tts_first_chunk",
    "t_audio_out",
)


@dataclass
class TurnTrace:
    session_id: str
    turn_id: str
    t_speech_start: Optional[float] = None
    t_vad_trigger: Optional[float] = None
    t_endpoint_decision: Optional[float] = None
    t_asr_final: Optional[float] = None
    t_llm_first_token: Optional[float] = None
    t_tts_first_chunk: Optional[float] = None
    t_audio_out: Optional[float] = None

    def deltas_ms(self) -> dict:
        """Stage-to-stage latency in ms, matching the README Section 1 budget rows."""

        def gap(a: str, b: str) -> Optional[float]:
            ta, tb = getattr(self, a), getattr(self, b)
            return None if ta is None or tb is None else round((tb - ta) * 1000, 2)

        return {
            "endpointing_ms": gap("t_speech_start", "t_endpoint_decision"),
            "asr_final_ms": gap("t_endpoint_decision", "t_asr_final"),
            "llm_ttft_ms": gap("t_asr_final", "t_llm_first_token"),
            "tts_first_chunk_ms": gap("t_llm_first_token", "t_tts_first_chunk"),
            "e2e_ms": gap("t_speech_start", "t_audio_out"),
        }

    def to_json_line(self) -> str:
        row = asdict(self)
        row["deltas_ms"] = self.deltas_ms()
        return json.dumps(row)

    @classmethod
    def from_dict(cls, d: dict) -> "TurnTrace":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def read_jsonl(path: Path) -> Iterator[TurnTrace]:
    """Yield TurnTrace rows from a session JSONL file, in file order."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield TurnTrace.from_dict(json.loads(line))
