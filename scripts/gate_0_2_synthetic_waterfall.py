"""Phase 0.2 gate: prove the tracing harness end-to-end with synthetic spans.

No models are loaded here — stage durations are sampled from the budget
ranges fixed in README Section 1. This validates the Tracer -> JSONL ->
waterfall chart path before any real audio, ASR, LLM, or TTS component
exists.

Run:
    python scripts/gate_0_2_synthetic_waterfall.py
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tracing import STAGE_FIELDS, Tracer, read_jsonl

SESSION_ID = "gate-0.2-synthetic"
N_TURNS = 20
OUTLIER_TURN_INDEX = 17  # one deliberate p95-tail outlier to sanity-check the chart

# (min_ms, max_ms) per README Section 1.1 — same ranges the real components
# will be held to in Phase 2+.
BUDGET_MS = {
    "vad_trigger": (60, 150),        # VAD fires well before the full endpoint decision
    "endpointing": (300, 700),       # t_speech_start -> t_endpoint_decision
    "asr_final": (80, 150),
    "llm_ttft": (90, 300),
    "tts_first_chunk": (40, 90),
    "audio_out_flush": (10, 30),     # buffer flush after first TTS chunk is ready
}


def synth_turn(tracer: Tracer, turn_id: str, outlier: bool = False) -> None:
    mult = 2.2 if outlier else 1.0  # blow past budget on the outlier turn, on purpose

    base = time.monotonic()
    t = base
    tracer.start_turn(turn_id)

    tracer.mark("t_speech_start", t)

    t_vad = t + random.uniform(*BUDGET_MS["vad_trigger"]) / 1000
    tracer.mark("t_vad_trigger", t_vad)

    t_endpoint = t + (random.uniform(*BUDGET_MS["endpointing"]) * mult) / 1000
    t_endpoint = max(t_endpoint, t_vad + 0.01)
    tracer.mark("t_endpoint_decision", t_endpoint)

    t_asr = t_endpoint + (random.uniform(*BUDGET_MS["asr_final"]) * mult) / 1000
    tracer.mark("t_asr_final", t_asr)

    t_llm = t_asr + (random.uniform(*BUDGET_MS["llm_ttft"]) * mult) / 1000
    tracer.mark("t_llm_first_token", t_llm)

    t_tts = t_llm + (random.uniform(*BUDGET_MS["tts_first_chunk"]) * mult) / 1000
    tracer.mark("t_tts_first_chunk", t_tts)

    t_audio = t_tts + random.uniform(*BUDGET_MS["audio_out_flush"]) / 1000
    tracer.mark("t_audio_out", t_audio)

    tracer.end_turn()


def generate(trace_dir: Path) -> Path:
    random.seed(0)
    tracer = Tracer(session_id=SESSION_ID, trace_dir=trace_dir)
    for i in range(N_TURNS):
        synth_turn(tracer, turn_id=f"turn-{i:02d}", outlier=(i == OUTLIER_TURN_INDEX))
    return tracer.path


def validate(path: Path) -> list:
    traces = list(read_jsonl(path))
    assert len(traces) == N_TURNS, f"expected {N_TURNS} rows, got {len(traces)}"
    for tr in traces:
        for field_name in STAGE_FIELDS:
            assert getattr(tr, field_name) is not None, f"{tr.turn_id} missing {field_name}"
        deltas = tr.deltas_ms()
        assert deltas["e2e_ms"] > 0, f"{tr.turn_id} has non-positive e2e latency"
    print(f"validated {len(traces)} turns from {path}")
    return traces


STAGE_COLORS = {
    "endpointing_ms": "#4C72B0",
    "asr_final_ms": "#DD8452",
    "llm_ttft_ms": "#55A868",
    "tts_first_chunk_ms": "#C44E52",
}
STAGE_LABELS = {
    "endpointing_ms": "Endpointing",
    "asr_final_ms": "ASR final",
    "llm_ttft_ms": "LLM TTFT",
    "tts_first_chunk_ms": "TTS first chunk",
}


def plot_waterfall(traces: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for row, tr in enumerate(traces):
        deltas = tr.deltas_ms()
        left = 0.0
        for stage in STAGE_COLORS:
            width = deltas[stage]
            ax.barh(row, width, left=left, color=STAGE_COLORS[stage], edgecolor="white", height=0.7)
            left += width

    ax.set_yticks(range(len(traces)))
    ax.set_yticklabels([tr.turn_id for tr in traces])
    ax.invert_yaxis()
    ax.set_xlabel("ms since t_speech_start")
    ax.set_title(f"Synthetic turn latency waterfall — session {SESSION_ID}")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS.values()]
    ax.legend(handles, [STAGE_LABELS[s] for s in STAGE_COLORS], loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    trace_path = generate(ROOT / "traces")
    traces = validate(trace_path)
    plot_waterfall(traces, ROOT / "outputs" / "waterfall_synthetic.png")

    e2e = [tr.deltas_ms()["e2e_ms"] for tr in traces]
    e2e.sort()
    p50 = e2e[len(e2e) // 2]
    p95 = e2e[int(len(e2e) * 0.95)]
    print(f"synthetic e2e p50={p50:.1f}ms p95={p95:.1f}ms (target: p50<=700ms, p95<=1100ms)")


if __name__ == "__main__":
    main()
