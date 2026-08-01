"""Phase 4.4: analyze end-to-end trace data, produce the final latency
breakdown table.

Reads back traces/e2e_phase4_4.jsonl (written by scripts/run_e2e_traces.py)
using Phase 0.2's read_jsonl() + TurnTrace.deltas_ms(), computes p50/p95/p99
per stage across all 100 turns, and compares against the README Section 1
budget.

Reading this table honestly requires knowing what's real vs. not, per
scripts/run_e2e_traces.py's own docstring:
  - endpointing_ms: REAL, but by construction near-constant (~500ms +/- one
    VAD frame of jitter) -- it's a fixed threshold, not a distribution.
    This is expected, not a bug: a fixed-threshold endpointer's entire
    latency IS the threshold, every time.
  - asr_final_ms / tts_first_chunk_ms: REAL compute time, but measured with
    CPU-sized models (faster-whisper "tiny", Piper) standing in for the
    production stack (distil-large-v3, Kokoro on GPU) -- expect these to
    run well above the README budget on this hardware/model combination.
    That's a hardware/model-size gap, not a pipeline defect.
  - llm_ttft_ms: TODO(gpu-required) mock, sampled from the budget range.
    vLLM cannot run on this machine at all.

Run:
    python scripts/analyze_e2e_traces.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing import read_jsonl

TRACE_PATH = ROOT / "traces" / "e2e_phase4_4.jsonl"
CSV_PATH = ROOT / "outputs" / "e2e_latency_breakdown.csv"

# (budget_min_ms, budget_max_ms, is_real_measurement) -- per README Section 1
# and the caveats in this module's docstring.
STAGE_INFO = {
    "endpointing_ms": (300, 700, "real (fixed-threshold wait, near-constant by construction)"),
    "asr_final_ms": (80, 150, "real compute (tiny/CPU model, not production distil-large-v3/GPU)"),
    "llm_ttft_ms": (90, 300, "TODO(gpu-required): mock, sampled from budget -- vLLM can't run here"),
    "tts_first_chunk_ms": (40, 90, "real compute (Piper/CPU, not production Kokoro/GPU)"),
    "e2e_ms": (510, 1240, "sum of the above"),
}


def percentile(values: list, p: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main() -> None:
    if not TRACE_PATH.exists():
        raise SystemExit(f"missing {TRACE_PATH} -- run scripts/run_e2e_traces.py first")

    traces = list(read_jsonl(TRACE_PATH))
    if not traces:
        raise SystemExit(f"{TRACE_PATH} is empty")

    print(f"loaded {len(traces)} turns from {TRACE_PATH}")

    per_stage_values = {stage: [] for stage in STAGE_INFO}
    for trace in traces:
        deltas = trace.deltas_ms()
        for stage in STAGE_INFO:
            v = deltas.get(stage)
            if v is not None:
                per_stage_values[stage].append(v)

    rows = []
    for stage, (budget_min, budget_max, note) in STAGE_INFO.items():
        values = per_stage_values[stage]
        if not values:
            continue
        p50 = percentile(values, 0.50)
        p95 = percentile(values, 0.95)
        p99 = percentile(values, 0.99)
        within_budget = budget_min <= p50 <= budget_max
        rows.append({
            "stage": stage,
            "n": len(values),
            "budget_min_ms": budget_min,
            "budget_max_ms": budget_max,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "within_budget_p50": within_budget,
            "note": note,
        })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'stage':<20}{'n':>5}{'budget':>14}{'p50':>10}{'p95':>10}{'p99':>10}   within budget (p50)")
    for r in rows:
        budget_str = f"{r['budget_min_ms']}-{r['budget_max_ms']}"
        flag = "yes" if r["within_budget_p50"] else "NO"
        print(f"{r['stage']:<20}{r['n']:>5}{budget_str:>14}{r['p50_ms']:>10}{r['p95_ms']:>10}{r['p99_ms']:>10}   {flag}")
        print(f"{'':<20}  note: {r['note']}")

    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
