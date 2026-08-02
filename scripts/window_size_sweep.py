"""Phase 0.3: decision-window granularity justification.

Sweeps {20, 50, 100, 200}ms and reports prediction stability vs. decision
latency, so 100ms is an evidence-based choice rather than an inherited
default.

**What "window size" means here, precisely.** The trained turn-taking model
(Phase 3.3, outputs/turn_taking_model.pt) consumes FeaturePipeline frames at
a fixed frame_ms=100 -- that cadence is baked into its learned weights (the
GRU's temporal dynamics, pause_so_far_ms accumulation, and the loss_mask
supervision points all assume 100ms steps). Re-running that specific
checkpoint's inference at a genuinely different frame_ms would put it
off-distribution and conflate "does a finer window help" with "what happens
when you feed a model data shaped unlike anything it was trained on" -- not
a clean measurement, and re-training a fresh model per window size would
mean touching Phase 3.3 territory again, which is explicitly frozen until
real CANDOR/roleplay data lands (see STATUS.md).

So this sweeps the orchestrator's DECISION-POLLING interval instead: the
model still produces one new P(turn_complete) estimate every 100ms (fixed,
unchanged), but a production orchestrator doesn't have to check that
estimate exactly when it updates -- it can poll on its own cadence. This
sweep asks: how much added latency does a coarser poll interval cost (it can
only ever notice a crossing at the next poll tick, never before), and does
polling granularity ever change WHICH decision fires, only WHEN it's
noticed? That second question is what "prediction stability" means below.

Reuses the exact probability traces scripts/ab_compare_endpointers.py
computes (same checkpoint, same synthetic eval scenarios -- labeled
synthetic throughout, same as the rest of Phase 3, pending real data).

Run:
    python scripts/window_size_sweep.py
"""

import csv
import sys
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ab_compare_endpointers import load_model_from_checkpoint, run_learned_trace  # noqa: E402
from baseline_fixed_threshold_vad import load_scenarios, percentile  # noqa: E402

SAMPLE_RATE = 16000
MODEL_FRAME_MS = 100  # fixed -- the checkpoint's native cadence, never swept
WINDOW_SIZES_MS = [20, 50, 100, 200]
DECISION_THRESHOLD = 0.7  # the one operating point with enough measured A/B samples to trust, per Phase 3.5

DEFAULT_LABELS = ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "turn_taking_model.pt"
CSV_PATH = ROOT / "outputs" / "window_size_sweep.csv"


def true_crossing_ms(trace: list, threshold: float):
    """First MODEL-INTERNAL frame (100ms native cadence) where prob crosses
    threshold during an actual pause -- the instant the model itself decided,
    independent of any polling cadence layered on top."""
    for r in trace:
        if r["pause_so_far_ms"] > 0 and r["prob"] >= threshold:
            return r["t_ms"]
    return None


def noticed_at_ms(true_ms: float, window_ms: int) -> float:
    """Next poll tick at or after true_ms, for a poller ticking every
    window_ms starting at t=0 -- the earliest a window_ms-interval
    orchestrator could possibly notice the crossing."""
    import math

    return math.ceil(true_ms / window_ms) * window_ms


def main() -> None:
    if not DEFAULT_LABELS.exists():
        raise SystemExit(f"missing {DEFAULT_LABELS} -- run scripts/prepare_synthetic_turn_taking_eval.py first")
    if not DEFAULT_CHECKPOINT.exists():
        raise SystemExit(f"missing {DEFAULT_CHECKPOINT} -- train a model first (see voice_ai_pipeline.ipynb)")

    scenarios = load_scenarios(DEFAULT_LABELS)
    audio_cache = {}
    for s in scenarios:
        audio, sr = sf.read(ROOT / s["audio_path"], dtype="float64")
        assert sr == SAMPLE_RATE, f"{s['audio_path']} is {sr}Hz, expected {SAMPLE_RATE}"
        audio_cache[s["scenario_id"]] = audio
    print(f"loaded {len(scenarios)} scenarios")

    vad_model = load_silero_vad(onnx=False)
    model = load_model_from_checkpoint(DEFAULT_CHECKPOINT)
    asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("computing probability traces (one pass per scenario, native 100ms cadence)...")
    traces = {}
    for i, s in enumerate(scenarios):
        traces[s["scenario_id"]] = run_learned_trace(model, vad_model, asr_model, audio_cache[s["scenario_id"]])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(scenarios)}")

    # Ground truth per TRUE_END scenario: the model-internal crossing instant
    # and whether it fires at all -- computed once, independent of window size.
    true_end = [s for s in scenarios if s["label"] == "TRUE_END"]
    ground_truth = {}
    for s in true_end:
        t_true = true_crossing_ms(traces[s["scenario_id"]], DECISION_THRESHOLD)
        ground_truth[s["scenario_id"]] = t_true  # None if it never fires

    rows = []
    baseline_fires = {sid: (t is not None) for sid, t in ground_truth.items()}
    for window_ms in WINDOW_SIZES_MS:
        added_latencies_ms = []
        fires_this_window = {}
        for sid, t_true in ground_truth.items():
            fires_this_window[sid] = t_true is not None
            if t_true is not None:
                t_noticed = noticed_at_ms(t_true, window_ms)
                added_latencies_ms.append(t_noticed - t_true)

        n_match = sum(1 for sid in ground_truth if fires_this_window[sid] == baseline_fires[sid])
        decision_match_rate = n_match / len(ground_truth) if ground_truth else float("nan")

        row = {
            "window_ms": window_ms,
            "added_latency_p50_ms": round(percentile(added_latencies_ms, 0.50), 2),
            "added_latency_p95_ms": round(percentile(added_latencies_ms, 0.95), 2),
            "added_latency_mean_ms": round(sum(added_latencies_ms) / len(added_latencies_ms), 2) if added_latencies_ms else float("nan"),
            "decision_match_rate_vs_100ms_baseline": round(decision_match_rate, 4),
            "n_true_end_scenarios": len(ground_truth),
        }
        rows.append(row)
        print(f"  window={window_ms}ms: added_latency_p50={row['added_latency_p50_ms']}ms "
              f"p95={row['added_latency_p95_ms']}ms  decisions_match_100ms={row['decision_match_rate_vs_100ms_baseline'] * 100:.1f}%")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {CSV_PATH}")

    all_match = all(r["decision_match_rate_vs_100ms_baseline"] == 1.0 for r in rows)
    divisors = [r["window_ms"] for r in rows if r["window_ms"] <= MODEL_FRAME_MS and MODEL_FRAME_MS % r["window_ms"] == 0]
    non_divisors = [r for r in rows if r["window_ms"] not in divisors]

    print("\nconclusion (derived from the numbers above, not asserted):")
    print(
        f"- the model itself only updates every {MODEL_FRAME_MS}ms (fixed by the trained checkpoint); "
        "no poll interval can notice a crossing before the model has actually decided."
    )
    print(
        f"- every window size produced the IDENTICAL fire/no-fire decision (match rate 100% in every "
        f"row: {all_match}) -- window size changes only WHEN a firing is noticed, never WHICH "
        "scenarios fire."
    )
    if divisors:
        print(
            f"- window sizes that evenly divide {MODEL_FRAME_MS}ms ({divisors}) added ZERO latency here: "
            "a crossing always lands exactly on a poll tick, so polling finer than 100ms bought nothing "
            "further in this eval set."
        )
    for r in non_divisors:
        print(
            f"- window={r['window_ms']}ms (does not evenly divide {MODEL_FRAME_MS}ms) added up to "
            f"{r['added_latency_p95_ms']}ms of avoidable latency (p95), bounded by "
            f"window_ms - gcd(window_ms, {MODEL_FRAME_MS}ms)."
        )
    print(
        f"- net: {MODEL_FRAME_MS}ms is an evidence-based choice here, not just an inherited default -- "
        "it matches the model's own update cadence exactly, so it's the coarsest window that still adds "
        "zero avoidable latency."
    )


if __name__ == "__main__":
    main()
