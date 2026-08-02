"""Phase 6: evaluation suite -- 3-condition harness, statistical
significance, and regression gates.

**What this proves right now, and what it doesn't.** Phase 6.1 calls for a
100 clean + 100 spontaneous + 100 noisy real-turn eval set; that data still
doesn't exist (same CANDOR/roleplay blocker as Phase 1.1/3.x -- see
STATUS.md). What CAN be built and validated now is the harness itself:
paired per-scenario latency extraction, bootstrap CI + Wilcoxon
signed-rank on the latency delta, false-interruption / over-wait rates, and
the 3 regression-gate asserts. This script dry-runs that harness against
the existing synthetic 30-scenario corpus under a single condition labeled
"synthetic_dry_run" -- explicitly NOT one of the plan's three real
conditions. Point --labels/--condition at real per-condition eval sets once
they exist; nothing else here needs to change.

Bootstrap CI and Wilcoxon both run on PAIRED per-scenario latencies (same
scenario_id, baseline vs. learned) -- only scenarios where both endpointers
actually fired are paired ("complete cases"); scenarios where either missed
are reported separately, not silently dropped from the summary.

Run:
    python scripts/eval.py
    python scripts/eval.py --update-reference   # accept this run's numbers as the new regression baseline
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import wilcoxon
from silero_vad import load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ab_compare_endpointers import load_model_from_checkpoint, run_learned_trace  # noqa: E402
from baseline_fixed_threshold_vad import load_scenarios, run_endpointer  # noqa: E402

SAMPLE_RATE = 16000
LEARNED_FRAME_SAMPLES = 1600  # 100ms @ 16kHz, matches Phase 3.2 FeaturePipeline

# The matched operating point compared: NOT just "a reasonable default" --
# matched by false_interruption_rate, per outputs/ab_comparison.csv (Phase
# 3.5's own A/B sweep). At learned threshold 0.7, false_interruption_rate is
# 0.4; the fixed-threshold baseline hits that same 0.4 rate at 800ms
# (200/300/500ms all interrupt far more often -- 0.8/0.667/0.6 -- so
# pairing against any of those would be comparing apples to oranges, not a
# stricter baseline). This reproduces the "614ms vs 802ms at 40% interrupt
# rate" comparison already reported in STATUS.md, now as a proper paired
# per-scenario stat test instead of two independent aggregates.
BASELINE_THRESHOLD_MS = 800
LEARNED_THRESHOLD = 0.7

ENDPOINTING_BUDGET_MS = 700  # README latency budget, upper bound -- over this counts as "over-wait"

DEFAULT_LABELS = ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "turn_taking_model.pt"
CSV_PATH = ROOT / "outputs" / "eval_summary.csv"
REFERENCE_PATH = ROOT / "outputs" / "eval_regression_reference.json"

P95_REGRESSION_MAX_PCT = 10.0
INTERRUPTION_RATE_REGRESSION_MAX_PP = 2.0  # percentage points, not relative percent


def percentile(vals: list, p: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)] if s else float("nan")


def baseline_per_scenario_latency(vad_model, scenarios: list, audio_cache: dict, threshold_ms: int) -> dict:
    """scenario_id -> response latency ms (TRUE_END only), or None if missed."""
    out = {}
    for s in scenarios:
        if s["label"] != "TRUE_END":
            continue
        events = run_endpointer(vad_model, audio_cache[s["scenario_id"]], threshold_ms)
        end_decisions = [d for d, event in events if "end" in event]
        splice = s["splice_sample"]
        fires_after = [d for d in end_decisions if d >= splice]
        out[s["scenario_id"]] = (fires_after[0] - splice) / SAMPLE_RATE * 1000 if fires_after else None
    return out


def learned_per_scenario_latency(traces: dict, scenarios: list, threshold: float) -> dict:
    out = {}
    for s in scenarios:
        if s["label"] != "TRUE_END":
            continue
        splice = s["splice_sample"]
        fired_sample = None
        for r in traces[s["scenario_id"]]:
            if r["pause_so_far_ms"] > 0 and r["prob"] >= threshold:
                fired_sample = r["frame_index"] * LEARNED_FRAME_SAMPLES
                break
        out[s["scenario_id"]] = (fired_sample - splice) / SAMPLE_RATE * 1000 if fired_sample is not None and fired_sample >= splice else None
    return out


def interruption_rate(scenarios: list, fired_lookup) -> float:
    n_mid_turn = 0
    n_interrupted = 0
    for s in scenarios:
        if s["label"] == "TRUE_END":
            continue
        n_mid_turn += 1
        if fired_lookup(s):
            n_interrupted += 1
    return n_interrupted / n_mid_turn if n_mid_turn else float("nan")


def bootstrap_ci(diffs: list, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI on the mean of `diffs` (learned - baseline,
    paired complete cases only)."""
    if not diffs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs_arr = np.asarray(diffs)
    boot_means = [rng.choice(diffs_arr, size=len(diffs_arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_condition(labels_path: Path, checkpoint_path: Path, condition: str,
                   baseline_threshold_ms: int = BASELINE_THRESHOLD_MS, learned_threshold: float = LEARNED_THRESHOLD) -> dict:
    """baseline_threshold_ms / learned_threshold default to the values tuned
    against the SYNTHETIC corpus (outputs/ab_comparison.csv) -- pass real,
    matched-by-interrupt-rate values for any other corpus. Reusing the
    synthetic defaults on a differently-distributed real corpus reproduces
    exactly the unmatched-operating-point bug this project already hit once
    (see git history): don't."""
    scenarios = load_scenarios(labels_path)
    audio_cache = {}
    for s in scenarios:
        audio, sr = sf.read(ROOT / s["audio_path"], dtype="float64")
        assert sr == SAMPLE_RATE, f"{s['audio_path']} is {sr}Hz, expected {SAMPLE_RATE}"
        audio_cache[s["scenario_id"]] = audio

    vad_model = load_silero_vad(onnx=False)
    model = load_model_from_checkpoint(checkpoint_path)

    from faster_whisper import WhisperModel

    # Hardcoded to CPU before -- on a T4 this was the actual cause of the paired-latency
    # computation (which needs a learned trace for every scenario, not just TRUE_END) being
    # too slow to complete in one Colab session. Use CUDA when available instead of ignoring it.
    asr_device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = WhisperModel("tiny", device=asr_device, compute_type="int8")
    print(f"[{condition}] trace-computation ASR device: {asr_device}")

    print(f"[{condition}] computing baseline latencies (threshold={baseline_threshold_ms}ms)...")
    baseline_lat = baseline_per_scenario_latency(vad_model, scenarios, audio_cache, baseline_threshold_ms)

    print(f"[{condition}] computing learned traces + latencies (threshold={learned_threshold})...")
    traces = {s["scenario_id"]: run_learned_trace(model, vad_model, asr_model, audio_cache[s["scenario_id"]])
              for s in scenarios if s["label"] == "TRUE_END"}
    learned_lat = learned_per_scenario_latency(traces, scenarios, learned_threshold)

    # Paired complete cases: both endpointers actually fired on this scenario.
    paired_ids = [sid for sid in baseline_lat if baseline_lat[sid] is not None and learned_lat.get(sid) is not None]
    baseline_paired = [baseline_lat[sid] for sid in paired_ids]
    learned_paired = [learned_lat[sid] for sid in paired_ids]
    diffs = [l - b for l, b in zip(learned_paired, baseline_paired)]

    n_baseline_missed = sum(1 for v in baseline_lat.values() if v is None)
    n_learned_missed = sum(1 for v in learned_lat.values() if v is None)

    ci_lo, ci_hi = bootstrap_ci(diffs)
    if len(diffs) >= 1 and any(d != 0 for d in diffs):
        wilcoxon_stat, wilcoxon_p = wilcoxon(learned_paired, baseline_paired)
    else:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    # Re-run traces for MID_TURN scenarios too, needed for interruption rate.
    mid_turn = [s for s in scenarios if s["label"] != "TRUE_END"]
    mid_turn_traces = {s["scenario_id"]: run_learned_trace(model, vad_model, asr_model, audio_cache[s["scenario_id"]]) for s in mid_turn}

    def baseline_fires_during_pause(s) -> bool:
        events = run_endpointer(vad_model, audio_cache[s["scenario_id"]], baseline_threshold_ms)
        end_decisions = [d for d, event in events if "end" in event]
        return any(s["splice_sample"] <= d < s["speech_b_start_sample"] for d in end_decisions)

    def learned_fires_during_pause(s) -> bool:
        for r in mid_turn_traces[s["scenario_id"]]:
            if r["pause_so_far_ms"] > 0 and r["prob"] >= learned_threshold:
                fired_sample = r["frame_index"] * LEARNED_FRAME_SAMPLES
                return s["splice_sample"] <= fired_sample < s["speech_b_start_sample"]
        return False

    baseline_interrupt_rate = interruption_rate(scenarios, baseline_fires_during_pause)
    learned_interrupt_rate = interruption_rate(scenarios, learned_fires_during_pause)

    baseline_overwait_rate = sum(1 for v in baseline_paired if v > ENDPOINTING_BUDGET_MS) / len(baseline_paired) if baseline_paired else float("nan")
    learned_overwait_rate = sum(1 for v in learned_paired if v > ENDPOINTING_BUDGET_MS) / len(learned_paired) if learned_paired else float("nan")

    return {
        "condition": condition,
        "n_scenarios": len(scenarios),
        "n_true_end": sum(1 for s in scenarios if s["label"] == "TRUE_END"),
        "n_paired_complete_cases": len(paired_ids),
        "n_baseline_missed": n_baseline_missed,
        "n_learned_missed": n_learned_missed,
        "baseline_latency_p50_ms": round(percentile(baseline_paired, 0.50), 1),
        "baseline_latency_p95_ms": round(percentile(baseline_paired, 0.95), 1),
        "learned_latency_p50_ms": round(percentile(learned_paired, 0.50), 1),
        "learned_latency_p95_ms": round(percentile(learned_paired, 0.95), 1),
        "latency_delta_mean_ms": round(float(np.mean(diffs)), 1) if diffs else float("nan"),
        "latency_delta_ci95_lo_ms": round(ci_lo, 1),
        "latency_delta_ci95_hi_ms": round(ci_hi, 1),
        "wilcoxon_statistic": round(float(wilcoxon_stat), 3) if wilcoxon_stat == wilcoxon_stat else float("nan"),
        "wilcoxon_p_value": round(float(wilcoxon_p), 5) if wilcoxon_p == wilcoxon_p else float("nan"),
        "baseline_false_interruption_rate": round(baseline_interrupt_rate, 4),
        "learned_false_interruption_rate": round(learned_interrupt_rate, 4),
        "baseline_over_wait_rate": round(baseline_overwait_rate, 4),
        "learned_over_wait_rate": round(learned_overwait_rate, 4),
    }


def regression_gate(current: dict, reference: dict) -> list:
    """Returns a list of (passed: bool, message: str) -- 3 asserts per
    Phase 6.4: p95 latency regression and interruption-rate regression,
    checked against a stored reference from a prior run."""
    checks = []

    p95_delta_pct = (current["learned_latency_p95_ms"] - reference["learned_latency_p95_ms"]) / reference["learned_latency_p95_ms"] * 100
    checks.append((
        p95_delta_pct <= P95_REGRESSION_MAX_PCT,
        f"learned p95 latency regression: {p95_delta_pct:+.1f}% vs reference "
        f"({reference['learned_latency_p95_ms']}ms -> {current['learned_latency_p95_ms']}ms), "
        f"gate is <= {P95_REGRESSION_MAX_PCT}%",
    ))

    interrupt_delta_pp = (current["learned_false_interruption_rate"] - reference["learned_false_interruption_rate"]) * 100
    checks.append((
        interrupt_delta_pp <= INTERRUPTION_RATE_REGRESSION_MAX_PP,
        f"learned false-interruption rate regression: {interrupt_delta_pp:+.1f}pp vs reference "
        f"({reference['learned_false_interruption_rate'] * 100:.1f}% -> {current['learned_false_interruption_rate'] * 100:.1f}%), "
        f"gate is <= {INTERRUPTION_RATE_REGRESSION_MAX_PP}pp",
    ))

    checks.append((
        current["n_paired_complete_cases"] > 0,
        f"at least one paired complete case exists ({current['n_paired_complete_cases']}) -- "
        "guards against a silently-empty eval set passing the other two gates vacuously",
    ))

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--condition", default="synthetic_dry_run", help="label for this eval set; use clean/spontaneous/noisy once real per-condition data exists")
    parser.add_argument("--baseline-threshold-ms", type=int, default=BASELINE_THRESHOLD_MS, help=f"defaults to {BASELINE_THRESHOLD_MS}ms, tuned for the SYNTHETIC corpus -- pass a value matched by interrupt rate on whatever corpus --labels points at instead")
    parser.add_argument("--learned-threshold", type=float, default=LEARNED_THRESHOLD, help=f"defaults to {LEARNED_THRESHOLD}, tuned for the SYNTHETIC corpus -- see --baseline-threshold-ms")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH, help="defaults to outputs/eval_summary.csv; pass a separate path (e.g. outputs/ab_comparison_real.csv) to avoid mixing corpora in one file")
    parser.add_argument("--reference-path", type=Path, default=REFERENCE_PATH, help="defaults to outputs/eval_regression_reference.json; pass a separate path per corpus -- comparing a real-corpus run against a synthetic-corpus reference (or vice versa) is not a regression check, it's noise")
    parser.add_argument("--update-reference", action="store_true", help="accept this run's numbers as the new regression-gate reference instead of checking against the existing one")
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"missing {args.labels} -- run scripts/prepare_synthetic_turn_taking_eval.py first")
    if not args.checkpoint.exists():
        raise SystemExit(f"missing {args.checkpoint} -- train a model first (see voice_ai_pipeline.ipynb)")

    result = run_condition(args.labels, args.checkpoint, args.condition, args.baseline_threshold_ms, args.learned_threshold)

    print(f"\n=== {args.condition} ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.csv_path.exists()
    with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(result)
    print(f"\nwrote {args.csv_path}")

    if args.update_reference or not args.reference_path.exists():
        args.reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.reference_path} (new regression-gate reference)" if args.update_reference else f"wrote {args.reference_path} (first run, establishing reference)")
        return

    reference = json.loads(args.reference_path.read_text(encoding="utf-8"))
    print(f"\n=== regression gates vs. {args.reference_path.name} ===")
    checks = regression_gate(result, reference)
    all_passed = True
    for passed, message in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {message}")
        all_passed = all_passed and passed

    if not all_passed:
        raise SystemExit(1)
    print("\nGATE PASS")


if __name__ == "__main__":
    main()
