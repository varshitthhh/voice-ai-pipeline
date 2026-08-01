"""Phase 3.5: A/B comparison -- fixed-threshold VAD baseline (Phase 3.1) vs.
the trained LearnedEndpointer (Phase 3.3/3.4), on the same eval scenarios.

TODO: the plan's Phase 6.1 three-condition eval set (100 clean scripted +
100 spontaneous + 100 noisy turns) doesn't exist yet. This runs on the same
synthetic splice corpus used throughout Phase 3 (real SLURP clips spliced
with controlled-duration silence gaps) -- rerun against the real Phase 6.1
set, and ultimately the real CANDOR + roleplay corpus, once they exist.
Nothing else here needs to change to do that.

The learned endpointer's probability trace is computed ONCE per scenario
(one LearnedEndpointer pass, ASR included) and every threshold in the sweep
is applied post-hoc to that stored trace -- avoids re-running ASR/GRU
inference once per threshold, which would otherwise multiply runtime by
len(LEARNED_THRESHOLDS).

Run:
    python scripts/ab_compare_endpointers.py
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_fixed_threshold_vad import evaluate_threshold, load_scenarios, percentile  # noqa: E402
from turn_taking import LearnedEndpointer, TurnTakingGRU  # noqa: E402

SAMPLE_RATE = 16000
LEARNED_FRAME_SAMPLES = 1600  # 100ms @ 16kHz, per Phase 3.2's FeaturePipeline (vs. the baseline's 512-sample VAD-native frames)
BASELINE_THRESHOLDS_MS = [200, 300, 500, 800]
LEARNED_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

DEFAULT_LABELS = ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "turn_taking_model.pt"
CSV_PATH = ROOT / "outputs" / "ab_comparison.csv"
PLOT_PATH = ROOT / "outputs" / "ab_comparison_tradeoff.png"


def run_learned_trace(model, vad_model, asr_model, audio) -> list:
    """One LearnedEndpointer pass, ignoring its internal threshold (set
    impossibly high so it never fires on its own) -- we read .prob directly
    off every frame instead, to apply the real threshold sweep post-hoc."""
    ep = LearnedEndpointer(model, vad_model, asr_model, threshold=2.0)
    n_frames = len(audio) // LEARNED_FRAME_SAMPLES
    return [ep.step(audio[i * LEARNED_FRAME_SAMPLES : (i + 1) * LEARNED_FRAME_SAMPLES]) for i in range(n_frames)]


def first_crossing_sample(trace: list, threshold: float):
    """First frame where prob crosses `threshold` AMONG FRAMES WHERE A PAUSE
    IS ACTUALLY ONGOING (pause_so_far_ms > 0). Restricting to pause frames
    matters: without it, a probability spike during active speech (ASR
    partial-token flicker can cause these) gets treated as a valid
    decision -- confirmed as a real bug via direct evaluation, where lower
    thresholds were "catching" pre-pause spikes and getting filtered out as
    invalid, while a higher threshold only caught later, legitimate
    crossings -- producing a nonsensical non-monotonic threshold sweep."""
    for r in trace:
        if r["pause_so_far_ms"] > 0 and r["prob"] >= threshold:
            return r["frame_index"] * LEARNED_FRAME_SAMPLES
    return None


def evaluate_learned(traces: dict, scenarios: list, threshold: float) -> dict:
    latencies_ms = []
    n_mid_turn = 0
    n_interrupted = 0
    n_true_end_missed = 0

    for scenario in scenarios:
        fired_sample = first_crossing_sample(traces[scenario["scenario_id"]], threshold)
        splice = scenario["splice_sample"]

        if scenario["label"] == "TRUE_END":
            if fired_sample is not None and fired_sample >= splice:
                latencies_ms.append((fired_sample - splice) / SAMPLE_RATE * 1000)
            else:
                # either never crossed threshold, or crossed it before the
                # pause even started (mid-speech) -- neither is a valid
                # post-pause decision, so both count as a miss
                n_true_end_missed += 1
        else:
            n_mid_turn += 1
            speech_b_start = scenario["speech_b_start_sample"]
            if fired_sample is not None and splice <= fired_sample < speech_b_start:
                n_interrupted += 1

    return {
        "threshold": threshold,
        "response_latency_p50_ms": round(percentile(latencies_ms, 0.50), 1),
        "response_latency_p95_ms": round(percentile(latencies_ms, 0.95), 1),
        "n_true_end_measured": len(latencies_ms),
        "n_true_end_missed": n_true_end_missed,
        "false_interruption_rate": round(n_interrupted / n_mid_turn, 4) if n_mid_turn else float("nan"),
        "n_mid_turn": n_mid_turn,
        "n_interrupted": n_interrupted,
    }


def load_model_from_checkpoint(checkpoint_path: Path) -> TurnTakingGRU:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden_dim = state["gru.weight_hh_l0"].shape[1]
    embed_dim = state["token_embedding.weight"].shape[1]
    input_dim = state["gru.weight_ih_l0"].shape[1]
    feature_mode = "both" if input_dim == embed_dim + 4 else ("text" if input_dim == embed_dim else "prosody")

    model = TurnTakingGRU(hidden_dim=hidden_dim, embed_dim=embed_dim, feature_mode=feature_mode)
    model.load_state_dict(state)
    print(f"loaded checkpoint: hidden_dim={hidden_dim} embed_dim={embed_dim} feature_mode={feature_mode!r}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"missing {args.labels} -- run scripts/prepare_synthetic_turn_taking_eval.py first")
    if not args.checkpoint.exists():
        raise SystemExit(f"missing {args.checkpoint} -- train a model first (see voice_ai_pipeline.ipynb)")

    scenarios = load_scenarios(args.labels)
    audio_cache = {}
    for s in scenarios:
        audio, sr = sf.read(ROOT / s["audio_path"], dtype="float64")
        assert sr == SAMPLE_RATE, f"{s['audio_path']} is {sr}Hz, expected {SAMPLE_RATE}"
        audio_cache[s["scenario_id"]] = audio
    print(f"loaded {len(scenarios)} scenarios")

    vad_model = load_silero_vad(onnx=False)

    print("\n--- fixed-threshold baseline (Phase 3.1) ---")
    baseline_rows = []
    for threshold_ms in BASELINE_THRESHOLDS_MS:
        row = evaluate_threshold(vad_model, scenarios, audio_cache, threshold_ms)
        row["endpointer"] = "fixed_threshold"
        row["threshold_label"] = f"{threshold_ms}ms"
        baseline_rows.append(row)
        print(f"  {threshold_ms}ms: latency_p50={row['response_latency_p50_ms']}ms interrupt={row['false_interruption_rate'] * 100:.1f}%")

    print("\n--- learned endpointer (Phase 3.3/3.4) ---")
    model = load_model_from_checkpoint(args.checkpoint)
    asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("computing probability traces (one pass per scenario)...")
    traces = {}
    for i, s in enumerate(scenarios):
        traces[s["scenario_id"]] = run_learned_trace(model, vad_model, asr_model, audio_cache[s["scenario_id"]])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(scenarios)}")

    learned_rows = []
    for threshold in LEARNED_THRESHOLDS:
        row = evaluate_learned(traces, scenarios, threshold)
        row["endpointer"] = "learned"
        row["threshold_label"] = f"p>={threshold}"
        learned_rows.append(row)
        print(f"  threshold={threshold}: latency_p50={row['response_latency_p50_ms']}ms interrupt={row['false_interruption_rate'] * 100:.1f}%")

    all_rows = baseline_rows + learned_rows
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "endpointer", "threshold_label", "response_latency_p50_ms", "response_latency_p95_ms",
        "false_interruption_rate", "n_true_end_measured", "n_true_end_missed", "n_mid_turn", "n_interrupted",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(7, 5))
    bx = [r["false_interruption_rate"] * 100 for r in baseline_rows]
    by = [r["response_latency_p50_ms"] for r in baseline_rows]
    ax.plot(bx, by, "o-", color="#C44E52", label="fixed-threshold VAD (Phase 3.1)")
    for r, x, y in zip(baseline_rows, bx, by):
        ax.annotate(r["threshold_label"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    lx = [r["false_interruption_rate"] * 100 for r in learned_rows]
    ly = [r["response_latency_p50_ms"] for r in learned_rows]
    ax.plot(lx, ly, "o-", color="#4C72B0", label="learned endpointer (Phase 3.3/3.4)")
    for r, x, y in zip(learned_rows, lx, ly):
        ax.annotate(r["threshold_label"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("false interruption rate (%)")
    ax.set_ylabel("response latency, p50 (ms)")
    ax.set_title("Phase 3.5 A/B: learned endpointer vs. fixed-threshold baseline\n(synthetic corpus -- rerun on the real Phase 6.1 eval set later)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
