"""Phase 3.1: fixed-threshold VAD baseline -- the tradeoff curve the
learned turn-taking model (Phase 3.3+) must beat.

Uses Silero VAD v5's own VADIterator with min_silence_duration_ms swept
across {200, 300, 500, 800}ms -- that parameter *is* the fixed-threshold
endpointer, so no custom threshold logic is hand-rolled here; this is
exactly how a fixed-threshold endpointer built directly on Silero VAD v5
would behave in production.

Two metrics per threshold, matching the project's central latency/
interruption tradeoff:
    response latency        -- ms from true speech offset to the
                                endpointer's 'end' event, on TRUE_END
                                scenarios.
    false interruption rate -- fraction of MID_TURN scenarios where the
                                endpointer fires before the speaker resumes.

Defaults to the synthetic scenario corpus (scripts/
prepare_synthetic_turn_taking_eval.py) since the real hand-labeled CANDOR +
roleplay corpus (Phase 1.1) is still pending -- point --labels at the real
corpus once it exists; nothing else here needs to change.

Run:
    python scripts/baseline_fixed_threshold_vad.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from silero_vad import VADIterator, load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl"
CSV_PATH = ROOT / "outputs" / "fixed_threshold_vad_baseline.csv"
PLOT_PATH = ROOT / "outputs" / "fixed_threshold_vad_tradeoff.png"

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # required by Silero VAD v5 at 16kHz
THRESHOLDS_MS = [200, 300, 500, 800]
VAD_SPEECH_PROB_THRESHOLD = 0.5


def load_scenarios(labels_path: Path) -> list:
    with open(labels_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_endpointer(model, audio: np.ndarray, min_silence_duration_ms: int) -> list:
    """Feeds `audio` through a fresh VADIterator frame by frame (each call
    resets the model's internal state, so scenarios never leak state into
    each other). Returns a list of (decision_sample, event) pairs.

    decision_sample is the processing position -- (i+1)*FRAME_SAMPLES --
    at the moment VADIterator returned the event, i.e. when the endpointer
    actually made its decision in the audio timeline. This is NOT the same
    as event['end'], which VADIterator deliberately back-dates to when the
    silence acoustically began (temp_end + padding), not when the
    min_silence_duration_ms wait was satisfied and the decision was made.
    Using event['end'] for latency gives ~0ms regardless of threshold --
    confirmed by direct debugging before writing this -- since it points
    at the acoustic offset, not the endpointer's real decision time."""
    vad = VADIterator(
        model,
        threshold=VAD_SPEECH_PROB_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    events = []
    n_frames = len(audio) // FRAME_SAMPLES
    for i in range(n_frames):
        chunk = audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        event = vad(chunk, return_seconds=False)
        if event is not None:
            events.append(((i + 1) * FRAME_SAMPLES, event))
    return events


def percentile(vals: list, p: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)] if s else float("nan")


def evaluate_threshold(model, scenarios: list, audio_cache: dict, threshold_ms: int) -> dict:
    latencies_ms = []
    n_mid_turn = 0
    n_interrupted = 0
    n_true_end_missed = 0

    for scenario in scenarios:
        audio = audio_cache[scenario["scenario_id"]]
        events = run_endpointer(model, audio, threshold_ms)
        # decision_sample: when the endpointer actually fired, in the audio timeline
        end_decisions = [decision_sample for decision_sample, event in events if "end" in event]

        if scenario["label"] == "TRUE_END":
            splice = scenario["splice_sample"]
            fires_after_splice = [d for d in end_decisions if d >= splice]
            if fires_after_splice:
                latencies_ms.append((fires_after_splice[0] - splice) / SAMPLE_RATE * 1000)
            else:
                n_true_end_missed += 1  # VAD never detected the end at all -- worth surfacing, not hiding
        else:
            n_mid_turn += 1
            splice = scenario["splice_sample"]
            speech_b_start = scenario["speech_b_start_sample"]
            if any(splice <= d < speech_b_start for d in end_decisions):
                n_interrupted += 1

    return {
        "threshold_ms": threshold_ms,
        "response_latency_p50_ms": round(percentile(latencies_ms, 0.50), 1),
        "response_latency_p95_ms": round(percentile(latencies_ms, 0.95), 1),
        "n_true_end_measured": len(latencies_ms),
        "n_true_end_missed": n_true_end_missed,
        "false_interruption_rate": round(n_interrupted / n_mid_turn, 4) if n_mid_turn else float("nan"),
        "n_mid_turn": n_mid_turn,
        "n_interrupted": n_interrupted,
    }


def plot_tradeoff(rows: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["false_interruption_rate"] * 100 for r in rows]
    ys = [r["response_latency_p50_ms"] for r in rows]
    ax.plot(xs, ys, "o-", color="#4C72B0")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"{r['threshold_ms']}ms", (x, y), textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("false interruption rate (%)")
    ax.set_ylabel("response latency, p50 (ms)")
    ax.set_title(
        "Fixed-threshold VAD baseline: latency vs. false-interruption tradeoff\n"
        "(this is what the learned turn-taking model must beat)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--thresholds-ms", nargs="+", type=int, default=THRESHOLDS_MS)
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"missing {args.labels} -- run scripts/prepare_synthetic_turn_taking_eval.py first")

    scenarios = load_scenarios(args.labels)
    audio_cache = {}
    for s in scenarios:
        audio, sr = sf.read(ROOT / s["audio_path"], dtype="float64")
        assert sr == SAMPLE_RATE, f"{s['audio_path']} is {sr}Hz, expected {SAMPLE_RATE}"
        audio_cache[s["scenario_id"]] = audio

    model = load_silero_vad(onnx=False)

    rows = []
    for threshold_ms in args.thresholds_ms:
        row = evaluate_threshold(model, scenarios, audio_cache, threshold_ms)
        rows.append(row)
        print(
            f"threshold={threshold_ms}ms: latency_p50={row['response_latency_p50_ms']}ms "
            f"false_interrupt_rate={row['false_interruption_rate'] * 100:.1f}% "
            f"({row['n_interrupted']}/{row['n_mid_turn']})"
            + (f"  [{row['n_true_end_missed']} TRUE_END never detected]" if row["n_true_end_missed"] else "")
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")

    plot_tradeoff(rows, PLOT_PATH)


if __name__ == "__main__":
    main()
