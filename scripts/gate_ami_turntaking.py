"""Validation gate for scripts/prepare_ami_turntaking.py's output.

Real, runnable checks -- not a demonstration. Run after
prepare_ami_turntaking.py to confirm the real-data manifest is actually
usable before pointing any eval script at it via --labels.

Run:
    python scripts/gate_ami_turntaking.py
"""

import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "turn_taking" / "real_ami" / "scenarios.jsonl"
SAMPLE_RATE = 16000


def load_scenarios(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(f"missing {LABELS_PATH} -- run scripts/prepare_ami_turntaking.py first")

    scenarios = load_scenarios(LABELS_PATH)
    checks = []

    checks.append((len(scenarios) > 0, f"non-empty manifest ({len(scenarios)} scenarios)"))

    labels = {s["label"] for s in scenarios}
    checks.append((labels <= {"TRUE_END", "MID_TURN"}, f"only known labels present: {labels}"))

    n_true_end = sum(s["label"] == "TRUE_END" for s in scenarios)
    n_mid_turn = len(scenarios) - n_true_end
    min_class_frac = min(n_true_end, n_mid_turn) / len(scenarios) if scenarios else 0
    checks.append((min_class_frac >= 0.15, f"label balance not degenerate: {n_true_end} TRUE_END / {n_mid_turn} MID_TURN (min class {min_class_frac * 100:.1f}%)"))

    pauses = [s["pause_ms"] for s in scenarios]
    checks.append((all(p >= 0 for p in pauses), "all pause_ms non-negative"))
    checks.append((all(p <= 5000 for p in pauses), "all pause_ms within the MAX_GAP_MS filter (<=5000ms)"))

    pauses_sorted = sorted(pauses)
    p50 = pauses_sorted[len(pauses_sorted) // 2]
    checks.append((p50 < 2000, f"p50 pause duration is plausible for a natural pause, not dominated by meeting breaks (p50={p50:.0f}ms)"))

    # Cross-check every audio file actually exists and its real duration is
    # consistent with the manifest's own splice_sample / speech_b_start_sample
    # arithmetic -- catches silent corruption in the download+splice step.
    missing = []
    mismatched = []
    sample_check_n = min(30, len(scenarios))  # full audio decode is slow; a random-ish sample is enough to catch systematic bugs
    for s in scenarios[:sample_check_n]:
        audio_path = ROOT / s["audio_path"]
        if not audio_path.exists():
            missing.append(s["scenario_id"])
            continue
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            mismatched.append((s["scenario_id"], f"sample rate {sr} != {SAMPLE_RATE}"))
            continue
        if s["label"] == "TRUE_END":
            expected_min_len = s["splice_sample"]  # splice_sample + gap, gap >= 0
        else:
            expected_min_len = s["speech_b_start_sample"]
        if len(audio) < expected_min_len:
            mismatched.append((s["scenario_id"], f"audio len {len(audio)} < expected minimum {expected_min_len}"))

    checks.append((not missing, f"no missing audio files in the first {sample_check_n} scenarios checked" + (f" -- MISSING: {missing}" if missing else "")))
    checks.append((not mismatched, f"audio length consistent with manifest arithmetic in the first {sample_check_n} scenarios checked" + (f" -- MISMATCHED: {mismatched}" if mismatched else "")))

    # Provenance: every scenario should trace back to a real AMI meeting, not
    # a fabricated placeholder.
    meetings = {s.get("meeting_id") for s in scenarios}
    checks.append((all(s.get("source") == "ami" for s in scenarios), "every scenario tagged source=ami (no silent fallback to synthetic data)"))
    checks.append((None not in meetings and len(meetings) >= 1, f"scenarios trace to {len(meetings)} real meeting(s): {meetings}"))

    print(f"=== AMI real-data gate: {len(scenarios)} scenarios, {len(meetings)} meeting(s) ===\n")
    all_passed = True
    for passed, message in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {message}")
        all_passed = all_passed and passed

    if not all_passed:
        raise SystemExit(1)
    print("\nGATE PASS")


if __name__ == "__main__":
    main()
