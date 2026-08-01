"""Phase 3.1 eval data: synthetic turn-taking scenarios.

Stand-in for the real corpus (CANDOR + self-recorded roleplays, Phase 1.1)
which is still pending -- hand-labeling real conversational pauses was
explicitly deferred to the user earlier in this project. This generates
fully-controlled scenarios instead: real SLURP speech clips spliced with
silence gaps of known, exact duration, so ground truth is exact by
construction rather than hand-labeled.

Two scenario types, matching docs/turn_taking_label_schema.md's
TURN_COMPLETE / MID_TURN_PAUSE labels:
    TRUE_END:  speech_A + pause + long trailing silence, nothing follows.
    MID_TURN:  speech_A + pause_of_duration_P + speech_B (speaker resumes)
               -- the hard negative a fixed-threshold endpointer cuts off.

Pause durations for MID_TURN scenarios are spread deliberately across and
between the four thresholds scripts/baseline_fixed_threshold_vad.py sweeps
(200/300/500/800ms) so the resulting tradeoff curve isn't degenerate.

IMPORTANT: this is not the real corpus. When the real hand-labeled CANDOR +
roleplay data exists, point baseline_fixed_threshold_vad.py at it via
--labels instead -- the evaluation script doesn't care which corpus
produced its input, by design.

Run:
    python scripts/prepare_synthetic_turn_taking_eval.py
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
SLURP_AUDIO_DIR = ROOT / "data" / "intent" / "raw" / "slurp_eval" / "audio"
OUT_AUDIO_DIR = ROOT / "data" / "turn_taking" / "synthetic" / "audio"
OUT_LABELS_PATH = ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl"

SAMPLE_RATE = 16000
N_SCENARIOS = 30
TRUE_END_TRAILING_SILENCE_MS = 1500
MID_TURN_PAUSE_DURATIONS_MS = [120, 180, 250, 350, 450, 600, 700, 900, 1100, 1300]
RANDOM_SEED = 0


def silence(duration_ms: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_ms / 1000), dtype=np.float64)


def load_clip(path: Path, vad_model) -> np.ndarray:
    """Loads a raw SLURP clip and trims it to its actual speech content.

    Raw SLURP clips carry their own trailing near-silence (confirmed by
    inspecting frame energies directly: several hundred ms of near-noise-
    floor audio after the real speech ends, in every clip checked). Using
    the raw file length as the ground-truth speech-offset ("splice_sample")
    would be wrong by that same amount -- the fixed-threshold VAD baseline
    would then correctly detect the *real* offset and get scored as if it
    fired early or not at all.

    No padding is added after the trim point: `splice_sample` in the
    scenario record is `len(clip_a)`, so the trimmed clip's boundary must
    exactly equal the VAD-detected speech end, not that plus a cosmetic
    pad -- otherwise the baseline's own 'end' event (measured from this
    same clip) lands just before the padded boundary and gets scored as a
    miss. (This is exactly the bug the first version of this function had.)
    """
    audio, sr = sf.read(path, dtype="float64")
    assert sr == SAMPLE_RATE, f"{path} is {sr}Hz, expected {SAMPLE_RATE}"

    vad_model.reset_states()
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio).float(), vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=100
    )
    if not timestamps:
        return audio  # no speech detected at all; leave as-is rather than guess

    last_speech_end = timestamps[-1]["end"]
    return audio[: min(len(audio), last_speech_end)]


def main() -> None:
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clip_paths = sorted(SLURP_AUDIO_DIR.glob("*.flac"))
    if len(clip_paths) < N_SCENARIOS * 2:
        raise SystemExit(
            f"need at least {N_SCENARIOS * 2} SLURP clips, found {len(clip_paths)} "
            "-- run scripts/prepare_slurp_eval.py first"
        )

    rng = random.Random(RANDOM_SEED)
    shuffled = clip_paths.copy()
    rng.shuffle(shuffled)

    vad_model = load_silero_vad(onnx=False)

    scenarios = []
    for i in range(N_SCENARIOS):
        clip_a_path = shuffled[2 * i]
        clip_a = load_clip(clip_a_path, vad_model)
        splice_sample = len(clip_a)

        is_true_end = i % 2 == 0
        if is_true_end:
            pause_ms = TRUE_END_TRAILING_SILENCE_MS
            audio = np.concatenate([clip_a, silence(pause_ms)])
            speech_b_start_sample = None
            clip_b_path = None
        else:
            pause_ms = MID_TURN_PAUSE_DURATIONS_MS[i % len(MID_TURN_PAUSE_DURATIONS_MS)]
            clip_b_path = shuffled[2 * i + 1]
            clip_b = load_clip(clip_b_path, vad_model)
            gap = silence(pause_ms)
            audio = np.concatenate([clip_a, gap, clip_b])
            speech_b_start_sample = splice_sample + len(gap)

        scenario_id = f"synth_{i:03d}"
        audio_path = OUT_AUDIO_DIR / f"{scenario_id}.wav"
        sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_16")

        scenarios.append({
            "scenario_id": scenario_id,
            "label": "TRUE_END" if is_true_end else "MID_TURN",
            "pause_ms": pause_ms,
            "splice_sample": splice_sample,
            "speech_b_start_sample": speech_b_start_sample,
            "audio_path": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
            "clip_a": clip_a_path.stem,
            "clip_b": clip_b_path.stem if clip_b_path else None,
        })

    with open(OUT_LABELS_PATH, "w", encoding="utf-8") as f:
        for row in scenarios:
            f.write(json.dumps(row) + "\n")

    n_true_end = sum(s["label"] == "TRUE_END" for s in scenarios)
    print(f"wrote {len(scenarios)} scenarios ({n_true_end} TRUE_END, {len(scenarios) - n_true_end} MID_TURN)")
    print(f"audio -> {OUT_AUDIO_DIR}")
    print(f"labels -> {OUT_LABELS_PATH}")


if __name__ == "__main__":
    main()
