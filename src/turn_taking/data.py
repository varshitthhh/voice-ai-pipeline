"""Phase 3.3/3.4 training data: scaled-up synthetic turn-taking scenarios
+ frame-level feature/label tensors.

Same caveat as every other turn-taking script so far: the real hand-labeled
CANDOR + roleplay corpus (Phase 1.1) is still pending on the user's side.
This generates a LARGER synthetic corpus using the same real-SLURP-clip-
splicing approach as scripts/prepare_synthetic_turn_taking_eval.py (Phase
3.1), scaled up for training rather than a small fixed baseline-eval set.
Swap in the real corpus by producing scenario dicts in the same shape
({"audio", "label", "splice_sample", "pause_end_sample"}) -- everything
downstream (featurization, model, training loop) is corpus-agnostic.

Label convention matches docs/turn_taking_label_schema.md: the label is
constant across every frame of a pause. Frames before the pause (the
speaker's active speech) are fed to the GRU as context but excluded from
the loss via `loss_mask` -- they aren't labeled decision points.
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps

from features import FeaturePipeline
from turn_taking.model import MAX_TOKENS_PER_FRAME, PAD_TOKEN_ID, hash_token

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1600  # 100ms @ 16kHz
TRUE_END_TRAILING_SILENCE_MS = 1500
MID_TURN_PAUSE_DURATIONS_MS = [120, 180, 250, 350, 450, 600, 700, 900, 1100, 1300]

# Fixed-range normalization (not learned standardization) -- simple and
# deterministic, adequate for this scope. pitch/pause_so_far are divided
# down to roughly [0,1]; energy is already ~[0,1] for float PCM.
PITCH_NORM_HZ = 400.0
PAUSE_NORM_MS = 1500.0
RATE_NORM_TPS = 10.0


def _load_trimmed(path: Path, vad_model) -> np.ndarray:
    """Same trailing-silence trim as Phase 3.1's prepare_synthetic_turn_taking_eval.py
    (see that script's load_clip() for why this matters for ground truth)."""
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{path} is {sr}Hz, expected {SAMPLE_RATE}"
    vad_model.reset_states()
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio), vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=100
    )
    if not timestamps:
        return audio
    return audio[: min(len(audio), timestamps[-1]["end"])]


def generate_scenarios(slurp_audio_dir: Path, n_scenarios: int, vad_model, seed: int = 0) -> list:
    clip_paths = sorted(Path(slurp_audio_dir).glob("*.flac"))
    if len(clip_paths) < 2:
        raise SystemExit(f"need SLURP clips in {slurp_audio_dir} -- run scripts/prepare_slurp_eval.py first")

    rng = random.Random(seed)
    scenarios = []
    for i in range(n_scenarios):
        clip_a = _load_trimmed(rng.choice(clip_paths), vad_model)
        splice_sample = len(clip_a)

        is_true_end = i % 2 == 0
        if is_true_end:
            pause_ms = TRUE_END_TRAILING_SILENCE_MS
            silence = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
            audio = np.concatenate([clip_a, silence])
            pause_end_sample = len(audio)
        else:
            pause_ms = MID_TURN_PAUSE_DURATIONS_MS[i % len(MID_TURN_PAUSE_DURATIONS_MS)]
            clip_b = _load_trimmed(rng.choice(clip_paths), vad_model)
            gap = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
            audio = np.concatenate([clip_a, gap, clip_b])
            pause_end_sample = splice_sample + len(gap)

        scenarios.append({
            "audio": audio,
            "label": 1 if is_true_end else 0,
            "splice_sample": splice_sample,
            "pause_end_sample": pause_end_sample,
        })

    return scenarios


def load_real_scenarios_jsonl(labels_path: Path, root: Path) -> list:
    """Adapts a real-data manifest (the EVAL schema written by
    prepare_ami_turntaking.py / prepare_synthetic_turn_taking_eval.py --
    on-disk audio_path, string label, speech_b_start_sample, one row per
    scenario) into the TRAINING schema generate_scenarios() produces
    in-memory (loaded audio array, int label, pause_end_sample) -- these are
    two different shapes in this codebase for two different consumers
    (eval scripts read the manifest directly; featurize_scenario() below
    expects the training shape), and this is the one place that converts
    between them, so the two don't silently drift out of sync.

    speech_b_start_sample -> pause_end_sample is a real semantic mapping,
    not just a rename: for MID_TURN they're the same sample (where clip_b's
    speech starts); for TRUE_END, speech_b_start_sample is None (nothing
    follows) but pause_end_sample must be the actual end of the rendered
    audio (the trailing silence goes all the way to the end of the file),
    which is only known once the audio is loaded.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    scenarios = []
    for row in rows:
        audio, sr = sf.read(root / row["audio_path"], dtype="float64")
        assert sr == SAMPLE_RATE, f"{row['audio_path']} is {sr}Hz, expected {SAMPLE_RATE}"

        if row["label"] == "TRUE_END":
            pause_end_sample = len(audio)
        elif row["label"] == "MID_TURN":
            pause_end_sample = row["speech_b_start_sample"]
        else:
            raise ValueError(f"unknown label {row['label']!r} in {row['scenario_id']} -- expected TRUE_END or MID_TURN")

        scenarios.append({
            "audio": audio,
            "label": 1 if row["label"] == "TRUE_END" else 0,
            "splice_sample": row["splice_sample"],
            "pause_end_sample": pause_end_sample,
        })

    return scenarios


def featurize_scenario(scenario: dict, vad_model, asr_model, asr_refresh_every_n_frames: int = 5) -> tuple:
    """Runs FeaturePipeline over one scenario's audio (streaming-safe by
    construction, per Phase 3.2). Returns (token_ids [T,20], prosody [T,4],
    loss_mask [T], label)."""
    pipeline = FeaturePipeline(vad_model, asr_model, asr_refresh_every_n_frames=asr_refresh_every_n_frames)
    audio = scenario["audio"]
    n_frames = len(audio) // FRAME_SAMPLES

    pause_start_frame = scenario["splice_sample"] // FRAME_SAMPLES
    pause_end_frame = scenario["pause_end_sample"] // FRAME_SAMPLES

    token_ids_seq, prosody_seq, mask_seq = [], [], []
    for i in range(n_frames):
        chunk = audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        feats = pipeline.step(chunk)

        ids = [hash_token(t) for t in feats.partial_tokens][:MAX_TOKENS_PER_FRAME]
        ids = ids + [PAD_TOKEN_ID] * (MAX_TOKENS_PER_FRAME - len(ids))
        token_ids_seq.append(ids)

        prosody_seq.append([
            feats.pitch_hz / PITCH_NORM_HZ,
            feats.energy_rms,
            feats.pause_so_far_ms / PAUSE_NORM_MS,
            min(feats.speech_rate_tps / RATE_NORM_TPS, 1.0),
        ])

        mask_seq.append(1.0 if pause_start_frame <= i < pause_end_frame else 0.0)

    return (
        torch.tensor(token_ids_seq, dtype=torch.long),
        torch.tensor(prosody_seq, dtype=torch.float32),
        torch.tensor(mask_seq, dtype=torch.float32),
        scenario["label"],
    )


def collate(batch: list) -> tuple:
    """Pads a list of (token_ids, prosody, mask, label) to the batch's max length."""
    max_len = max(t.shape[0] for t, _, _, _ in batch)
    B = len(batch)

    token_ids = torch.full((B, max_len, MAX_TOKENS_PER_FRAME), PAD_TOKEN_ID, dtype=torch.long)
    prosody = torch.zeros((B, max_len, 4), dtype=torch.float32)
    loss_mask = torch.zeros((B, max_len), dtype=torch.float32)
    targets = torch.zeros((B, max_len), dtype=torch.float32)

    for b, (t_ids, pros, mask, label) in enumerate(batch):
        T = t_ids.shape[0]
        token_ids[b, :T] = t_ids
        prosody[b, :T] = pros
        loss_mask[b, :T] = mask
        targets[b, :T] = float(label)  # constant target across the pause, per the label schema

    return token_ids, prosody, loss_mask, targets


def make_batches(examples: list, batch_size: int) -> list:
    return [collate(examples[i : i + batch_size]) for i in range(0, len(examples), batch_size)]
