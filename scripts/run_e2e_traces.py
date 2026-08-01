"""Phase 4.4: generate 100 turns of real end-to-end trace data.

Wires together VAD + fixed-threshold endpointer (Phase 3.1) + ASR (Phase
4.1's faster-whisper model) + a TODO-marked mock LLM stage (GPU-only) +
real Piper TTS (Phase 2.3), writing one Phase 0.2 TurnTrace per turn to
traces/e2e_phase4_4.jsonl.

User audio for each turn is a real SLURP clip (trimmed to its detected
speech end, same methodology as scripts/prepare_synthetic_turn_taking_eval.py)
plus 1500ms of trailing silence, guaranteeing the fixed 500ms endpointer
threshold always has enough silence to fire on every turn.

Run:
    python scripts/run_e2e_traces.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from piper import PiperVoice
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2e_pipeline import run_turn_trace
from tracing import Tracer

SAMPLE_RATE = 16000
N_TURNS = 100
TRAILING_SILENCE_MS = 1500
SESSION_ID = "e2e_phase4_4"
RANDOM_SEED = 0

SLURP_AUDIO_DIR = ROOT / "data" / "intent" / "raw" / "slurp_eval" / "audio"
PIPER_MODEL_PATH = ROOT / "data" / "tts" / "models" / "piper" / "en_US-lessac-medium.onnx"
TRACE_DIR = ROOT / "traces"

# A few varied first-sentence lengths so TTS timing isn't a single repeated
# value across all 100 turns -- real responses vary.
RESPONSE_FIRST_SENTENCES = [
    "I'm sorry to hear that.",
    "Let me look into your order right away and see what's going on with the shipment.",
    "I can help with that.",
    "Thanks for your patience while I check on this for you.",
]


def load_trimmed(path: Path, vad_model) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE
    vad_model.reset_states()
    timestamps = get_speech_timestamps(torch.from_numpy(audio), vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=100)
    if not timestamps:
        return audio
    return audio[: min(len(audio), timestamps[-1]["end"])]


def build_turn_audio(clip: np.ndarray) -> np.ndarray:
    silence = np.zeros(int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000), dtype=np.float32)
    return np.concatenate([clip, silence])


def main() -> None:
    if not PIPER_MODEL_PATH.exists():
        raise SystemExit(f"missing {PIPER_MODEL_PATH} -- run scripts/prepare_tts_models.py first")

    clip_paths = sorted(SLURP_AUDIO_DIR.glob("*.flac"))
    if len(clip_paths) < N_TURNS:
        raise SystemExit(f"need >= {N_TURNS} SLURP clips, found {len(clip_paths)} -- run scripts/prepare_slurp_eval.py first")

    rng = random.Random(RANDOM_SEED)
    selected_clips = rng.sample(clip_paths, N_TURNS)

    vad_model = load_silero_vad(onnx=False)
    asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    list(voice.synthesize("warm up."))  # warmup, excluded from timing

    def tts_synthesize_fn(text: str) -> None:
        list(voice.synthesize(text))  # consume the generator -- this is the real synth cost

    tracer = Tracer(session_id=SESSION_ID, trace_dir=TRACE_DIR)

    n_ok = 0
    n_skipped = 0
    for i, clip_path in enumerate(selected_clips):
        clip = load_trimmed(clip_path, vad_model)
        turn_audio = build_turn_audio(clip)
        response = RESPONSE_FIRST_SENTENCES[i % len(RESPONSE_FIRST_SENTENCES)]

        deltas = run_turn_trace(
            turn_id=f"turn-{i:03d}",
            user_audio=turn_audio,
            response_first_sentence=response,
            vad_model=vad_model,
            asr_model=asr_model,
            tts_synthesize_fn=tts_synthesize_fn,
            tracer=tracer,
            rng=rng,
        )

        if deltas is None:
            n_skipped += 1
            print(f"  turn {i:03d}: SKIPPED (no endpoint found for {clip_path.name})")
            continue

        n_ok += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TURNS} turns done (e2e_ms={deltas['e2e_ms']:.1f} last)")

    print(f"\nwrote {n_ok} turns ({n_skipped} skipped) -> {tracer.path}")


if __name__ == "__main__":
    main()
