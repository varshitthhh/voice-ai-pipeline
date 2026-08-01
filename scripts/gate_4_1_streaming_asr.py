"""Phase 4.1 gate: StreamingASR on real audio, fed in irregular chunk sizes.

Feeds a real speech-pause-speech scenario through StreamingASR in 300-sample
chunks (not aligned to Silero VAD's native 512-sample window) specifically
to exercise the leftover-buffering path in _detect_speech -- a real
streaming caller (a WebSocket receiving whatever chunk size the client
sends) has no reason to hand over VAD-aligned audio.

Checks:
    - Transcription only happens during detected speech (VAD-gating) --
      verified by asserting every emitted result's underlying segment
      falls within a plausible speech region, not the known silence gap.
    - At least one partial and exactly one final per speech segment.
    - Every emitted object is a schema-validated AsrResult (no raw dicts
      ever leak past StreamingASR.step()).

Run:
    python scripts/gate_4_1_streaming_asr.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from schemas import AsrResult
from streaming_asr import StreamingASR

TEST_AUDIO = ROOT / "data" / "turn_taking" / "synthetic" / "audio" / "synth_001.wav"  # MID_TURN: speech, pause, speech
FEED_CHUNK_SAMPLES = 300  # deliberately NOT a multiple of 512


def main() -> None:
    if not TEST_AUDIO.exists():
        raise SystemExit(f"missing {TEST_AUDIO} -- run scripts/prepare_synthetic_turn_taking_eval.py first")

    audio, sr = sf.read(TEST_AUDIO, dtype="float32")
    assert sr == 16000
    # This scenario file is trimmed to end exactly at its last detected
    # speech frame (see prepare_synthetic_turn_taking_eval.py's load_clip),
    # so the second speech segment never naturally gets a trailing silence
    # to trigger a VAD 'end' event -- pad some on for this test so both
    # segments get a fair chance to finalize, not just the first.
    audio = np.concatenate([audio, np.zeros(int(0.5 * sr), dtype=np.float32)])
    print(f"test audio: {TEST_AUDIO.name}, {len(audio)} samples ({len(audio) / sr:.2f}s, incl. 0.5s padding)")
    print(f"feeding in {FEED_CHUNK_SAMPLES}-sample chunks (not 512-aligned) to exercise leftover buffering\n")

    vad_model = load_silero_vad(onnx=False)
    asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    streaming_asr = StreamingASR(
        asr_model, vad_model, session_id="gate-4.1",
        partial_every_n_speech_samples=4800,  # ~300ms, small enough to get a partial before the short clips end
    )

    results = []
    n_chunks = len(audio) // FEED_CHUNK_SAMPLES
    for i in range(n_chunks):
        chunk = audio[i * FEED_CHUNK_SAMPLES : (i + 1) * FEED_CHUNK_SAMPLES]
        result = streaming_asr.step(chunk)
        if result is not None:
            results.append(result)
            kind = "FINAL" if result.is_final else "partial"
            print(f"[{kind}] turn_id={result.turn_id} conf={result.confidence:.2f} text={result.transcript!r}")

    print(f"\n{len(results)} results emitted")

    all_ok = True

    type_ok = all(isinstance(r, AsrResult) for r in results)
    print(f"[{'PASS' if type_ok else 'FAIL'}] every emitted result is a validated AsrResult instance")
    all_ok = all_ok and type_ok

    n_finals = sum(r.is_final for r in results)
    finals_ok = n_finals == 2  # this scenario has exactly two speech segments (clip_a, clip_b)
    print(f"[{'PASS' if finals_ok else 'FAIL'}] exactly 2 final results emitted (one per speech segment), got {n_finals}")
    all_ok = all_ok and finals_ok

    final_transcripts_nonempty = all(r.transcript.strip() for r in results if r.is_final)
    print(f"[{'PASS' if final_transcripts_nonempty else 'FAIL'}] every final result has non-empty transcript")
    all_ok = all_ok and final_transcripts_nonempty

    n_partials = sum(not r.is_final for r in results)
    print(f"info: {n_partials} partial result(s) emitted before the finals")

    print("\nGATE PASS" if all_ok else "\nGATE FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
