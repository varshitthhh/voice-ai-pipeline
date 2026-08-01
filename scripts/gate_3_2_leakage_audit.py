"""Phase 3.2 gate: explicit future-context leakage audit for FeaturePipeline.

The pipeline's step() API only ever accepts one new chunk at a time, which
structurally prevents it from touching future audio -- but a structural
argument isn't proof against implementation bugs (an off-by-one in a
buffer index could still leak a sample or two). This is the actual
verification:

    1. Run the pipeline once over the FULL audio -> features_full.
    2. For several earlier cut points, run a FRESH pipeline instance over
       ONLY audio[:cut] -> features_truncated.
    3. Assert every frame before the cut is bit-identical between the two
       runs. If frame i's features depend on anything after the cut, this
       is exactly the comparison that would catch it: features_full[i] was
       computed with future audio available in the buffer; features_truncated[i]
       was computed with it physically absent. Any dependence on the future
       shows up as a mismatch.

Run:
    python scripts/gate_3_2_leakage_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from features import run_pipeline

TEST_AUDIO = ROOT / "data" / "turn_taking" / "synthetic" / "audio" / "synth_001.wav"
CUT_FRAMES = [8, 15, 22, 30]  # frame indices (100ms each) to truncate at
FIELDS_TO_COMPARE = ["pitch_hz", "energy_rms", "pause_so_far_ms", "partial_tokens", "speech_rate_tps"]


def features_equal(a, b) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in FIELDS_TO_COMPARE)


def main() -> None:
    if not TEST_AUDIO.exists():
        raise SystemExit(f"missing {TEST_AUDIO} -- run scripts/prepare_synthetic_turn_taking_eval.py first")

    audio, sr = sf.read(TEST_AUDIO, dtype="float32")
    assert sr == 16000
    print(f"test audio: {TEST_AUDIO.name}, {len(audio)} samples ({len(audio) / sr:.2f}s)")

    vad_model = load_silero_vad(onnx=False)
    asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("\nrunning full-audio pass...")
    features_full = run_pipeline(vad_model, asr_model, audio)
    print(f"{len(features_full)} frames computed")

    all_ok = True
    for cut_frame in CUT_FRAMES:
        cut_sample = cut_frame * 1600  # frame_samples for 100ms @ 16kHz
        if cut_sample >= len(audio):
            continue

        print(f"\nrunning truncated pass, cut at frame {cut_frame} ({cut_sample} samples)...")
        features_truncated = run_pipeline(vad_model, asr_model, audio[:cut_sample])

        n_compare = min(len(features_truncated), cut_frame)
        mismatches = []
        for i in range(n_compare):
            if not features_equal(features_full[i], features_truncated[i]):
                mismatches.append(i)

        status = "PASS" if not mismatches else "FAIL"
        print(f"[{status}] cut@frame{cut_frame}: {n_compare} frames compared, {len(mismatches)} mismatches")
        if mismatches:
            all_ok = False
            for i in mismatches[:3]:
                print(f"    frame {i}: full={features_full[i]}  truncated={features_truncated[i]}")

    print("\nGATE PASS: no frame's features changed when future audio was removed" if all_ok else "\nGATE FAIL: leakage detected, see mismatches above")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
