"""Phase 1.3 gate: feed deliberately corrupt audio (and malformed stage
payloads) through the validation layer and confirm every case comes back
as a structured rejection report — never an unhandled exception.

Run:
    python scripts/gate_1_3_corrupt_audio.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemas import AsrResult, AudioFrame, EndpointDecision, check_audio, validate_payload

SAMPLE_RATE = 16000


def clean_signal(duration_s: float = 1.0, freq_hz: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


AUDIO_CASES = []


def case(name: str, samples, sample_rate: int, expect_valid: bool, expect_failed_check: str = None):
    AUDIO_CASES.append((name, samples, sample_rate, expect_valid, expect_failed_check))


case("clean_1s_tone", clean_signal(1.0), SAMPLE_RATE, expect_valid=True)

case("wrong_sample_rate_8k", clean_signal(1.0), 8000, expect_valid=False, expect_failed_check="sample_rate")

_clipped = clean_signal(1.0, amplitude=0.3)
_clipped[: len(_clipped) // 2] = 1.0  # hard-clip half the clip at full scale
case("heavily_clipped", _clipped, SAMPLE_RATE, expect_valid=False, expect_failed_check="clipping")

case("dc_offset", clean_signal(1.0) + 0.3, SAMPLE_RATE, expect_valid=False, expect_failed_check="dc_offset")

case("too_short_10ms", clean_signal(0.01), SAMPLE_RATE, expect_valid=False, expect_failed_check="duration_bounds")

case("too_long_60s", clean_signal(60.0), SAMPLE_RATE, expect_valid=False, expect_failed_check="duration_bounds")

_nan = clean_signal(1.0)
_nan[100:110] = np.nan
case("contains_nan", _nan, SAMPLE_RATE, expect_valid=False, expect_failed_check="finite_values")

_inf = clean_signal(1.0)
_inf[200] = np.inf
case("contains_inf", _inf, SAMPLE_RATE, expect_valid=False, expect_failed_check="finite_values")

case("empty_array", np.array([]), SAMPLE_RATE, expect_valid=False, expect_failed_check="finite_values")

case("non_numeric_garbage", ["not", "audio", "data"], SAMPLE_RATE, expect_valid=False, expect_failed_check="array_shape")

# np.asarray(None, dtype=float64) coerces to nan rather than raising, so this
# is correctly caught by finite_values, not array_shape -- still a clean
# non-crashing rejection, just via a different check than truly non-numeric input.
case("none_input", None, SAMPLE_RATE, expect_valid=False, expect_failed_check="finite_values")

_stereo = np.stack([clean_signal(1.0), clean_signal(1.0)], axis=1)  # (N, 2) — should flatten, not crash
case("unexpected_stereo_shape", _stereo, SAMPLE_RATE, expect_valid=True)


def run_audio_gate() -> bool:
    print("=== audio sanity checks ===")
    all_ok = True
    for name, samples, sample_rate, expect_valid, expect_failed_check in AUDIO_CASES:
        report = check_audio(samples, sample_rate)  # must never raise
        ok = report.valid == expect_valid
        if not expect_valid and expect_failed_check:
            failed_names = {c.name for c in report.failed_checks}
            ok = ok and expect_failed_check in failed_names

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: valid={report.valid} -> {report.summary}")
        all_ok = all_ok and ok
    return all_ok


def run_stage_payload_gate() -> bool:
    print("\n=== stage-boundary payload validation ===")
    all_ok = True

    good_frame = {
        "session_id": "s1", "frame_index": 0, "sample_rate": SAMPLE_RATE,
        "samples": [0.1, 0.2, 0.1], "t_captured": time.monotonic(),
    }
    bad_frame = dict(good_frame, sample_rate=44100)  # wrong sample rate
    bad_frame_types = {"session_id": "", "frame_index": -1, "sample_rate": "sixteen-k", "samples": [], "t_captured": "now"}

    bad_endpoint = {
        "session_id": "s1", "turn_id": "t1", "turn_complete": True, "confidence": 0.9,
        "segment_start_frame": 50, "segment_end_frame": 10,  # end before start
        "t_decision": time.monotonic(),
    }

    bad_asr = {
        "session_id": "s1", "turn_id": "t1", "transcript": "   ", "is_final": True,
        "confidence": 1.5,  # out of [0, 1]
        "language": "en", "t_result": time.monotonic(),
    }

    payload_cases = [
        ("AudioFrame", AudioFrame, good_frame, True),
        ("AudioFrame (wrong sample_rate)", AudioFrame, bad_frame, False),
        ("AudioFrame (wrong types)", AudioFrame, bad_frame_types, False),
        ("EndpointDecision (end<start)", EndpointDecision, bad_endpoint, False),
        ("AsrResult (empty final transcript + bad confidence)", AsrResult, bad_asr, False),
        ("AsrResult (not even a dict)", AsrResult, "definitely not a payload", False),
    ]

    for label, model_cls, data, expect_valid in payload_cases:
        instance, report = validate_payload(model_cls, data)  # must never raise
        ok = report.valid == expect_valid and (instance is not None) == expect_valid
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {report.summary}")
        all_ok = all_ok and ok

    return all_ok


def main() -> None:
    audio_ok = run_audio_gate()
    payload_ok = run_stage_payload_gate()

    print()
    if audio_ok and payload_ok:
        print("GATE PASS: every corrupt case produced a structured rejection, nothing crashed")
    else:
        print("GATE FAIL: see FAIL rows above")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
