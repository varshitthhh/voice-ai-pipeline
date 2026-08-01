"""Audio sanity checks: sample rate, clipping, DC offset, duration bounds.

These run on raw samples before audio is trusted to cross a stage boundary
(AudioFrame in, TtsAudioChunk out). check_audio() never raises — malformed
input (NaN/Inf, empty arrays, non-numeric data, wrong shape) is itself one
of the things being checked for, not a reason to crash the caller.
"""

from typing import Optional

import numpy as np
from pydantic import BaseModel

EXPECTED_SAMPLE_RATE = 16000
MIN_DURATION_S = 0.05
MAX_DURATION_S = 30.0
CLIP_ABS_THRESHOLD = 0.999      # sample magnitude counted as "at the rail"
MAX_CLIPPED_FRACTION = 0.001    # >0.1% of samples at the rail -> reject
MAX_DC_OFFSET = 0.02            # |mean(samples)| above this -> reject


class AudioCheckResult(BaseModel):
    name: str
    passed: bool
    message: str
    value: Optional[float] = None


class AudioCheckReport(BaseModel):
    valid: bool
    checks: list[AudioCheckResult]

    @property
    def failed_checks(self) -> list[AudioCheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def summary(self) -> str:
        if self.valid:
            return "valid"
        return "REJECTED: " + "; ".join(f"{c.name} ({c.message})" for c in self.failed_checks)


def _check_finite(samples: np.ndarray) -> AudioCheckResult:
    if samples.size == 0:
        return AudioCheckResult(name="finite_values", passed=False, message="audio is empty")
    n_bad = int(np.sum(~np.isfinite(samples)))
    if n_bad > 0:
        return AudioCheckResult(
            name="finite_values",
            passed=False,
            message=f"{n_bad}/{samples.size} samples are NaN or Inf",
            value=float(n_bad),
        )
    return AudioCheckResult(name="finite_values", passed=True, message="ok")


def _check_sample_rate(sample_rate: int) -> AudioCheckResult:
    if sample_rate != EXPECTED_SAMPLE_RATE:
        return AudioCheckResult(
            name="sample_rate",
            passed=False,
            message=f"got {sample_rate}Hz, expected {EXPECTED_SAMPLE_RATE}Hz",
            value=float(sample_rate),
        )
    return AudioCheckResult(name="sample_rate", passed=True, message="ok", value=float(sample_rate))


def _check_duration(samples: np.ndarray, sample_rate: int) -> AudioCheckResult:
    duration_s = samples.size / sample_rate if sample_rate > 0 else float("inf")
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        return AudioCheckResult(
            name="duration_bounds",
            passed=False,
            message=f"{duration_s:.3f}s outside [{MIN_DURATION_S}, {MAX_DURATION_S}]s",
            value=duration_s,
        )
    return AudioCheckResult(name="duration_bounds", passed=True, message="ok", value=duration_s)


def _check_clipping(samples: np.ndarray) -> AudioCheckResult:
    clipped_fraction = float(np.mean(np.abs(samples) >= CLIP_ABS_THRESHOLD))
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        return AudioCheckResult(
            name="clipping",
            passed=False,
            message=f"{clipped_fraction:.4%} of samples at the rail (max {MAX_CLIPPED_FRACTION:.4%})",
            value=clipped_fraction,
        )
    return AudioCheckResult(name="clipping", passed=True, message="ok", value=clipped_fraction)


def _check_dc_offset(samples: np.ndarray) -> AudioCheckResult:
    dc = float(np.mean(samples))
    if abs(dc) > MAX_DC_OFFSET:
        return AudioCheckResult(
            name="dc_offset",
            passed=False,
            message=f"mean offset {dc:.4f} exceeds +/-{MAX_DC_OFFSET}",
            value=dc,
        )
    return AudioCheckResult(name="dc_offset", passed=True, message="ok", value=dc)


def check_audio(samples, sample_rate: int) -> AudioCheckReport:
    """Run every sanity check and return a structured report. Never raises."""
    try:
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - any coercion failure is itself a rejection reason
        return AudioCheckReport(
            valid=False,
            checks=[
                AudioCheckResult(
                    name="array_shape",
                    passed=False,
                    message=f"could not coerce input to a 1-D float array: {exc}",
                )
            ],
        )

    finite_check = _check_finite(arr)
    checks = [finite_check]

    if finite_check.passed:
        # Numeric checks below assume finite values; skip them rather than
        # let NaN/Inf silently propagate into mean()/comparisons.
        checks.append(_check_sample_rate(sample_rate))
        checks.append(_check_duration(arr, sample_rate))
        checks.append(_check_clipping(arr))
        checks.append(_check_dc_offset(arr))

    return AudioCheckReport(valid=all(c.passed for c in checks), checks=checks)
