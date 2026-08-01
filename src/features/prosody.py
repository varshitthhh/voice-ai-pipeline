"""Causal, single-chunk prosody features: pitch and energy.

Both functions take exactly one audio chunk and nothing else -- there is no
buffer, no lookahead, no state. A function that literally cannot see
anything but its argument cannot leak future context, by construction.
"""

import numpy as np

F0_MIN_HZ = 80
F0_MAX_HZ = 400
SILENCE_RMS_THRESHOLD = 0.01


def rms_energy(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def estimate_pitch(chunk: np.ndarray, sample_rate: int) -> float:
    """Autocorrelation-based F0 estimate over `chunk` alone, in Hz.
    Returns 0.0 for silence/unvoiced chunks (no reliable pitch)."""
    chunk = chunk.astype(np.float64)
    if rms_energy(chunk) < SILENCE_RMS_THRESHOLD:
        return 0.0

    chunk = chunk - np.mean(chunk)
    autocorr = np.correlate(chunk, chunk, mode="full")[len(chunk) - 1 :]
    if autocorr[0] <= 0:
        return 0.0

    lag_min = int(sample_rate / F0_MAX_HZ)
    lag_max = min(int(sample_rate / F0_MIN_HZ), len(autocorr) - 1)
    if lag_max <= lag_min:
        return 0.0

    segment = autocorr[lag_min : lag_max + 1]
    peak_lag = lag_min + int(np.argmax(segment))
    if peak_lag == 0:
        return 0.0
    return sample_rate / peak_lag
