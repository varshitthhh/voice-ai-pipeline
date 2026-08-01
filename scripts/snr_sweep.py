"""Phase 1.3 (light): additive MUSAN-noise SNR sweep.

Scope, by explicit instruction: additive noise mixing at controlled SNR
only. No resampling (clean speech and MUSAN noise are both already 16kHz
mono, confirmed via soundfile.info before writing this), no mu-law, no
frame-drop simulation — those are out of scope for this pass.

Mixes each of a handful of clean speech clips (drawn from the SLURP eval
sample fetched in Phase 1.2 — a real speech stand-in; the project has no
dedicated clean-speech corpus yet) against a MUSAN noise clip at a swept
set of target SNR levels, using RMS energy scaling. Reports achieved vs.
target SNR and flags clipping, since a mixer that silently clips or drifts
off its target SNR would poison any WER-vs-SNR curve built on top of it
later in Phase 2.

Run:
    python scripts/snr_sweep.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NOISE_DIR = ROOT / "data" / "noise" / "raw" / "musan_noise"
CLEAN_DIR = ROOT / "data" / "intent" / "raw" / "slurp_eval" / "audio"
OUT_DIR = ROOT / "data" / "noise" / "mixed"

SNR_LEVELS_DB = [20, 10, 5, 0, -5]
N_CLEAN_CLIPS = 5
CLIP_PEAK_HEADROOM = 0.99  # target ceiling if a mix would otherwise clip
RANDOM_SEED = 0


def load_mono(path: Path) -> tuple:
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def fit_noise_to_length(noise: np.ndarray, target_len: int, offset_seed: int) -> np.ndarray:
    """Trim or tile `noise` to exactly `target_len` samples (no resampling)."""
    if len(noise) >= target_len:
        rng = np.random.default_rng(offset_seed)
        start = rng.integers(0, len(noise) - target_len + 1)
        return noise[start : start + target_len]
    reps = int(np.ceil(target_len / len(noise)))
    return np.tile(noise, reps)[:target_len]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> dict:
    p_clean = float(np.mean(clean**2))
    p_noise = float(np.mean(noise**2))
    if p_noise == 0:
        raise ValueError("noise clip is silent, cannot scale to a target SNR")

    scale = np.sqrt(p_clean / (p_noise * 10 ** (target_snr_db / 10)))
    mixed = clean + scale * noise

    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    clipped = peak > 1.0
    headroom_scale = (CLIP_PEAK_HEADROOM / peak) if clipped else 1.0
    mixed = mixed * headroom_scale  # uniform gain preserves the signal/noise ratio

    p_noise_scaled = p_noise * (scale * headroom_scale) ** 2
    p_clean_scaled = p_clean * headroom_scale**2
    achieved_snr_db = 10 * np.log10(p_clean_scaled / p_noise_scaled)

    return {
        "mixed": mixed,
        "achieved_snr_db": round(float(achieved_snr_db), 3),
        "clipped_before_headroom_fix": clipped,
        "peak_before_headroom_fix": round(peak, 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    noise_paths = sorted(NOISE_DIR.glob("*.wav"))
    clean_paths = sorted(CLEAN_DIR.glob("*.flac"))[:N_CLEAN_CLIPS]
    if not noise_paths or not clean_paths:
        raise SystemExit("missing noise clips or clean clips — run prepare_musan_noise.py / prepare_slurp_eval.py first")

    manifest = []
    for i, clean_path in enumerate(clean_paths):
        clean, clean_sr = load_mono(clean_path)
        noise_path = noise_paths[i % len(noise_paths)]
        noise, noise_sr = load_mono(noise_path)

        assert clean_sr == noise_sr, (
            f"sample rate mismatch {clean_sr} vs {noise_sr} for {clean_path.name}/{noise_path.name} "
            "— resampling is out of scope for this pass, so mismatched pairs are a hard error, not silently handled"
        )

        noise_fit = fit_noise_to_length(noise, len(clean), offset_seed=RANDOM_SEED + i)

        for target_db in SNR_LEVELS_DB:
            result = mix_at_snr(clean, noise_fit, target_db)

            out_name = f"{clean_path.stem}__{noise_path.stem}__snr{target_db}dB.wav"
            out_path = OUT_DIR / out_name
            sf.write(out_path, result["mixed"], clean_sr, subtype="PCM_16")

            manifest.append({
                "clean_id": clean_path.stem,
                "noise_id": noise_path.stem,
                "sample_rate": clean_sr,
                "target_snr_db": target_db,
                "achieved_snr_db": result["achieved_snr_db"],
                "clipped_before_headroom_fix": result["clipped_before_headroom_fix"],
                "peak_before_headroom_fix": result["peak_before_headroom_fix"],
                "mixed_path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            })

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    csv_path = OUT_DIR / "snr_sweep_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    max_drift = max(abs(r["target_snr_db"] - r["achieved_snr_db"]) for r in manifest)
    n_clipped = sum(r["clipped_before_headroom_fix"] for r in manifest)
    print(f"wrote {len(manifest)} mixed clips -> {OUT_DIR}")
    print(f"manifest -> {manifest_path}")
    print(f"report -> {csv_path}")
    print(f"max |target - achieved| SNR drift: {max_drift:.4f} dB")
    print(f"clips that hit clipping headroom fix: {n_clipped}/{len(manifest)}")


if __name__ == "__main__":
    main()
