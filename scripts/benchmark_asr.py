"""Phase 2.1 ASR benchmark: N model sizes x M quantizations x clean/noisy.

Production comparison matrix (matches the README's stated expectation that
distil-large-v3 is ~6x faster than large-v3 at ~1% WER cost):
    model sizes:   distil-large-v3, large-v3, small
    quantizations: int8, float16

TODO(gpu-required): float16 needs CUDA -- CTranslate2 does not run float16
on CPU (see `_resolve_compute_type` below) -- and large-v3 / distil-large-v3
CPU inference is impractically slow for a benchmark sweep (each is a
multi-hundred-MB to multi-GB download, and large-v3 alone can take tens of
seconds per short clip on CPU). The full production matrix above must run
on the 24GB GPU; this script is written generically so the identical
command works there unchanged.

Locally (CPU-only laptop), this is smoke-tested against a lighter
substitute matrix (tiny/base, int8 only) on real audio: the SLURP clean
clips (Phase 1.2) paired with their MUSAN SNR-mixed noisy counterparts
(Phase 1.3) -- proving the WER / RTF / p50-p95 harness is correct before
it is ever pointed at the real models.

Run (defaults to the CPU smoke matrix):
    python scripts/benchmark_asr.py

Run the real matrix on GPU hardware:
    python scripts/benchmark_asr.py --model-sizes distil-large-v3 large-v3 small --compute-types int8 float16
"""

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
SLURP_JSONL = ROOT / "data" / "intent" / "processed" / "slurp_eval.jsonl"
NOISE_MANIFEST = ROOT / "data" / "noise" / "mixed" / "manifest.jsonl"
CSV_PATH = ROOT / "outputs" / "asr_benchmark.csv"

DEFAULT_MODEL_SIZES = ["tiny", "base"]  # CPU-feasible smoke matrix; see module docstring for the real one
DEFAULT_COMPUTE_TYPES = ["int8"]


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_clean_and_noisy_clips(n_clean: int) -> tuple:
    """Pairs of (clip_id, audio_path, reference_text) for clean and noisy conditions.

    Noisy clips are the Phase 1.3 MUSAN SNR mixes of the same `n_clean`
    SLURP clips used for clean, so the two conditions are a matched
    comparison rather than different audio.
    """
    if not SLURP_JSONL.exists() or not NOISE_MANIFEST.exists():
        raise SystemExit(
            "missing data -- run scripts/prepare_slurp_eval.py and scripts/snr_sweep.py first"
        )

    text_by_id = {}
    with open(SLURP_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text_by_id[row["id"]] = row["text"]

    with open(NOISE_MANIFEST, "r", encoding="utf-8") as f:
        noise_rows = [json.loads(line) for line in f]

    clean_ids = list(dict.fromkeys(r["clean_id"] for r in noise_rows))[:n_clean]

    clean_clips = [
        (cid, ROOT / "data" / "intent" / "raw" / "slurp_eval" / "audio" / f"{cid}.flac", text_by_id[cid])
        for cid in clean_ids
    ]
    noisy_clips = [
        (f"{r['clean_id']}__{r['noise_id']}__snr{r['target_snr_db']}dB", ROOT / r["mixed_path"], text_by_id[r["clean_id"]])
        for r in noise_rows
        if r["clean_id"] in clean_ids
    ]
    return clean_clips, noisy_clips


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(requested: str, device: str) -> tuple:
    """Returns (usable_compute_type, todo_note). TODO(gpu-required): CTranslate2
    only runs float16 on CUDA; on CPU it raises, so we skip rather than crash.
    """
    if requested == "float16" and device == "cpu":
        return None, "float16 requires CUDA; skipped on CPU (TODO: run on 24GB GPU / Colab T4)"
    return requested, None


def percentile(values: list, p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def transcribe_clip(model: WhisperModel, audio_path: Path) -> tuple:
    """Returns (hypothesis_text, latency_ms, audio_duration_s)."""
    duration_s = sf.info(audio_path).duration
    t0 = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), beam_size=1)
    hypothesis = " ".join(seg.text for seg in segments).strip()  # generator is lazy; consuming it is the actual decode
    latency_ms = (time.perf_counter() - t0) * 1000
    return hypothesis, latency_ms, duration_s


def benchmark_combo(model_size: str, compute_type: str, device: str, clean_clips: list, noisy_clips: list) -> list:
    print(f"\n--- model={model_size} compute_type={compute_type} device={device} ---")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # TODO(gpu-validation): VRAM read path is unexercised on this CPU-only
    # laptop -- confirm units/timing of peak-allocation reads on real CUDA.
    vram_total_gb = None
    vram_used_gb = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = round(props.total_memory / 1e9, 2)

    rows = []
    for condition, clips in [("clean", clean_clips), ("noisy", noisy_clips)]:
        latencies_ms, rtfs, refs, hyps = [], [], [], []
        for clip_id, audio_path, reference in clips:
            hypothesis, latency_ms, duration_s = transcribe_clip(model, audio_path)
            latencies_ms.append(latency_ms)
            rtfs.append((latency_ms / 1000) / duration_s if duration_s > 0 else float("nan"))
            refs.append(normalize_text(reference))
            hyps.append(normalize_text(hypothesis) or " ")  # jiwer chokes on a fully empty hypothesis string

        wer = jiwer.wer(refs, hyps) if refs else float("nan")

        if device == "cuda":
            vram_used_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)  # TODO(gpu-validation)

        row = {
            "model_size": model_size,
            "compute_type": compute_type,
            "device": device,
            "condition": condition,
            "n_clips": len(clips),
            "wer": round(wer, 4),
            "rtf_p50": round(percentile(rtfs, 0.50), 4),
            "rtf_p95": round(percentile(rtfs, 0.95), 4),
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 2),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 2),
            "vram_total_gb": vram_total_gb,
            "vram_used_gb": vram_used_gb,
        }
        print(f"  {condition}: WER={row['wer']} RTF_p50={row['rtf_p50']} latency_p50={row['latency_p50_ms']}ms")
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-sizes", nargs="+", default=DEFAULT_MODEL_SIZES)
    parser.add_argument("--compute-types", nargs="+", default=DEFAULT_COMPUTE_TYPES)
    parser.add_argument("--n-clean-clips", type=int, default=5)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH, help="defaults to outputs/asr_benchmark.csv; pass a separate path (e.g. outputs/asr_benchmark_gpu.csv) to avoid mixing CPU-smoke and real-GPU rows in one file")
    args = parser.parse_args()

    device = resolve_device()
    clean_clips, noisy_clips = load_clean_and_noisy_clips(args.n_clean_clips)
    print(f"device={device}, {len(clean_clips)} clean clips, {len(noisy_clips)} noisy clips")

    all_rows = []
    skipped = []
    for model_size in args.model_sizes:
        for requested_ct in args.compute_types:
            compute_type, todo_note = resolve_compute_type(requested_ct, device)
            if compute_type is None:
                print(f"\n--- model={model_size} compute_type={requested_ct} SKIPPED: {todo_note} ---")
                skipped.append((model_size, requested_ct, todo_note))
                continue
            all_rows.extend(benchmark_combo(model_size, compute_type, device, clean_clips, noisy_clips))
            # Defensive cleanup between the 6 sequential model loads in the
            # real GPU matrix (distil-large-v3/large-v3/small x int8/float16)
            # -- CTranslate2 models aren't tracked by torch's allocator, so
            # this is insurance against VRAM fragmentation/buildup on a
            # memory-constrained card (e.g. a 16GB T4), not a fix for a
            # confirmed leak.
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.csv_path.exists()
    if all_rows:
        with open(args.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            if is_new:
                writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nappended {len(all_rows)} rows -> {args.csv_path}")

    if skipped:
        print("\nskipped combos (need GPU):")
        for model_size, ct, note in skipped:
            print(f"  {model_size} / {ct}: {note}")


if __name__ == "__main__":
    main()
