"""Phase 2.3 TTS benchmark: Kokoro vs XTTS v2 vs Piper.

Metric is time-to-first-chunk (TTFC), not total generation time, per the
project's Phase 4.2 design: TTS is triggered at the first LLM sentence
boundary, not after the full response is ready. Both engines are fed one
sentence at a time to match that real invocation pattern -- Kokoro's
internal phoneme-batcher doesn't split on sentence boundaries the way
Piper's does natively (verified by reading kokoro_onnx's create_stream
source and by direct testing: a 3-sentence, ~170-char string came back as
a single ~15s chunk from Kokoro), so per-sentence calls are what make the
two comparable at all.

Engines:
    Kokoro (kokoro-onnx, int8) -- runs and is timed locally on CPU; auto-
                                   detects and uses a GPU if the
                                   `onnxruntime-gpu` package is installed
                                   (see Kokoro's own __init__, which checks
                                   `importlib.util.find_spec("onnxruntime-gpu")`)
                                   -- no code change needed here for that.
    Piper  (piper-tts)             -- runs and is timed locally; pass
                                       --use-cuda to run on GPU instead of CPU.
    XTTS v2 (coqui-tts)             -- TODO(gpu-required), see below.

TODO(gpu-required): XTTS v2 is ~2.1GB (checked via the HF API before
deciding), CPML-licensed (non-commercial), and its autoregressive GPT2-style
decoder is known to be impractically slow for a benchmark sweep on CPU --
so unlike Kokoro/Piper it was never downloaded here, and `_synth_xtts` below
is written against coqui-tts's real streaming API from documentation, never
imported or executed against an actual checkpoint. Verify the API surface
(`XttsConfig`, `Xtts.load_checkpoint`, `Xtts.inference_stream`) against
whatever coqui-tts version lands on the GPU box before trusting this.

Run (Kokoro + Piper, real, local):
    python scripts/benchmark_tts.py

Run including the (GPU-only, currently unimplemented-for-real) XTTS path:
    python scripts/benchmark_tts.py --engines kokoro piper xtts
"""

import argparse
import asyncio
import csv
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "tts" / "models"
CSV_PATH = ROOT / "outputs" / "tts_benchmark.csv"

# A realistic multi-sentence customer-support response -- exactly the shape
# an LLM would stream out, sentence by sentence, in the real pipeline.
TEST_TEXT = (
    "I'm sorry to hear your order hasn't arrived yet. "
    "Let me look into that for you right away. "
    "I can see it's currently in transit and should arrive within two days."
)
WARMUP_SENTENCE = "Warm up the model please."
N_REPEATS = 3


def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def percentile(values: list, p: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


# ---------------------------------------------------------------------------
# Kokoro (real, runs locally)
# ---------------------------------------------------------------------------

async def _synth_kokoro_one(kokoro, text: str) -> dict:
    t0 = time.perf_counter()
    ttfc_s = None
    n_chunks = 0
    n_samples = 0
    sample_rate = None
    async for samples, sr in kokoro.create_stream(text, voice="af_heart"):
        if ttfc_s is None:
            ttfc_s = time.perf_counter() - t0
        n_chunks += 1
        n_samples += len(samples)
        sample_rate = sr
    total_s = time.perf_counter() - t0
    return {"ttfc_s": ttfc_s, "total_s": total_s, "n_chunks": n_chunks, "audio_duration_s": n_samples / sample_rate}


async def bench_kokoro(sentences: list) -> list:
    from kokoro_onnx import Kokoro  # local import: only required if this engine is actually run

    kokoro = Kokoro(str(MODELS_DIR / "kokoro" / "kokoro-v1.0.int8.onnx"), str(MODELS_DIR / "kokoro" / "voices-v1.0.bin"))
    # Reports which onnxruntime execution provider actually got used --
    # CUDAExecutionProvider only if onnxruntime-gpu is installed AND a GPU
    # is visible; falls back to CPUExecutionProvider silently otherwise, so
    # this is worth recording rather than assuming from intent.
    device = "cuda" if "CUDAExecutionProvider" in kokoro.sess.get_providers() else "cpu"
    await _synth_kokoro_one(kokoro, WARMUP_SENTENCE)

    rows = []
    for idx, sentence in enumerate(sentences):
        ttfcs, totals = [], []
        for _ in range(N_REPEATS):
            result = await _synth_kokoro_one(kokoro, sentence)
            ttfcs.append(result["ttfc_s"])
            totals.append(result["total_s"])
        rows.append({
            "engine": "kokoro", "device": device, "sentence_index": idx, "sentence_chars": len(sentence),
            "ttfc_p50_ms": round(percentile(ttfcs, 0.50) * 1000, 1),
            "ttfc_p95_ms": round(percentile(ttfcs, 0.95) * 1000, 1),
            "total_p50_ms": round(percentile(totals, 0.50) * 1000, 1),
            "audio_duration_s": round(result["audio_duration_s"], 3),
        })
        print(f"kokoro[{idx}] ({device}): TTFC_p50={rows[-1]['ttfc_p50_ms']}ms  {sentence!r}")
    return rows


# ---------------------------------------------------------------------------
# Piper (real, runs locally)
# ---------------------------------------------------------------------------

def _synth_piper_one(voice, text: str) -> dict:
    t0 = time.perf_counter()
    ttfc_s = None
    n_chunks = 0
    n_samples = 0
    sample_rate = None
    for chunk in voice.synthesize(text):
        if ttfc_s is None:
            ttfc_s = time.perf_counter() - t0
        n_chunks += 1
        n_samples += len(chunk.audio_float_array)
        sample_rate = chunk.sample_rate
    total_s = time.perf_counter() - t0
    return {"ttfc_s": ttfc_s, "total_s": total_s, "n_chunks": n_chunks, "audio_duration_s": n_samples / sample_rate}


def bench_piper(sentences: list, use_cuda: bool = False) -> list:
    from piper import PiperVoice  # local import: only required if this engine is actually run

    voice = PiperVoice.load(str(MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"), use_cuda=use_cuda)
    # Real bug found and fixed here: this used to trust the use_cuda REQUEST rather than
    # checking what actually loaded, same class of bug Kokoro's code (above) already guards
    # against. Confirmed on a real Colab T4 run: PiperVoice.load(use_cuda=True) only requests
    # CUDAExecutionProvider (see piper's own source -- no CPU entry in that providers list),
    # but onnxruntime logged "Failed to create CUDAExecutionProvider... Require cuDNN 9.*
    # and CUDA 13.*" and silently still produced a working session -- the committed "cuda"
    # numbers for that run were actually CPU, mislabeled. Reading the real provider list is
    # the only way to know, not the flag that was passed in.
    actual_providers = voice.session.get_providers()
    device = "cuda" if "CUDAExecutionProvider" in actual_providers else "cpu"
    if use_cuda and device == "cpu":
        print(f"WARNING: --use-cuda was requested but CUDAExecutionProvider is not active "
              f"(actual providers: {actual_providers}) -- reporting device=cpu, not the requested flag")
    _synth_piper_one(voice, WARMUP_SENTENCE)

    rows = []
    for idx, sentence in enumerate(sentences):
        ttfcs, totals = [], []
        for _ in range(N_REPEATS):
            result = _synth_piper_one(voice, sentence)
            ttfcs.append(result["ttfc_s"])
            totals.append(result["total_s"])
        rows.append({
            "engine": "piper", "device": device, "sentence_index": idx, "sentence_chars": len(sentence),
            "ttfc_p50_ms": round(percentile(ttfcs, 0.50) * 1000, 1),
            "ttfc_p95_ms": round(percentile(ttfcs, 0.95) * 1000, 1),
            "total_p50_ms": round(percentile(totals, 0.50) * 1000, 1),
            "audio_duration_s": round(result["audio_duration_s"], 3),
        })
        print(f"piper[{idx}] ({device}): TTFC_p50={rows[-1]['ttfc_p50_ms']}ms  {sentence!r}")
    return rows


# ---------------------------------------------------------------------------
# XTTS v2 -- TODO(gpu-required), written but never executed here
# ---------------------------------------------------------------------------

def bench_xtts(sentences: list) -> list:
    """TODO(gpu-required): unexecuted. Sketch of the real coqui-tts
    streaming path -- confirm against the installed version on the GPU box:

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(MODELS_DIR / "xtts" / "config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(MODELS_DIR / "xtts"), use_deepspeed=False)
        model.cuda()

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(MODELS_DIR / "xtts" / "reference_voice.wav")]
        )

        for sentence in sentences:
            t0 = time.perf_counter()
            ttfc_s = None
            chunks = model.inference_stream(sentence, "en", gpt_cond_latent, speaker_embedding)
            for i, chunk in enumerate(chunks):
                if ttfc_s is None:
                    ttfc_s = time.perf_counter() - t0
            # ... aggregate same as Kokoro/Piper above

    Left unimplemented for real execution: needs the 2.1GB checkpoint
    (not downloaded, see module docstring) plus a reference speaker wav
    and CUDA to run at a speed worth benchmarking.
    """
    raise NotImplementedError(
        "XTTS v2 requires the 24GB GPU: 2.1GB CPML-licensed checkpoint not fetched, "
        "and CPU inference is impractically slow for a benchmark sweep. "
        "See this function's docstring for the real API sketch."
    )


ENGINE_RUNNERS = {"kokoro": bench_kokoro, "piper": bench_piper, "xtts": bench_xtts}


def write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


async def main_async(engines: list, use_cuda: bool, csv_path: Path) -> None:
    sentences = split_sentences(TEST_TEXT)
    print(f"{len(sentences)} test sentences")

    all_rows = []
    for engine in engines:
        if engine == "xtts":
            print("\n--- xtts SKIPPED: TODO(gpu-required), see benchmark_xtts() docstring ---")
            continue
        print(f"\n--- {engine} ---")
        if engine == "piper":
            result = bench_piper(sentences, use_cuda=use_cuda)
        else:
            result = bench_kokoro(sentences)  # GPU use is auto-detected via onnxruntime-gpu, no arg needed
        if asyncio.iscoroutine(result):
            result = await result
        all_rows.extend(result)

    write_csv(all_rows, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", default=["kokoro", "piper"], choices=list(ENGINE_RUNNERS.keys()))
    parser.add_argument("--use-cuda", action="store_true", help="run Piper on GPU (Kokoro auto-detects via onnxruntime-gpu)")
    parser.add_argument("--csv-path", type=Path, default=None, help="defaults to outputs/tts_benchmark_gpu.csv if --use-cuda else outputs/tts_benchmark.csv (schemas differ by a 'device' column, kept separate on purpose)")
    args = parser.parse_args()

    csv_path = args.csv_path or (ROOT / "outputs" / ("tts_benchmark_gpu.csv" if args.use_cuda else "tts_benchmark.csv"))
    asyncio.run(main_async(args.engines, args.use_cuda, csv_path))


if __name__ == "__main__":
    main()
