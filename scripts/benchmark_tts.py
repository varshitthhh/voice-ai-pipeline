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
    Kokoro (kokoro-onnx, int8, CPU) -- runs and is timed locally.
    Piper  (piper-tts, CPU)          -- runs and is timed locally.
    XTTS v2 (coqui-tts)              -- TODO(gpu-required), see below.

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
    await _synth_kokoro_one(kokoro, WARMUP_SENTENCE)

    rows = []
    for idx, sentence in enumerate(sentences):
        ttfcs, totals = [], []
        for _ in range(N_REPEATS):
            result = await _synth_kokoro_one(kokoro, sentence)
            ttfcs.append(result["ttfc_s"])
            totals.append(result["total_s"])
        rows.append({
            "engine": "kokoro", "sentence_index": idx, "sentence_chars": len(sentence),
            "ttfc_p50_ms": round(percentile(ttfcs, 0.50) * 1000, 1),
            "ttfc_p95_ms": round(percentile(ttfcs, 0.95) * 1000, 1),
            "total_p50_ms": round(percentile(totals, 0.50) * 1000, 1),
            "audio_duration_s": round(result["audio_duration_s"], 3),
        })
        print(f"kokoro[{idx}]: TTFC_p50={rows[-1]['ttfc_p50_ms']}ms  {sentence!r}")
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


def bench_piper(sentences: list) -> list:
    from piper import PiperVoice  # local import: only required if this engine is actually run

    voice = PiperVoice.load(str(MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"))
    _synth_piper_one(voice, WARMUP_SENTENCE)

    rows = []
    for idx, sentence in enumerate(sentences):
        ttfcs, totals = [], []
        for _ in range(N_REPEATS):
            result = _synth_piper_one(voice, sentence)
            ttfcs.append(result["ttfc_s"])
            totals.append(result["total_s"])
        rows.append({
            "engine": "piper", "sentence_index": idx, "sentence_chars": len(sentence),
            "ttfc_p50_ms": round(percentile(ttfcs, 0.50) * 1000, 1),
            "ttfc_p95_ms": round(percentile(ttfcs, 0.95) * 1000, 1),
            "total_p50_ms": round(percentile(totals, 0.50) * 1000, 1),
            "audio_duration_s": round(result["audio_duration_s"], 3),
        })
        print(f"piper[{idx}]: TTFC_p50={rows[-1]['ttfc_p50_ms']}ms  {sentence!r}")
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


async def main_async(engines: list) -> None:
    sentences = split_sentences(TEST_TEXT)
    print(f"{len(sentences)} test sentences")

    all_rows = []
    for engine in engines:
        if engine == "xtts":
            print("\n--- xtts SKIPPED: TODO(gpu-required), see benchmark_xtts() docstring ---")
            continue
        print(f"\n--- {engine} ---")
        runner = ENGINE_RUNNERS[engine]
        result = runner(sentences)
        if asyncio.iscoroutine(result):
            result = await result
        all_rows.extend(result)

    write_csv(all_rows, CSV_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", default=["kokoro", "piper"], choices=list(ENGINE_RUNNERS.keys()))
    args = parser.parse_args()
    asyncio.run(main_async(args.engines))


if __name__ == "__main__":
    main()
