"""Phase 2.2 LLM benchmark: 2 models x 2 prompt lengths -> TTFT, tok/s.
Plus vLLM prefix caching on/off across a 6-turn conversation.

TODO(gpu-required): vLLM publishes no Windows wheel at all -- PyPI's
0.26.0 release has only `manylinux_2_28_{x86_64,aarch64}` wheels plus a
source sdist that needs a Linux + CUDA build toolchain vLLM does not
support building on Windows, checked directly against the PyPI JSON API
before writing this. This script is therefore entirely unrunnable on this
CPU-only Windows laptop and must run on the 24GB GPU (or a Linux+CUDA
Colab/cloud box). It is written against vLLM's real AsyncLLMEngine
streaming API, unexecuted and unverified against an actual vLLM install --
double check the API surface against whatever vLLM version lands on the
GPU box before trusting this end to end.

The harness's own math (TTFT/tok-s aggregation, prefix-cache speedup, CSV
schema) is validated separately with a mocked engine in
scripts/gate_2_2_synthetic_llm_bench.py, which actually runs and shares
this file's measurement code via src/llm_bench/harness.py.

Run on GPU hardware:
    python scripts/benchmark_llm.py
    python scripts/benchmark_llm.py --models Qwen/Qwen2.5-7B-Instruct-AWQ Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_bench import (
    CONVERSATION_TURNS,
    LONG_USER_MESSAGE,
    SHORT_USER_MESSAGE,
    SYSTEM_PROMPT,
    build_prompt,
    measure_stream,
    percentile,
)

PROMPT_LENGTH_CSV = ROOT / "outputs" / "llm_benchmark.csv"
PREFIX_CACHE_CSV = ROOT / "outputs" / "llm_prefix_caching.csv"

# Production pick vs. the unquantized baseline it was chosen over -- this is
# exactly the "decision doc justifying each pick against its rejects" the
# Phase 2 gate calls for, not an arbitrary pair.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-7B-Instruct",
]

MAX_OUTPUT_TOKENS = 200
N_REPEATS_PER_COMBO = 3  # for p50/p95 across repeats, not just a single sample


async def bench_prompt_lengths(models: list) -> list:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # TODO(gpu-required)

    rows = []
    prompt_lengths = {
        "short": build_prompt(SYSTEM_PROMPT, [], SHORT_USER_MESSAGE),
        "long": build_prompt(SYSTEM_PROMPT, [], LONG_USER_MESSAGE),
    }

    for model in models:
        # TODO(gpu-required): confirm the right `quantization=` value for
        # whatever vLLM version is installed on the GPU box -- AWQ models
        # are often auto-detected from the checkpoint's quant config, in
        # which case this kwarg may be unnecessary.
        engine_args = AsyncEngineArgs(model=model)
        engine = AsyncLLMEngine.from_engine_args(engine_args)  # TODO(gpu-required)

        for length_name, prompt in prompt_lengths.items():
            sampling_params = SamplingParams(max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0)

            ttfts, tok_s_values = [], []
            for rep in range(N_REPEATS_PER_COMBO):
                result = await measure_stream(engine, prompt, sampling_params, f"{model}-{length_name}-{rep}")
                ttfts.append(result.ttft_s)
                tok_s_values.append(result.tok_s)

            rows.append({
                "model": model,
                "prompt_length": length_name,
                "prompt_chars": len(prompt),
                "ttft_p50_ms": round(percentile(ttfts, 0.50) * 1000, 2),
                "ttft_p95_ms": round(percentile(ttfts, 0.95) * 1000, 2),
                "tok_s_p50": round(percentile(tok_s_values, 0.50), 2),
                "tok_s_p95": round(percentile(tok_s_values, 0.95), 2),
                "n_repeats": N_REPEATS_PER_COMBO,
            })
            print(f"{model} / {length_name}: TTFT_p50={rows[-1]['ttft_p50_ms']}ms tok/s_p50={rows[-1]['tok_s_p50']}")

    return rows


async def bench_prefix_caching(model: str) -> list:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # TODO(gpu-required)

    rows = []
    for enable_caching in (False, True):
        engine_args = AsyncEngineArgs(model=model, enable_prefix_caching=enable_caching)  # TODO(gpu-required)
        engine = AsyncLLMEngine.from_engine_args(engine_args)  # TODO(gpu-required)

        history = []
        for turn_idx, user_message in enumerate(CONVERSATION_TURNS):
            prompt = build_prompt(SYSTEM_PROMPT, history, user_message)
            sampling_params = SamplingParams(max_tokens=150, temperature=0.0)
            result = await measure_stream(engine, prompt, sampling_params, f"prefix-{enable_caching}-{turn_idx}")

            rows.append({
                "model": model,
                "enable_prefix_caching": enable_caching,
                "turn_index": turn_idx,
                "prompt_chars": len(prompt),
                "ttft_ms": round(result.ttft_s * 1000, 2) if result.ttft_s is not None else None,
                "tok_s": round(result.tok_s, 2),
            })
            history.append((user_message, result.text))
            print(f"caching={enable_caching} turn={turn_idx}: TTFT={rows[-1]['ttft_ms']}ms")

    return rows


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


async def main_async(args) -> None:
    length_rows = await bench_prompt_lengths(args.models)
    write_csv(length_rows, PROMPT_LENGTH_CSV)

    prefix_rows = await bench_prefix_caching(args.models[0])
    write_csv(prefix_rows, PREFIX_CACHE_CSV)

    # Turn 0 has no shared prefix to reuse; turns 1-5 do -- that's the
    # regime prefix caching should show a speedup in.
    def mean_ttft(rows, caching, min_turn=1):
        vals = [r["ttft_ms"] for r in rows if r["enable_prefix_caching"] == caching and r["turn_index"] >= min_turn and r["ttft_ms"] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    off = mean_ttft(prefix_rows, False)
    on = mean_ttft(prefix_rows, True)
    print(f"\nmean TTFT turns 1-5: caching off={off:.1f}ms, caching on={on:.1f}ms, speedup={off / on:.2f}x" if on else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
