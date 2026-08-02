"""Phase 2.2 LLM benchmark: 2 models x 2 prompt lengths -> TTFT, tok/s.
Plus vLLM prefix caching on/off across a 6-turn conversation.

TODO(gpu-required): vLLM publishes no Windows wheel at all -- PyPI's
0.26.0 release has only `manylinux_2_28_{x86_64,aarch64}` wheels plus a
source sdist that needs a Linux + CUDA build toolchain vLLM does not
support building on Windows, checked directly against the PyPI JSON API
before writing this. This script is therefore entirely unrunnable on this
CPU-only Windows laptop and must run on a Linux+CUDA box (Colab T4). It is
written against vLLM's real AsyncLLMEngine streaming API, unexecuted and
unverified against an actual vLLM install -- double check the API surface
against whatever vLLM version lands on the GPU box before trusting this
end to end.

The harness's own math (TTFT/tok-s aggregation, prefix-cache speedup, CSV
schema) is validated separately with a mocked engine in
scripts/gate_2_2_synthetic_llm_bench.py, which actually runs and shares
this file's measurement code via src/llm_bench/harness.py.

**GPU memory: one engine per process, not per script invocation.** Creating
more than one vllm.AsyncLLMEngine sequentially in the same Python process is
a known vLLM failure mode on memory-constrained cards -- each engine grabs
~90% of currently-free VRAM for its KV cache by default, and that isn't
reliably released back when a Python object goes out of scope (see
vllm-project/vllm issues #654 and #14376: "second model requires more
memory... No available memory for the cache blocks"). This script needs up
to 4 engine instantiations total (2 models x prompt-lengths, plus 1 model x
2 caching states). On a 16GB T4 the old default (all 4 in one process) will
likely OOM on the 2nd-4th engine. Use --model/--mode for a single engine per
process -- a full process exit guarantees the GPU memory is actually freed
before the next one starts:

    python scripts/benchmark_llm.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --mode prompt-lengths
    python scripts/benchmark_llm.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --mode prompt-lengths
    python scripts/benchmark_llm.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --mode prefix-caching --caching off
    python scripts/benchmark_llm.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --mode prefix-caching --caching on

The old all-in-one-process form still exists for local/CI convenience on
hardware with enough VRAM to absorb 4 sequential engines (e.g. a 24GB+ card):
    python scripts/benchmark_llm.py
    python scripts/benchmark_llm.py --models Qwen/Qwen2.5-7B-Instruct-AWQ Qwen/Qwen2.5-7B-Instruct  # AWQ vs unquantized instead
"""

import argparse
import asyncio
import csv
import gc
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

PROMPT_LENGTH_CSV = ROOT / "outputs" / "llm_benchmark_gpu.csv"
PREFIX_CACHE_CSV = ROOT / "outputs" / "llm_prefix_caching_gpu.csv"

# Production pick vs. a smaller same-family AWQ model -- the "is the extra
# 7B worth it over 3B" question, not the "was quantizing worth it" question
# (that comparison, 7B-AWQ vs unquantized 7B, was the original default here;
# switched per direct instruction to compare against the smaller model instead).
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-3B-Instruct-AWQ",
]

MAX_OUTPUT_TOKENS = 200
N_REPEATS_PER_COMBO = 3  # for p50/p95 across repeats, not just a single sample


def _make_engine(model: str, enable_prefix_caching: bool = False):
    """Isolated so both call sites (and the try/except around vLLM's real
    API surface, unverified until this actually runs) share one place to
    fix things when the API doesn't match what's assumed here.

    TODO(gpu-required): confirm the right `quantization=` value for
    whatever vLLM version is installed on the GPU box -- AWQ models are
    often auto-detected from the checkpoint's quant config, in which case
    this kwarg may be unnecessary.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # TODO(gpu-required)

    try:
        engine_args = AsyncEngineArgs(model=model, enable_prefix_caching=enable_prefix_caching)
        return AsyncLLMEngine.from_engine_args(engine_args)
    except TypeError as e:
        raise SystemExit(
            f"AsyncEngineArgs/AsyncLLMEngine construction failed for model={model!r}: {e}\n"
            "This is exactly the API-surface risk flagged in this script's module docstring -- "
            "vLLM's AsyncEngineArgs kwargs (e.g. enable_prefix_caching, quantization) have moved "
            "between versions. Check `python -c \"from vllm import AsyncEngineArgs; help(AsyncEngineArgs)\"` "
            "against the installed version and adjust the call in _make_engine() above."
        ) from e


def _free_engine(engine) -> None:
    """Best-effort GPU memory cleanup between engines in the same process.
    Defense in depth only -- the reliable fix for the known vLLM
    multi-engine-OOM failure mode is running one engine per process (see
    --model/--mode below), not this. Safe to call even when nothing needs
    freeing (e.g. the CPU-only import-check path)."""
    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


async def _bench_one_model_prompt_lengths(model: str) -> list:
    from vllm import SamplingParams  # TODO(gpu-required)

    prompt_lengths = {
        "short": build_prompt(SYSTEM_PROMPT, [], SHORT_USER_MESSAGE),
        "long": build_prompt(SYSTEM_PROMPT, [], LONG_USER_MESSAGE),
    }

    engine = _make_engine(model)
    rows = []
    try:
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
    finally:
        _free_engine(engine)

    return rows


async def bench_prompt_lengths(models: list) -> list:
    rows = []
    for model in models:
        rows.extend(await _bench_one_model_prompt_lengths(model))
    return rows


async def _bench_one_caching_config(model: str, enable_caching: bool) -> list:
    from vllm import SamplingParams  # TODO(gpu-required)

    engine = _make_engine(model, enable_prefix_caching=enable_caching)
    rows = []
    try:
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
    finally:
        _free_engine(engine)

    return rows


async def bench_prefix_caching(model: str) -> list:
    rows = []
    for enable_caching in (False, True):
        rows.extend(await _bench_one_caching_config(model, enable_caching))
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


async def main_async_single(args) -> None:
    """One engine, one process, one CSV append -- the safe path for
    memory-constrained GPUs (see module docstring). Run this 4x from the
    notebook/shell for the full matrix; results accumulate in the same CSVs
    the all-in-one-process path writes to."""
    if args.mode == "prompt-lengths":
        rows = await _bench_one_model_prompt_lengths(args.model)
        write_csv(rows, PROMPT_LENGTH_CSV)
    else:
        if args.caching is None:
            raise SystemExit("--mode prefix-caching requires --caching {on,off}")
        enable_caching = args.caching == "on"
        rows = await _bench_one_caching_config(args.model, enable_caching)
        write_csv(rows, PREFIX_CACHE_CSV)


async def main_async_all_in_one(args) -> None:
    """Original all-4-engines-in-one-process path. Kept for convenience on
    hardware with enough VRAM to absorb it (e.g. 24GB+); on a memory-
    constrained card use --model/--mode instead (see module docstring)."""
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
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="all-in-one-process mode: benchmark all of these sequentially")
    parser.add_argument("--model", default=None, help="single-engine-per-process mode: benchmark just this model (use with --mode)")
    parser.add_argument("--mode", choices=["prompt-lengths", "prefix-caching"], default=None, help="required together with --model")
    parser.add_argument("--caching", choices=["on", "off"], default=None, help="required when --mode prefix-caching")
    args = parser.parse_args()

    if args.model is not None:
        if args.mode is None:
            raise SystemExit("--model requires --mode {prompt-lengths,prefix-caching}")
        asyncio.run(main_async_single(args))
    else:
        asyncio.run(main_async_all_in_one(args))


if __name__ == "__main__":
    main()
