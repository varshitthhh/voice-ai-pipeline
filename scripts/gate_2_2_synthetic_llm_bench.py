"""Phase 2.2 gate: validate the LLM benchmark harness with no real model.

vLLM cannot run on this machine at all (see scripts/benchmark_llm.py's
docstring), so there is no CPU-feasible substitute the way tiny/base
Whisper stood in for the ASR benchmark. What CAN be validated here is the
harness itself: TTFT/tok-s aggregation, the prefix-caching speedup
calculation, and CSV output -- using a mocked async engine with synthetic,
controllable per-token delays instead of a real model.

Critically, this exercises `measure_stream` from src/llm_bench/harness.py
directly -- the exact same function scripts/benchmark_llm.py calls against
the real vLLM engine -- so a bug in the measurement/aggregation logic
would show up here too, not just in unexecuted GPU-only code.

Run:
    python scripts/gate_2_2_synthetic_llm_bench.py
"""

import asyncio
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_bench import CONVERSATION_TURNS, SYSTEM_PROMPT, build_prompt, measure_stream, percentile

PROMPT_LENGTH_CSV = ROOT / "outputs" / "llm_benchmark_synthetic_gate.csv"
PREFIX_CACHE_CSV = ROOT / "outputs" / "llm_prefix_caching_synthetic_gate.csv"


@dataclass
class FakeCompletionOutput:
    text: str
    token_ids: list


@dataclass
class FakeRequestOutput:
    outputs: list


@dataclass
class FakeSamplingParams:
    max_tokens: int = 50


class FakeAsyncEngine:
    """Stands in for vllm.AsyncLLMEngine: same generate() interface, fully
    synthetic timing. Prefill delay scales with prompt length; when
    `enable_prefix_caching` is on, a prompt that extends a previously seen
    prompt gets its prefill cost discounted by the shared-prefix fraction,
    same shape as a real KV-cache prefix hit."""

    def __init__(self, base_prefill_s: float, per_char_prefill_s: float, decode_token_s: float, enable_prefix_caching: bool = False):
        self.base_prefill_s = base_prefill_s
        self.per_char_prefill_s = per_char_prefill_s
        self.decode_token_s = decode_token_s
        self.enable_prefix_caching = enable_prefix_caching
        self._seen_prompts: list = []

    def _prefill_delay(self, prompt: str) -> float:
        full_delay = self.base_prefill_s + self.per_char_prefill_s * len(prompt)
        if not self.enable_prefix_caching:
            return full_delay

        best_shared = max((len(p) for p in self._seen_prompts if prompt.startswith(p)), default=0)
        cache_hit_ratio = best_shared / len(prompt) if prompt else 0.0
        return full_delay * (1 - cache_hit_ratio)

    async def generate(self, prompt: str, sampling_params: FakeSamplingParams, request_id: str):
        await asyncio.sleep(self._prefill_delay(prompt))

        tokens = []
        text = ""
        for i in range(sampling_params.max_tokens):
            tokens.append(i)
            text += "x "
            yield FakeRequestOutput(outputs=[FakeCompletionOutput(text=text, token_ids=list(tokens))])
            await asyncio.sleep(self.decode_token_s)

        self._seen_prompts.append(prompt)


async def bench_prompt_lengths_synthetic() -> list:
    fake_models = {
        "fake-model-a": FakeAsyncEngine(base_prefill_s=0.02, per_char_prefill_s=0.00015, decode_token_s=0.01),
        "fake-model-b": FakeAsyncEngine(base_prefill_s=0.05, per_char_prefill_s=0.00040, decode_token_s=0.02),
    }
    prompt_lengths = {
        "short": build_prompt(SYSTEM_PROMPT, [], "Hi, do you have this jacket in size medium?"),
        "long": build_prompt(SYSTEM_PROMPT, [], "x " * 400),  # synthetic long prompt, length is all that matters here
    }

    rows = []
    for model_name, engine in fake_models.items():
        for length_name, prompt in prompt_lengths.items():
            ttfts, tok_s_values = [], []
            for rep in range(3):
                result = await measure_stream(engine, prompt, FakeSamplingParams(max_tokens=30), f"{model_name}-{length_name}-{rep}")
                ttfts.append(result.ttft_s)
                tok_s_values.append(result.tok_s)

            rows.append({
                "model": model_name,
                "prompt_length": length_name,
                "ttft_p50_ms": round(percentile(ttfts, 0.50) * 1000, 2),
                "ttft_p95_ms": round(percentile(ttfts, 0.95) * 1000, 2),
                "tok_s_p50": round(percentile(tok_s_values, 0.50), 2),
                "tok_s_p95": round(percentile(tok_s_values, 0.95), 2),
            })
    return rows


async def bench_prefix_caching_synthetic() -> list:
    rows = []
    for enable_caching in (False, True):
        engine = FakeAsyncEngine(base_prefill_s=0.02, per_char_prefill_s=0.0003, decode_token_s=0.01, enable_prefix_caching=enable_caching)
        history = []
        for turn_idx, user_message in enumerate(CONVERSATION_TURNS):
            prompt = build_prompt(SYSTEM_PROMPT, history, user_message)
            result = await measure_stream(engine, prompt, FakeSamplingParams(max_tokens=20), f"prefix-{enable_caching}-{turn_idx}")
            rows.append({
                "enable_prefix_caching": enable_caching,
                "turn_index": turn_idx,
                "prompt_chars": len(prompt),
                "ttft_ms": round(result.ttft_s * 1000, 2),
                "tok_s": round(result.tok_s, 2),
            })
            history.append((user_message, "a synthetic reply that grows the shared context"))
    return rows


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:  # gate output: overwrite, not accumulate
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


async def main_async() -> bool:
    print("=== synthetic prompt-length benchmark ===")
    length_rows = await bench_prompt_lengths_synthetic()
    for r in length_rows:
        print(r)
    write_csv(length_rows, PROMPT_LENGTH_CSV)

    long_ttft = next(r["ttft_p50_ms"] for r in length_rows if r["model"] == "fake-model-a" and r["prompt_length"] == "long")
    short_ttft = next(r["ttft_p50_ms"] for r in length_rows if r["model"] == "fake-model-a" and r["prompt_length"] == "short")
    length_ok = long_ttft > short_ttft
    print(f"[{'PASS' if length_ok else 'FAIL'}] long-prompt TTFT ({long_ttft}ms) > short-prompt TTFT ({short_ttft}ms)")

    print("\n=== synthetic prefix-caching benchmark ===")
    prefix_rows = await bench_prefix_caching_synthetic()
    for r in prefix_rows:
        print(r)
    write_csv(prefix_rows, PREFIX_CACHE_CSV)

    def mean_ttft(rows, caching, min_turn=1):
        vals = [r["ttft_ms"] for r in rows if r["enable_prefix_caching"] == caching and r["turn_index"] >= min_turn]
        return sum(vals) / len(vals)

    off = mean_ttft(prefix_rows, False)
    on = mean_ttft(prefix_rows, True)
    caching_ok = on < off
    print(f"[{'PASS' if caching_ok else 'FAIL'}] mean TTFT turns 1-5: caching on ({on:.1f}ms) < caching off ({off:.1f}ms), speedup={off / on:.2f}x")

    tok_s_finite = all(r["tok_s"] == r["tok_s"] for r in prefix_rows)  # NaN != NaN
    print(f"[{'PASS' if tok_s_finite else 'FAIL'}] all tok_s values are finite (no NaN from the aggregation math)")

    return length_ok and caching_ok and tok_s_finite


def main() -> None:
    ok = asyncio.run(main_async())
    print("\nGATE PASS" if ok else "\nGATE FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
