"""Engine-agnostic LLM streaming benchmark harness.

`measure_stream` only assumes its `engine` argument exposes vLLM's real
async streaming interface:

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        request_output.outputs[0].text        # cumulative text so far
        request_output.outputs[0].token_ids   # cumulative token ids so far

That's it — no other vLLM-specific behavior is assumed. This lets the same
measurement code run against the real `vllm.AsyncLLMEngine` (scripts/
benchmark_llm.py, GPU-only) and against a mocked engine with synthetic
timings (scripts/gate_2_2_synthetic_llm_bench.py, runs anywhere) — the gate
is exercising the actual aggregation logic, not a reimplementation of it.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

SYSTEM_PROMPT = (
    "You are a helpful, concise customer support agent for an e-commerce store. "
    "Answer the customer's question directly and offer a next step."
)

SHORT_USER_MESSAGE = "Hi, do you have this jacket in size medium?"

LONG_USER_MESSAGE = (
    "Hi, I'm following up on an order I placed about two weeks ago that still "
    "hasn't shipped. I ordered a pair of running shoes, size 10, in the blue "
    "colorway, along with a set of no-show socks, and I paid for standard "
    "shipping which the checkout page said would take 5 to 7 business days. "
    "It's now been well past that window and the tracking page still just says "
    "'label created', which as far as I can tell means the carrier hasn't even "
    "picked the package up yet. I've checked my spam folder for any shipping "
    "delay notifications and there's nothing there either. I work from home most "
    "days so missing a delivery attempt shouldn't be the issue, and my address on "
    "the order is correct and up to date, I double-checked it against my last "
    "three orders which all arrived on time with the same carrier. I'd like to "
    "understand what's actually going on with this order, whether it's stuck in "
    "a warehouse somewhere or if there was some kind of processing error on your "
    "end, and depending on the answer I may want to either expedite the remaining "
    "shipping at no extra cost given the delay, or cancel the shoes specifically "
    "and get a refund for just that item while keeping the socks, since those "
    "apparently shipped separately and already arrived two days ago."
)

# A 6-turn conversation that grows the shared prefix each turn — this is
# exactly the shape prefix caching is supposed to help with.
CONVERSATION_TURNS = [
    "Hi, I ordered a pair of running shoes last week and they still haven't arrived.",
    "The order number is 48213. Can you check the status?",
    "It's been over 10 days, that seems really long for standard shipping.",
    "Can I get a refund instead of waiting any longer?",
    "How long will the refund take to process?",
    "Great, thank you. Is there anything else I need to do on my end?",
]


def build_prompt(system_prompt: str, history: list, new_user_message: str) -> str:
    """Simple chat-style prompt join. Turn N+1's prompt is a strict prefix
    extension of turn N's (same system prompt + same prior turns), which is
    the property prefix caching depends on."""
    lines = [f"System: {system_prompt}"]
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    lines.append(f"User: {new_user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


@dataclass
class StreamResult:
    ttft_s: Optional[float]
    n_output_tokens: int
    total_time_s: float
    text: str

    @property
    def tok_s(self) -> float:
        """Decode throughput, excluding prefill: tokens produced after the
        first token, divided by the time spent producing them."""
        if self.ttft_s is None or self.n_output_tokens <= 1:
            return float("nan")
        decode_time = self.total_time_s - self.ttft_s
        if decode_time <= 0:
            return float("nan")
        return (self.n_output_tokens - 1) / decode_time


async def measure_stream(engine, prompt: str, sampling_params, request_id: str) -> StreamResult:
    """Streams one generation request and times it. Never assumes anything
    about `engine` beyond the async-generator interface documented above."""
    t0 = time.perf_counter()
    ttft_s = None
    n_output_tokens = 0
    text = ""

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        output = request_output.outputs[0]
        if ttft_s is None and output.text:
            ttft_s = time.perf_counter() - t0
        n_output_tokens = len(output.token_ids)
        text = output.text

    total_time_s = time.perf_counter() - t0
    return StreamResult(ttft_s=ttft_s, n_output_tokens=n_output_tokens, total_time_s=total_time_s, text=text)


def percentile(values: list, p: float) -> float:
    finite = sorted(v for v in values if v is not None)
    if not finite:
        return float("nan")
    idx = min(int(len(finite) * p), len(finite) - 1)
    return finite[idx]
