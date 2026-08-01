"""Phase 4.2 gate: streaming LLM -> TTS, mock LLM + REAL TTS.

The LLM side is mocked (vLLM can't run on this machine at all -- see
scripts/llm_to_tts_pipeline.py's docstring), but the TTS side is real
Piper synthesis (validated working in Phase 2.3, fast enough to run
locally). Only the genuinely GPU-blocked half of this pipeline is faked;
everything downstream of "the LLM produced this text" is real.

Demonstrates the actual claim Phase 4.2 exists to prove: time-to-first-
audio when triggering TTS at each sentence boundary is far below what it
would be waiting for the complete response before synthesizing anything.

Run:
    python scripts/gate_4_2_mock_llm_to_tts.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piper import PiperVoice

from streaming_pipeline import StreamingLLMToTTS

PIPER_MODEL_PATH = ROOT / "data" / "tts" / "models" / "piper" / "en_US-lessac-medium.onnx"

RESPONSE_TEXT = (
    "I'm sorry to hear your order hasn't arrived yet. "
    "Let me look into that for you right away. "
    "I can see it's currently in transit and should arrive within two days. "
    "Is there anything else I can help you with?"
)

# Timing calibrated to Phase 2.2's expected TTFT/tok-s ranges (README
# Section 1: LLM TTFT 90-300ms) -- prefill once, then per-token decode delay.
PREFILL_DELAY_S = 0.15
DECODE_DELAY_S_PER_TOKEN = 0.02  # ~50 tok/s


@dataclass
class FakeCompletionOutput:
    text: str


@dataclass
class FakeRequestOutput:
    outputs: list


class FakeAsyncEngine:
    """Same duck-typed interface as vllm.AsyncLLMEngine (see
    src/llm_bench/harness.py): async generate() yielding cumulative text.
    Streams RESPONSE_TEXT out word-by-word with realistic prefill + decode
    timing, standing in for the GPU-only real engine."""

    async def generate(self, prompt: str, sampling_params, request_id: str):
        await asyncio.sleep(PREFILL_DELAY_S)
        words = RESPONSE_TEXT.split(" ")
        text = ""
        for i, word in enumerate(words):
            text += word if i == 0 else " " + word
            yield FakeRequestOutput(outputs=[FakeCompletionOutput(text=text)])
            await asyncio.sleep(DECODE_DELAY_S_PER_TOKEN)


def make_tts_synthesize_fn(voice: PiperVoice, synthesized_log: list):
    """Wraps Piper's synchronous synthesize() in a thread so it doesn't
    block the event loop -- real async orchestration, not just a sync call
    dressed up as async."""

    async def synthesize(sentence: str) -> None:
        def _blocking_synthesize():
            chunks = list(voice.synthesize(sentence))
            total_samples = sum(len(c.audio_float_array) for c in chunks)
            return total_samples, chunks[0].sample_rate if chunks else 0

        t0 = time.perf_counter()
        total_samples, sample_rate = await asyncio.to_thread(_blocking_synthesize)
        elapsed = time.perf_counter() - t0
        audio_duration = total_samples / sample_rate if sample_rate else 0.0
        synthesized_log.append({"sentence": sentence, "synth_time_s": elapsed, "audio_duration_s": audio_duration})

    return synthesize


async def main_async() -> bool:
    if not PIPER_MODEL_PATH.exists():
        raise SystemExit(f"missing {PIPER_MODEL_PATH} -- run scripts/prepare_tts_models.py first")

    voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    list(voice.synthesize("warm up."))  # warmup, excluded from timing

    synthesized_log: list = []
    orchestrator = StreamingLLMToTTS(tts_synthesize_fn=make_tts_synthesize_fn(voice, synthesized_log))
    engine = FakeAsyncEngine()

    result = await orchestrator.run(engine, prompt="(mock prompt)", sampling_params=None, request_id="gate-4.2")

    print(f"sentences detected: {len(result.sentences)}")
    for i, s in enumerate(result.sentences):
        print(f"  [{i}] {s!r}")

    print("\nTTS synthesis log:")
    for row in synthesized_log:
        print(f"  {row['synth_time_s'] * 1000:.1f}ms synth -> {row['audio_duration_s']:.2f}s audio  {row['sentence']!r}")

    streaming_ms = result.time_to_first_audio_s * 1000 if result.time_to_first_audio_s is not None else float("nan")
    naive_ms = result.naive_time_to_first_audio_s * 1000 if result.naive_time_to_first_audio_s is not None else float("nan")
    full_response_ms = (result.t_llm_response_done - result.t_llm_start) * 1000

    print(f"\nfull LLM response time: {full_response_ms:.1f}ms")
    print(f"time-to-first-audio, STREAMING (trigger per sentence): {streaming_ms:.1f}ms")
    print(f"time-to-first-audio, NAIVE (wait for full response):   {naive_ms:.1f}ms")
    print(f"speedup: {naive_ms / streaming_ms:.2f}x")

    all_ok = True

    n_sentences_ok = len(result.sentences) == 4
    print(f"\n[{'PASS' if n_sentences_ok else 'FAIL'}] detected all 4 sentences")
    all_ok = all_ok and n_sentences_ok

    streaming_faster_ok = streaming_ms < naive_ms
    print(f"[{'PASS' if streaming_faster_ok else 'FAIL'}] streaming time-to-first-audio < naive time-to-first-audio")
    all_ok = all_ok and streaming_faster_ok

    streaming_faster_than_full_response_ok = streaming_ms < full_response_ms
    print(f"[{'PASS' if streaming_faster_than_full_response_ok else 'FAIL'}] first audio arrives before the full LLM response even finishes")
    all_ok = all_ok and streaming_faster_than_full_response_ok

    return all_ok


def main() -> None:
    ok = asyncio.run(main_async())
    print("\nGATE PASS" if ok else "\nGATE FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
