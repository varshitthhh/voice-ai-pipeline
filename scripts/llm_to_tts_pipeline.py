"""Phase 4.2: real streaming LLM -> TTS pipeline (Qwen2.5-7B-Instruct AWQ
on vLLM -> Kokoro).

TODO(gpu-required): vLLM publishes no Windows wheel at all -- PyPI's
release has only `manylinux` wheels plus a source sdist needing a Linux +
CUDA build toolchain vLLM doesn't support building on Windows, confirmed
directly against the PyPI JSON API (see scripts/benchmark_llm.py's
docstring). This script is therefore entirely unrunnable on this CPU-only
Windows laptop and must run on the 24GB GPU. It is written against vLLM's
real AsyncLLMEngine streaming API, unexecuted and unverified against an
actual vLLM install -- double check the API surface against whatever vLLM
version lands on the GPU box before trusting this end to end.

The orchestration logic itself (src/streaming_pipeline/) is fully
validated -- see scripts/gate_4_2_mock_llm_to_tts.py, which runs the exact
same StreamingLLMToTTS class against a mock LLM stream paired with REAL
Kokoro/Piper TTS synthesis. Only the LLM engine differs between that gate
and this script; everything from "cumulative text arrived" onward is
identical, real code.

Run on GPU hardware:
    python scripts/llm_to_tts_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streaming_pipeline import StreamingLLMToTTS

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
KOKORO_MODEL_PATH = ROOT / "data" / "tts" / "models" / "kokoro" / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH = ROOT / "data" / "tts" / "models" / "kokoro" / "voices-v1.0.bin"

SYSTEM_PROMPT = (
    "You are a helpful, concise customer support agent for an e-commerce store. "
    "Answer the customer's question directly and offer a next step."
)
TEST_USER_MESSAGE = "Hi, I ordered a pair of running shoes last week and they still haven't arrived."


def make_kokoro_tts_synthesize_fn():
    """TODO(gpu-required): unexecuted. Wraps Kokoro's real async
    create_stream() API (verified against the actual installed package in
    Phase 2.3 -- see scripts/benchmark_tts.py) for use as
    StreamingLLMToTTS's tts_synthesize_fn."""
    from kokoro_onnx import Kokoro  # TODO(gpu-required): unexecuted here

    kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    async def synthesize(sentence: str) -> None:
        async for _samples, _sr in kokoro.create_stream(sentence, voice="af_heart"):
            pass  # a real deployment would push these chunks to the audio-out websocket here (Phase 7)

    return synthesize


async def main_async() -> None:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # TODO(gpu-required)

    # TODO(gpu-required): confirm the right `quantization=` value for
    # whatever vLLM version is installed on the GPU box -- AWQ is often
    # auto-detected from the checkpoint's quant config.
    engine_args = AsyncEngineArgs(model=MODEL)
    engine = AsyncLLMEngine.from_engine_args(engine_args)  # TODO(gpu-required)
    sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

    prompt = f"System: {SYSTEM_PROMPT}\nUser: {TEST_USER_MESSAGE}\nAssistant:"

    orchestrator = StreamingLLMToTTS(tts_synthesize_fn=make_kokoro_tts_synthesize_fn())
    result = await orchestrator.run(engine, prompt, sampling_params, request_id="llm-to-tts-demo")

    print(f"sentences: {result.sentences}")
    print(f"time-to-first-audio (streaming): {result.time_to_first_audio_s * 1000:.1f}ms")
    print(f"time-to-first-audio (naive, full response first): {result.naive_time_to_first_audio_s * 1000:.1f}ms")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
