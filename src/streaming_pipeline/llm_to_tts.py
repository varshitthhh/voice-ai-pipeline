"""Phase 4.2: streaming LLM -> TTS -- trigger TTS at the first sentence
boundary, never after the full response.

TODO(gpu-required): the real LLM engine is vLLM's AsyncLLMEngine, GPU/Linux
only (see scripts/benchmark_llm.py's docstring for why vLLM can't even
install on this machine). This orchestrator is engine-agnostic via the
same duck-typed contract src/llm_bench/harness.py established for Phase
2.2:

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        request_output.outputs[0].text   # cumulative text so far

so the identical code runs against the real vLLM engine on GPU
(scripts/llm_to_tts_pipeline.py) and against a mock engine here for local
validation (scripts/gate_4_2_mock_llm_to_tts.py) -- which pairs the mock
LLM stream with REAL Kokoro/Piper TTS synthesis, since both run fine on
CPU and only the LLM side is GPU-blocked.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from .sentence_boundary import SentenceBoundaryDetector


@dataclass
class StreamingRunResult:
    t_llm_start: float
    t_first_sentence_detected: Optional[float] = None
    t_first_tts_dispatched: Optional[float] = None
    t_first_tts_done: Optional[float] = None
    t_llm_response_done: Optional[float] = None
    t_all_tts_done: Optional[float] = None
    sentences: List[str] = field(default_factory=list)

    @property
    def time_to_first_audio_s(self) -> Optional[float]:
        """The actual streaming behavior: LLM start -> first sentence's TTS done."""
        if self.t_first_tts_done is None:
            return None
        return self.t_first_tts_done - self.t_llm_start

    @property
    def naive_time_to_first_audio_s(self) -> Optional[float]:
        """What time-to-first-audio would have been if TTS only started
        after the FULL LLM response finished -- the "latency killer" this
        phase exists to avoid. Approximated as full-response LLM time plus
        the observed first-sentence TTS duration (a fair proxy: that
        sentence's TTS synthesis time doesn't depend on when it started)."""
        if self.t_llm_response_done is None or self.t_first_tts_dispatched is None or self.t_first_tts_done is None:
            return None
        first_sentence_tts_duration = self.t_first_tts_done - self.t_first_tts_dispatched
        return (self.t_llm_response_done - self.t_llm_start) + first_sentence_tts_duration


class StreamingLLMToTTS:
    def __init__(self, tts_synthesize_fn: Callable[[str], Awaitable[None]]):
        """`tts_synthesize_fn(sentence)` is awaited inside its own task,
        dispatched the instant a sentence boundary is detected -- LLM token
        consumption is never blocked waiting for TTS to finish a prior
        sentence."""
        self.tts_synthesize_fn = tts_synthesize_fn

    async def run(self, engine, prompt: str, sampling_params, request_id: str) -> StreamingRunResult:
        detector = SentenceBoundaryDetector()
        result = StreamingRunResult(t_llm_start=time.perf_counter())
        seen_text = ""
        tts_tasks: list = []

        async for request_output in engine.generate(prompt, sampling_params, request_id):
            cumulative_text = request_output.outputs[0].text
            delta = cumulative_text[len(seen_text) :]
            seen_text = cumulative_text

            for sentence in detector.feed(delta):
                self._dispatch(result, sentence, tts_tasks)

        trailing = detector.flush()
        if trailing:
            self._dispatch(result, trailing, tts_tasks)

        result.t_llm_response_done = time.perf_counter()

        if tts_tasks:
            await asyncio.gather(*tts_tasks)
        result.t_all_tts_done = time.perf_counter()

        return result

    def _dispatch(self, result: StreamingRunResult, sentence: str, tts_tasks: list) -> None:
        now = time.perf_counter()
        if result.t_first_sentence_detected is None:
            result.t_first_sentence_detected = now
        if result.t_first_tts_dispatched is None:
            result.t_first_tts_dispatched = now
        result.sentences.append(sentence)

        async def _run() -> None:
            await self.tts_synthesize_fn(sentence)
            if result.t_first_tts_done is None:
                result.t_first_tts_done = time.perf_counter()

        tts_tasks.append(asyncio.create_task(_run()))
