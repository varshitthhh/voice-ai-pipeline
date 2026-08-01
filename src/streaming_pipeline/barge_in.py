"""Phase 4.3: barge-in.

When the user starts speaking during playback: cancel TTS generation,
flush the audio buffer, truncate LLM generation, and rewrite conversation
history to reflect only what the user actually heard -- not the full
response the model was generating, and not even the full sentence that
was playing when the interrupt landed.

BargeInOrchestrator.interrupt() is the core of this phase. Cancellation
order matters and is deliberate:
    1. cancel the LLM generation task (stop producing more text/sentences)
    2. cancel any in-flight TTS synthesis tasks, await their cancellation
    3. THEN flush the audio player (by now nothing new can be enqueued,
       so the flush sees a stable, final queue state)
    4. rewrite history from the player's exact heard-content state

Steps 2 then 3 are strictly sequenced (await between them) specifically to
avoid a race where a TTS task still in flight enqueues a chunk into the
player after flush() has already drained the queue -- see the docstring
on interrupt() below for why this ordering is safe under asyncio's
single-threaded cooperative scheduling.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .audio_player import AudioChunk, AudioPlayer
from .sentence_boundary import SentenceBoundaryDetector

VAD_FRAME_SAMPLES = 512  # Silero VAD v5's required window at 16kHz


@dataclass
class Turn:
    role: str
    text: str
    interrupted: bool = False


class ConversationHistory:
    def __init__(self):
        self.turns: List[Turn] = []

    def add_user_turn(self, text: str) -> None:
        self.turns.append(Turn(role="user", text=text))

    def add_assistant_turn(self, text: str, interrupted: bool = False) -> None:
        self.turns.append(Turn(role="assistant", text=text, interrupted=interrupted))

    def __repr__(self) -> str:
        lines = []
        for t in self.turns:
            tag = " [interrupted]" if t.interrupted else ""
            lines.append(f"{t.role}: {t.text}{tag}")
        return "\n".join(lines)


class BargeInOrchestrator:
    """One instance runs one assistant turn at a time. `tts_synthesize_fn`
    must be `async def synthesize(sentence: str) -> float` returning the
    synthesized audio's duration in seconds (used to drive AudioPlayer's
    realistic playback simulation) -- a different contract from Phase
    4.2's tts_synthesize_fn, which only needed to complete, not report
    duration, since 4.2 never simulated playback."""

    def __init__(self, history: ConversationHistory, tts_synthesize_fn):
        self.history = history
        self.tts_synthesize_fn = tts_synthesize_fn
        self.player = AudioPlayer()
        self._llm_task: Optional[asyncio.Task] = None
        self._tts_tasks: List[asyncio.Task] = []
        self._interrupted = asyncio.Event()

    async def run_turn(self, engine, prompt, sampling_params, request_id: str) -> Optional[str]:
        """Runs one full assistant turn: LLM generation, per-sentence TTS,
        simulated playback. Returns the committed text on normal
        completion, or None if interrupted (interrupt() commits history
        itself in that case, so run_turn must not double-commit)."""
        self.player.reset()
        self.player.start()
        self._tts_tasks = []
        self._interrupted.clear()

        detector = SentenceBoundaryDetector()
        full_text_so_far = ""

        async def _consume_llm() -> None:
            nonlocal full_text_so_far
            seen = ""
            async for request_output in engine.generate(prompt, sampling_params, request_id):
                cumulative = request_output.outputs[0].text
                delta = cumulative[len(seen) :]
                seen = cumulative
                full_text_so_far = cumulative
                for sentence in detector.feed(delta):
                    self._dispatch_tts(sentence)
            trailing = detector.flush()
            if trailing:
                self._dispatch_tts(trailing)

        self._llm_task = asyncio.create_task(_consume_llm())
        try:
            await self._llm_task
        except asyncio.CancelledError:
            pass

        # Wait for TTS + playback to finish naturally, but bail the instant
        # interrupt() fires concurrently -- it owns the history commit then.
        while True:
            if self._interrupted.is_set():
                return None
            if all(t.done() for t in self._tts_tasks) and self.player.is_idle():
                break
            await asyncio.sleep(0.01)

        if self._interrupted.is_set():  # no `await` since the check above -- race-free under asyncio's cooperative scheduling
            return None

        committed = full_text_so_far.strip()
        self.history.add_assistant_turn(committed)
        return committed

    def _dispatch_tts(self, sentence: str) -> None:
        async def _synth_and_enqueue() -> None:
            duration_s = await self.tts_synthesize_fn(sentence)
            self.player.enqueue(AudioChunk(sentence=sentence, duration_s=duration_s))

        self._tts_tasks.append(asyncio.create_task(_synth_and_enqueue()))

    async def interrupt(self) -> str:
        """Cancels generation and playback, rewrites history to heard-only
        content, returns that text. Every task this orchestrator started is
        either awaited to completion or explicitly cancelled-and-awaited
        here -- nothing is left running or merely dereferenced."""
        self._interrupted.set()

        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
            try:
                await self._llm_task
            except asyncio.CancelledError:
                pass

        for t in self._tts_tasks:
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks = []

        # Only safe to flush now that every TTS task is settled -- otherwise
        # a task still in flight could enqueue a chunk after flush() drains
        # the queue, and that chunk would wrongly survive into the next turn.
        state = await self.player.flush()
        heard_text = state.heard_text()
        if heard_text:  # interrupting before anything was actually said leaves nothing to remember
            self.history.add_assistant_turn(heard_text, interrupted=True)
        return heard_text


class BargeInListener:
    """Feeds incoming (real or simulated) mic audio through Silero VAD;
    `feed()` returns True the first time sustained speech is detected
    since the last reset(). Requires `min_consecutive_speech_frames`
    consecutive speech-classified 512-sample windows, not a single frame
    -- a naive one-frame trigger flickers on ordinary noise the same way
    Phase 4.1's first StreamingASR attempt did on raw per-frame
    thresholding; a false barge-in trigger is worse there (just a
    fragmented transcript) than here (an interrupted, discarded response),
    so the debounce matters more, not less."""

    def __init__(self, vad_model, sample_rate: int = 16000, min_consecutive_speech_frames: int = 3, speech_prob_threshold: float = 0.5):
        self.vad_model = vad_model
        self.sample_rate = sample_rate
        self.min_consecutive_speech_frames = min_consecutive_speech_frames
        self.speech_prob_threshold = speech_prob_threshold
        self._leftover = np.zeros(0, dtype=np.float32)
        self._consecutive_speech_frames = 0
        self._triggered = False
        self.vad_model.reset_states()

    def reset(self) -> None:
        self.vad_model.reset_states()
        self._leftover = np.zeros(0, dtype=np.float32)
        self._consecutive_speech_frames = 0
        self._triggered = False

    def feed(self, chunk: np.ndarray) -> bool:
        if self._triggered:
            return False

        vad_input = np.concatenate([self._leftover, np.asarray(chunk, dtype=np.float32)])
        n_full = len(vad_input) // VAD_FRAME_SAMPLES

        for i in range(n_full):
            window = vad_input[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            prob = self.vad_model(torch.from_numpy(window), self.sample_rate).item()
            self._consecutive_speech_frames = self._consecutive_speech_frames + 1 if prob >= self.speech_prob_threshold else 0

            if self._consecutive_speech_frames >= self.min_consecutive_speech_frames:
                self._triggered = True
                self._leftover = vad_input[(i + 1) * VAD_FRAME_SAMPLES :]
                return True

        self._leftover = vad_input[n_full * VAD_FRAME_SAMPLES :]
        return False
