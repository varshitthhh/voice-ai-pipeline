"""Phase 4.3: simulated audio playback with precise "what was actually
heard" tracking -- the foundation barge-in's history rewrite depends on.

No real speaker output in this environment -- chunks are "played" by
sleeping for their real synthesized duration, which is what actually
matters for barge-in: knowing, at the exact moment of interruption, how
much audio had reached the (real or simulated) speaker. A chunk that
finished its full duration was fully heard; a chunk still sleeping when
flush() is called was heard for exactly `elapsed / duration` of its
length; chunks still sitting in the queue were never heard at all.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioChunk:
    sentence: str
    duration_s: float


@dataclass
class PlaybackState:
    fully_played_sentences: List[str] = field(default_factory=list)
    partially_played_sentence: Optional[str] = None
    partial_fraction: float = 0.0  # 0..1, fraction of partially_played_sentence's words actually heard

    def heard_text(self) -> str:
        """Reconstructs what the user actually heard: every fully-played
        sentence verbatim, plus a word-count-proportional prefix of
        whichever sentence was cut off mid-playback (we don't have
        per-word audio timestamps from TTS, so proportional word count is
        the honest approximation -- documented, not hidden)."""
        parts = list(self.fully_played_sentences)
        if self.partially_played_sentence and self.partial_fraction > 0:
            words = self.partially_played_sentence.split()
            n_heard = max(1, round(len(words) * self.partial_fraction))
            parts.append(" ".join(words[:n_heard]))
        return " ".join(parts).strip()


class AudioPlayer:
    def __init__(self):
        self._queue: "asyncio.Queue[AudioChunk]" = asyncio.Queue()
        self._state = PlaybackState()
        self._current_chunk: Optional[AudioChunk] = None
        self._current_chunk_started_at: Optional[float] = None
        self._play_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    def reset(self) -> None:
        self._state = PlaybackState()
        self._current_chunk = None
        self._current_chunk_started_at = None

    def enqueue(self, chunk: AudioChunk) -> None:
        self._queue.put_nowait(chunk)

    def start(self) -> None:
        self._stopped.clear()
        self._play_task = asyncio.create_task(self._play_loop())

    def is_idle(self) -> bool:
        return self._queue.empty() and self._current_chunk is None

    async def _play_loop(self) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    chunk = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                self._current_chunk = chunk
                self._current_chunk_started_at = time.perf_counter()
                await asyncio.sleep(chunk.duration_s)  # CancelledError propagates through here on flush()
                self._state.fully_played_sentences.append(chunk.sentence)
                self._current_chunk = None
                self._current_chunk_started_at = None
        except asyncio.CancelledError:
            pass

    async def flush(self) -> PlaybackState:
        """Stops playback immediately: records exactly how much of the
        currently-playing chunk had played, drains any queued-but-never-
        started chunks (never heard), and returns the resulting state.
        Always leaves the play task fully awaited -- never cancels and
        walks away."""
        if self._current_chunk is not None and self._current_chunk_started_at is not None:
            elapsed = time.perf_counter() - self._current_chunk_started_at
            fraction = min(1.0, elapsed / self._current_chunk.duration_s) if self._current_chunk.duration_s > 0 else 0.0
            self._state.partially_played_sentence = self._current_chunk.sentence
            self._state.partial_fraction = fraction

        self._stopped.set()
        if self._play_task is not None:
            self._play_task.cancel()
            try:
                await self._play_task
            except asyncio.CancelledError:
                pass
            self._play_task = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._current_chunk = None
        self._current_chunk_started_at = None
        return self._state
