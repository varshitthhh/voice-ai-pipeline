"""Streaming-safe sentence boundary detection.

`feed()` only ever sees newly-arrived text plus a small buffered tail --
never anything ahead, since in a real token stream there is no "ahead"
yet. Not a complete NLP sentence segmenter (no handling for nested quotes,
ellipses as a single token, etc.) -- a pragmatic regex + abbreviation
guard, which is what most production streaming-TTS triggers actually use,
since the cost of an occasional wrong split is low (TTS just synthesizes
a slightly short or long chunk) compared to the cost of waiting for a full
NLP parse before ever triggering audio.
"""

import re
from typing import List, Optional

SENTENCE_END_RE = re.compile(r"([.!?])(\s+|$)")
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "vs.", "e.g.", "i.e.", "etc.", "approx.",
}


class SentenceBoundaryDetector:
    def __init__(self):
        self._buffer = ""

    def feed(self, text_delta: str) -> List[str]:
        """Appends `text_delta` and returns any newly-completed sentences,
        in order. Text after the last completed sentence stays buffered."""
        self._buffer += text_delta
        sentences = []
        search_start = 0

        while True:
            match = SENTENCE_END_RE.search(self._buffer, search_start)
            if not match:
                break

            end = match.end()
            candidate = self._buffer[: match.end(1)]  # up to and including the punctuation
            words = candidate.strip().split()
            last_word = words[-1].lower() if words else ""

            if last_word in ABBREVIATIONS:
                search_start = end  # false boundary -- keep scanning, don't consume yet
                continue

            sentence = self._buffer[:end].strip()
            sentences.append(sentence)
            self._buffer = self._buffer[end:].lstrip()
            search_start = 0

        return sentences

    def flush(self) -> Optional[str]:
        """Call once the stream ends: returns any trailing text that never
        hit a terminal punctuation mark (a response can legitimately end
        without one), or None if nothing is buffered."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
