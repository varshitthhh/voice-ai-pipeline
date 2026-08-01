"""Phase 4.1: streaming ASR -- VAD-gated audio chunks, partial hypotheses
piped forward as they're produced.

Unlike src/features/pipeline.py's FeaturePipeline (which re-transcribes
continuously regardless of speech/silence, because it needs a fresh
partial transcript as a *feature* for the turn-taking model every 100ms
frame), StreamingASR is genuinely VAD-gated: audio is only accumulated
into the transcription buffer while Silero VAD detects an active speech
segment. Silence outside a segment is dropped, never transcribed -- this
is the ASR pipeline stage Phase 4.2 feeds into the LLM, so transcribing
silence would be wasted compute in production, not just a feature-
extraction nicety.

VAD boundary detection runs at Silero VAD v5's native 512-sample (32ms)
granularity, through VADIterator rather than a raw per-frame probability
threshold -- confirmed necessary by direct testing: a naive "is this one
512-sample window >= 0.5" check flickers in and out of "speech" during
ordinary momentary dips *within* a single utterance (between words,
plosives), fragmenting one real segment into a dozen tiny garbage ones.
VADIterator's `min_silence_duration_ms` debounce (a short one here --
this is "has this burst of audio stopped," not the higher-level "is the
user's conversational turn complete" question Phase 3's endpointer
answers separately and independently) is the same hysteresis Phase 3.1's
baseline relies on for exactly this reason.

Re-transcription -- an order of magnitude more expensive than a VAD
forward pass -- is throttled to a coarser cadence via
`partial_every_n_speech_samples`.

Every partial and the final hypothesis is emitted as a validated
schemas.AsrResult (Phase 1.4) -- reject-and-report all the way through
the pipeline, not just at this component's own boundary.
"""

import math
import time
import uuid
from typing import Optional

import numpy as np
from silero_vad import VADIterator

from schemas import AsrResult, validate_payload

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # Silero VAD v5's required window at 16kHz
DEFAULT_PARTIAL_EVERY_N_SPEECH_SAMPLES = 8000  # ~500ms of accumulated speech between re-decodes
DEFAULT_MIN_SILENCE_DURATION_MS = 200  # segment-boundary debounce, not the turn-completion threshold


class StreamingASR:
    def __init__(
        self,
        asr_model,
        vad_model,
        session_id: str,
        sample_rate: int = SAMPLE_RATE,
        partial_every_n_speech_samples: int = DEFAULT_PARTIAL_EVERY_N_SPEECH_SAMPLES,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
        language: str = "en",
    ):
        self.asr_model = asr_model
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.partial_every_n_speech_samples = partial_every_n_speech_samples
        self.language = language

        self._vad = VADIterator(vad_model, sampling_rate=sample_rate, min_silence_duration_ms=min_silence_duration_ms)
        self._vad_leftover = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._segment_audio = np.zeros(0, dtype=np.float32)
        self._samples_since_last_partial = 0
        self._turn_id: Optional[str] = None

    def _consume_vad_events(self, chunk: np.ndarray) -> tuple:
        """Runs VADIterator at its native 512-sample granularity over
        `chunk` (any length -- leftover samples below 512 carry to the next
        call). Returns (started, ended): whether a segment start/end event
        fired anywhere within this chunk. Both can be True in the same
        call for a very short segment, though that's not a shape this
        module's test audio exercises."""
        vad_input = np.concatenate([self._vad_leftover, chunk])
        n_full = len(vad_input) // VAD_FRAME_SAMPLES
        started, ended = False, False
        for i in range(n_full):
            window = vad_input[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            event = self._vad(window, return_seconds=False)
            if event is not None:
                if "start" in event:
                    started = True
                if "end" in event:
                    ended = True
        self._vad_leftover = vad_input[n_full * VAD_FRAME_SAMPLES :]
        return started, ended

    def _transcribe(self, audio: np.ndarray) -> tuple:
        segments, _info = self.asr_model.transcribe(audio, beam_size=1, language=self.language)
        segments = list(segments)
        text = " ".join(s.text for s in segments).strip()
        confidence = float(np.clip(np.mean([math.exp(s.avg_logprob) for s in segments]), 0.0, 1.0)) if segments else 0.0
        return text, confidence

    def _build_result(self, text: str, is_final: bool, confidence: float) -> Optional[AsrResult]:
        instance, report = validate_payload(AsrResult, {
            "session_id": self.session_id,
            "turn_id": self._turn_id,
            "transcript": text,
            "is_final": is_final,
            "confidence": confidence,
            "language": self.language,
            "t_result": time.monotonic(),
        })
        if not report.valid:
            # Reachable in practice mainly for a final result with empty
            # text (AsrResult requires non-empty transcript when
            # is_final=True) -- a VAD false-trigger with nothing actually
            # decoded. Dropping rather than forcing a fake string.
            return None
        return instance

    def step(self, chunk: np.ndarray) -> Optional[AsrResult]:
        """Feed one audio chunk (any length -- VAD-native 512-sample
        alignment is handled internally). Returns an AsrResult for a new
        partial or final hypothesis, or None if there's nothing new to
        report (still in silence, or not enough new speech accumulated yet
        for another partial)."""
        started, ended = self._consume_vad_events(chunk)

        if started and not self._in_speech:
            self._in_speech = True
            self._turn_id = uuid.uuid4().hex[:8]
            self._segment_audio = np.zeros(0, dtype=np.float32)
            self._samples_since_last_partial = 0

        if self._in_speech:
            self._segment_audio = np.concatenate([self._segment_audio, np.asarray(chunk, dtype=np.float32)])
            self._samples_since_last_partial += len(chunk)

        if ended and self._in_speech:
            self._in_speech = False
            text, confidence = self._transcribe(self._segment_audio)
            self._segment_audio = np.zeros(0, dtype=np.float32)
            return self._build_result(text, is_final=True, confidence=confidence) if text else None

        if self._in_speech and self._samples_since_last_partial >= self.partial_every_n_speech_samples:
            self._samples_since_last_partial = 0
            text, confidence = self._transcribe(self._segment_audio)
            return self._build_result(text, is_final=False, confidence=confidence) if text else None

        return None
