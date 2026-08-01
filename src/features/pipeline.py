"""Phase 3.2 streaming feature pipeline.

Per 100ms frame: last ~20 partial ASR tokens, pitch, energy, speech rate,
pause-so-far -- the feature set the Phase 3.3 turn-taking model consumes.

Streaming-safety is structural, not just windowed math: `FeaturePipeline`
exposes a single `step(chunk)` method that accepts exactly one new 100ms
chunk at a time and only ever appends it to internal history. There is no
method, path, or parameter anywhere in this class that accepts audio ahead
of the current frame -- a feature computed at frame t physically cannot
depend on frame t+1, because frame t+1 has not been passed to the object
yet when step() runs for frame t.

That said, "the API only takes one chunk" is a design argument, not proof
against algorithmic bugs (an off-by-one in a buffer index could still leak
a sample or two). scripts/gate_3_2_leakage_audit.py is the actual
verification: it runs this pipeline on the full audio and, separately, on
audio truncated at several earlier cut points, and asserts every frame
before the cut is bit-identical between runs. See that script for the
audit; this docstring is the "how you verified it" write-up it refers to.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .prosody import estimate_pitch, rms_energy

VAD_FRAME_SAMPLES = 512  # Silero VAD v5's required window size at 16kHz
VAD_SPEECH_PROB_THRESHOLD = 0.5


@dataclass
class FrameFeatures:
    frame_index: int
    t_ms: float
    pitch_hz: float
    energy_rms: float
    pause_so_far_ms: float
    partial_tokens: list
    speech_rate_tps: float


class FeaturePipeline:
    def __init__(
        self,
        vad_model,
        asr_model,
        sample_rate: int = 16000,
        frame_ms: float = 100,
        asr_refresh_every_n_frames: int = 5,
        max_tokens: int = 20,
    ):
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.asr_refresh_every_n_frames = asr_refresh_every_n_frames
        self.max_tokens = max_tokens

        self._audio_so_far = np.zeros(0, dtype=np.float32)  # past-only by construction: step() only appends
        self._vad_leftover = np.zeros(0, dtype=np.float32)
        self._pause_so_far_ms = 0.0
        self._all_tokens: list = []
        self._frame_index = 0

        self.vad_model.reset_states()

    def _update_vad(self, chunk: np.ndarray) -> bool:
        """Feeds `chunk` (plus any leftover samples below the 512-sample
        window Silero VAD requires) through the VAD model. Returns whether
        any sub-window in this frame was classified as speech."""
        vad_input = np.concatenate([self._vad_leftover, chunk])
        n_full_windows = len(vad_input) // VAD_FRAME_SAMPLES

        any_speech = False
        for i in range(n_full_windows):
            window = vad_input[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            prob = self.vad_model(torch.from_numpy(window), self.sample_rate).item()
            if prob >= VAD_SPEECH_PROB_THRESHOLD:
                any_speech = True

        self._vad_leftover = vad_input[n_full_windows * VAD_FRAME_SAMPLES :]
        return any_speech

    def _maybe_refresh_asr(self) -> None:
        if self._frame_index % self.asr_refresh_every_n_frames != 0:
            return
        # `self._audio_so_far` is everything step() has ever received up to
        # and including the current frame -- never anything beyond it.
        segments, _info = self.asr_model.transcribe(self._audio_so_far, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        self._all_tokens = text.split()

    def step(self, chunk: np.ndarray) -> FrameFeatures:
        if len(chunk) != self.frame_samples:
            raise ValueError(f"expected a {self.frame_samples}-sample chunk, got {len(chunk)}")
        # Silero VAD's JIT model is float32-only and errors on float64
        # ("expected Double but found Float" is the real message despite
        # naming Float32 "Float") -- cast once here so every caller's audio
        # dtype (float32 or float64 from soundfile, either is common) works,
        # instead of relying on each caller to remember to load as float32.
        chunk = np.asarray(chunk, dtype=np.float32)

        self._frame_index += 1
        self._audio_so_far = np.concatenate([self._audio_so_far, chunk])

        is_speech = self._update_vad(chunk)
        if is_speech:
            self._pause_so_far_ms = 0.0
        else:
            self._pause_so_far_ms += self.frame_ms

        self._maybe_refresh_asr()

        elapsed_s = self._frame_index * self.frame_ms / 1000
        speech_rate_tps = len(self._all_tokens) / elapsed_s if elapsed_s > 0 else 0.0

        return FrameFeatures(
            frame_index=self._frame_index,
            t_ms=self._frame_index * self.frame_ms,
            pitch_hz=estimate_pitch(chunk, self.sample_rate),
            energy_rms=rms_energy(chunk),
            pause_so_far_ms=self._pause_so_far_ms,
            partial_tokens=self._all_tokens[-self.max_tokens :],
            speech_rate_tps=round(speech_rate_tps, 3),
        )


def run_pipeline(vad_model, asr_model, audio: np.ndarray, **pipeline_kwargs) -> list:
    """Convenience: chunk `audio` into frame_samples-sized pieces (dropping
    any trailing partial frame) and step() through all of them. Used by
    both normal feature extraction and the leakage audit script."""
    pipeline = FeaturePipeline(vad_model, asr_model, **pipeline_kwargs)
    n_frames = len(audio) // pipeline.frame_samples
    return [
        pipeline.step(audio[i * pipeline.frame_samples : (i + 1) * pipeline.frame_samples])
        for i in range(n_frames)
    ]
