"""Pydantic contracts for every inter-stage boundary in the pipeline:

  audio capture -> [AudioFrame] -> VAD -> [VadEvent] -> endpointer ->
  [EndpointDecision] -> ASR -> [AsrResult] -> LLM -> [LlmResponseChunk] ->
  TTS -> [TtsAudioChunk] -> audio playback

Each model is the contract for what crosses that boundary — structural and
type validity only. Audio-signal-quality checks (clipping, DC offset,
duration bounds) live in audio_checks.py since they need to inspect raw
sample values, not just declare a shape.
"""

from pydantic import BaseModel, Field, field_validator

SUPPORTED_SAMPLE_RATE = 16000


class AudioFrame(BaseModel):
    """Capture -> VAD boundary: one raw audio chunk."""

    session_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    sample_rate: int
    samples: list[float] = Field(min_length=1)
    t_captured: float

    @field_validator("sample_rate")
    @classmethod
    def _known_sample_rate(cls, v: int) -> int:
        if v != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"unsupported sample_rate {v}, expected {SUPPORTED_SAMPLE_RATE}")
        return v


class VadEvent(BaseModel):
    """VAD -> endpointer boundary: per-frame speech/silence decision."""

    session_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    is_speech: bool
    speech_prob: float = Field(ge=0.0, le=1.0)
    t_event: float


class EndpointDecision(BaseModel):
    """Endpointer -> ASR boundary: is this turn complete?"""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    turn_complete: bool
    confidence: float = Field(ge=0.0, le=1.0)
    segment_start_frame: int = Field(ge=0)
    segment_end_frame: int = Field(ge=0)
    t_decision: float

    @field_validator("segment_end_frame")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        start = info.data.get("segment_start_frame")
        if start is not None and v < start:
            raise ValueError(f"segment_end_frame {v} precedes segment_start_frame {start}")
        return v


class AsrResult(BaseModel):
    """ASR -> LLM boundary: transcript for the completed turn."""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    transcript: str
    is_final: bool
    confidence: float = Field(ge=0.0, le=1.0)
    language: str = Field(min_length=2)
    t_result: float

    @field_validator("transcript")
    @classmethod
    def _final_transcript_not_empty(cls, v: str, info) -> str:
        if info.data.get("is_final") and not v.strip():
            raise ValueError("final ASR result has an empty transcript")
        return v


class LlmResponseChunk(BaseModel):
    """LLM -> TTS boundary: one streamed chunk of the response."""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    text: str
    token_index: int = Field(ge=0)
    is_sentence_boundary: bool
    is_final: bool
    t_chunk: float


class TtsAudioChunk(BaseModel):
    """TTS -> audio playback boundary: one chunk of synthesized audio."""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    sample_rate: int
    samples: list[float] = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    is_first_chunk: bool
    t_chunk: float

    @field_validator("sample_rate")
    @classmethod
    def _known_sample_rate(cls, v: int) -> int:
        if v != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"unsupported sample_rate {v}, expected {SUPPORTED_SAMPLE_RATE}")
        return v
