"""Phase 4.4: end-to-end latency measurement across the full pipeline.

Wires together every stage built so far -- VAD (Phase 3.1), the fixed-
threshold endpointer (Phase 3.1), ASR (Phase 4.1's model, used directly
rather than through StreamingASR's segment-detection machinery since the
endpoint decision already tells us exactly where the segment ends), a
GPU-only LLM stage (TODO-marked, calibrated mock timing), and TTS (Phase
2.3/4.2) -- and records one Phase 0.2 TurnTrace per turn.

Each turn's timeline is composed from two different clock sources, chained
together deliberately, not by accident:

  - t_speech_start / t_vad_trigger / t_endpoint_decision: derived from
    AUDIO SAMPLE POSITIONS, not real wall-clock waiting. A fixed-threshold
    endpointer's latency is a deliberate silence wait (500ms of required
    silence, say) -- that's a real, honest number, but there's no reason
    to actually asyncio.sleep(0.5) 100 times to "measure" it when the wait
    duration is dictated entirely by the threshold constant. This matches
    scripts/baseline_fixed_threshold_vad.py's own methodology exactly
    (including its fix for VADIterator's 'end' event reporting a back-
    dated acoustic timestamp rather than the real decision time).

  - t_asr_final / t_llm_first_token / t_tts_first_chunk / t_audio_out:
    each stage's real measured (ASR, TTS) or realistically-calibrated mock
    (LLM) compute DURATION is chained onto the endpoint decision's virtual
    timestamp. These durations come from actually running faster-whisper
    and Piper -- not simulated -- except the LLM stage.

TODO(gpu-required): the LLM stage uses mock timing sampled from the
README Section 1 budget (TTFT 90-300ms) because vLLM cannot run on this
machine at all (see scripts/benchmark_llm.py's docstring for why). Every
other stage below is real compute, chained onto a virtual endpoint-
decision clock that itself matches Phase 3.1's validated methodology.
"""

import random
import time
from typing import Optional

import numpy as np
from silero_vad import VADIterator

from tracing import Tracer

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # Silero VAD v5's required window at 16kHz
DEFAULT_ENDPOINT_THRESHOLD_MS = 500  # fixed-threshold baseline, per Phase 3.1

LLM_TTFT_RANGE_S = (0.090, 0.300)  # TODO(gpu-required): README Section 1 budget, not a real vLLM measurement


def find_endpoint(vad_model, audio: np.ndarray, threshold_ms: int = DEFAULT_ENDPOINT_THRESHOLD_MS) -> tuple:
    """Runs a fresh VADIterator over `audio`. Returns (speech_start_sample,
    vad_trigger_sample, endpoint_decision_sample), all None if no endpoint
    was found.

    speech_start_sample: VADIterator's own 'start' event sample (applies
    speech_pad_ms backward already) -- an approximation of the utterance's
    true acoustic onset.

    endpoint_decision_sample: the PROCESSING POSITION (frame index *
    VAD_FRAME_SAMPLES) at the moment the 'end' event was actually emitted
    -- not the event dict's own reported sample, which VADIterator
    deliberately back-dates to when silence began, not when the
    min_silence_duration_ms wait was satisfied. Confirmed the hard way in
    scripts/baseline_fixed_threshold_vad.py; same fix applied here.

    vad_trigger_sample: the moment silence actually BEGAN (i.e. the user
    stopped talking) -- this is what kicks off the endpointing wait, so
    `endpointing_ms = t_endpoint_decision - t_vad_trigger` is the fixed-
    threshold wait duration itself (the 300-700ms README budget), not the
    wait PLUS however long the user had been talking before that. This is
    not directly exposed by VADIterator (its `temp_end` bookkeeping is
    internal), but is exact: silence started `threshold_ms` of processing
    time before the 'end' event necessarily fired, by construction of
    VADIterator's own min_silence_duration_ms logic. So it's backed out
    by simple subtraction rather than needing our own detector.
    """
    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=threshold_ms)
    n_frames = len(audio) // VAD_FRAME_SAMPLES

    speech_start_sample = None
    endpoint_decision_sample = None

    for i in range(n_frames):
        chunk = audio[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
        event = vad(chunk, return_seconds=False)
        if event is None:
            continue
        if "start" in event and speech_start_sample is None:
            speech_start_sample = event["start"]
        if "end" in event and speech_start_sample is not None and endpoint_decision_sample is None:
            endpoint_decision_sample = (i + 1) * VAD_FRAME_SAMPLES
            break

    if endpoint_decision_sample is None:
        return speech_start_sample, None, None

    threshold_samples = int(SAMPLE_RATE * threshold_ms / 1000)
    vad_trigger_sample = max(speech_start_sample, endpoint_decision_sample - threshold_samples)

    return speech_start_sample, vad_trigger_sample, endpoint_decision_sample


def run_turn_trace(
    turn_id: str,
    user_audio: np.ndarray,
    response_first_sentence: str,
    vad_model,
    asr_model,
    tts_synthesize_fn,
    tracer: Tracer,
    endpoint_threshold_ms: int = DEFAULT_ENDPOINT_THRESHOLD_MS,
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    """Processes one turn through every pipeline stage and writes a
    TurnTrace row. `tts_synthesize_fn(text) -> None` should be a
    synchronous callable (Piper's real synthesize(), matching Phase 2.3).
    Returns the resulting TurnTrace's deltas_ms() dict, or None if VAD
    never found an endpoint in `user_audio` (shouldn't happen given the
    trailing silence every turn's audio is built with)."""
    rng = rng or random

    speech_start_sample, vad_trigger_sample, endpoint_sample = find_endpoint(vad_model, user_audio, endpoint_threshold_ms)
    if endpoint_sample is None:
        return None

    tracer.start_turn(turn_id)
    t0 = time.monotonic()  # anchor for the virtual audio-time clock

    t_speech_start = t0 + speech_start_sample / SAMPLE_RATE
    t_vad_trigger = t0 + vad_trigger_sample / SAMPLE_RATE
    t_endpoint_decision = t0 + endpoint_sample / SAMPLE_RATE
    tracer.mark("t_speech_start", t_speech_start)
    tracer.mark("t_vad_trigger", t_vad_trigger)
    tracer.mark("t_endpoint_decision", t_endpoint_decision)

    # ASR: real compute, chained onto the virtual endpoint-decision timestamp.
    segment_audio = user_audio[:endpoint_sample]
    asr_t0 = time.monotonic()
    segments, _info = asr_model.transcribe(segment_audio, beam_size=1)
    list(segments)  # force generator consumption -- this IS the decode cost being timed
    asr_elapsed_s = time.monotonic() - asr_t0
    t_asr_final = t_endpoint_decision + asr_elapsed_s
    tracer.mark("t_asr_final", t_asr_final)

    # LLM: TODO(gpu-required) -- mock timing sampled from the README budget.
    llm_ttft_s = rng.uniform(*LLM_TTFT_RANGE_S)
    t_llm_first_token = t_asr_final + llm_ttft_s
    tracer.mark("t_llm_first_token", t_llm_first_token)

    # TTS: real compute for the first sentence only, chained onto the LLM timestamp.
    tts_t0 = time.monotonic()
    tts_synthesize_fn(response_first_sentence)
    tts_elapsed_s = time.monotonic() - tts_t0
    t_tts_first_chunk = t_llm_first_token + tts_elapsed_s
    tracer.mark("t_tts_first_chunk", t_tts_first_chunk)

    # Buffer flush to the (real or simulated) speaker: treated as negligible.
    t_audio_out = t_tts_first_chunk
    tracer.mark("t_audio_out", t_audio_out)

    trace = tracer.end_turn()
    return trace.deltas_ms()
