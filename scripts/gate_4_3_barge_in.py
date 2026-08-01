"""Phase 4.3 gate: 20 consecutive barge-in interrupts.

Real components throughout except the LLM (vLLM is GPU/Linux-only, see
scripts/benchmark_llm.py): real Piper TTS synthesis, real Silero VAD-based
BargeInListener fed a real speech clip as the "user interrupting" signal,
real asyncio task cancellation. Only the LLM token stream is mocked.

Each cycle:
    1. Start an assistant turn (mock LLM -> real TTS -> simulated playback).
    2. Concurrently feed a real speech clip into BargeInListener, starting
       at a per-cycle delay that spans early/mid/late points across the 20
       cycles -- not just one fixed interruption point.
    3. The instant BargeInListener detects sustained speech, call
       orchestrator.interrupt().
    4. Assert: history committed correctly, no leftover TTS tasks, audio
       player idle, and -- the actual gate -- no orphaned asyncio tasks
       (diffed against a baseline snapshot after each cycle).

Run:
    python scripts/gate_4_3_barge_in.py
"""

import asyncio
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piper import PiperVoice
from silero_vad import load_silero_vad

from streaming_pipeline import BargeInListener, BargeInOrchestrator, ConversationHistory

PIPER_MODEL_PATH = ROOT / "data" / "tts" / "models" / "piper" / "en_US-lessac-medium.onnx"
INTERRUPT_AUDIO_PATH = ROOT / "data" / "turn_taking" / "synthetic" / "audio" / "synth_001.wav"

RESPONSE_TEXT = (
    "I'm sorry to hear your order hasn't arrived yet. "
    "Let me look into that for you right away. "
    "I can see it's currently in transit and should arrive within two days. "
    "Is there anything else I can help you with?"
)
PREFILL_DELAY_S = 0.1
DECODE_DELAY_S_PER_TOKEN = 0.015

N_CYCLES = 20
# Spans well before the first sentence finishes generating through well
# into mid-response playback -- not just one fixed interruption point.
INTERRUPT_DELAYS_S = [round(0.05 + (i % 10) * 0.28, 3) for i in range(N_CYCLES)]
MIC_FEED_CHUNK_SAMPLES = 512


class FakeOutput:
    def __init__(self, text):
        self.text = text


class FakeRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs


class FakeAsyncEngine:
    """Same duck-typed interface as vllm.AsyncLLMEngine / Phase 2.2's harness."""

    async def generate(self, prompt, sampling_params, request_id):
        text = ""
        await asyncio.sleep(PREFILL_DELAY_S)
        for i, word in enumerate(RESPONSE_TEXT.split(" ")):
            text += word if i == 0 else " " + word
            yield FakeRequestOutput([FakeOutput(text)])
            await asyncio.sleep(DECODE_DELAY_S_PER_TOKEN)


def make_tts_fn(voice: PiperVoice):
    async def synthesize(sentence: str) -> float:
        def _blocking():
            chunks = list(voice.synthesize(sentence))
            total_samples = sum(len(c.audio_float_array) for c in chunks)
            sample_rate = chunks[0].sample_rate if chunks else 16000
            return total_samples / sample_rate

        return await asyncio.to_thread(_blocking)

    return synthesize


async def run_cycle(cycle_idx: int, voice: PiperVoice, vad_model, interrupt_audio, interrupt_delay_s: float) -> dict:
    history = ConversationHistory()
    history.add_user_turn("(user's initial message)")
    orchestrator = BargeInOrchestrator(history, make_tts_fn(voice))
    listener = BargeInListener(vad_model, min_consecutive_speech_frames=3)

    turn_task = asyncio.create_task(orchestrator.run_turn(FakeAsyncEngine(), "(prompt)", None, f"cycle-{cycle_idx}"))

    async def mic_feed() -> None:
        await asyncio.sleep(interrupt_delay_s)
        n_chunks = len(interrupt_audio) // MIC_FEED_CHUNK_SAMPLES
        for i in range(n_chunks):
            chunk = interrupt_audio[i * MIC_FEED_CHUNK_SAMPLES : (i + 1) * MIC_FEED_CHUNK_SAMPLES]
            if listener.feed(chunk):
                return
            await asyncio.sleep(0.032)  # real-time pacing: 512 samples @ 16kHz = 32ms

    await mic_feed()  # blocks until barge-in detected (test audio always has real speech in it)
    heard = await orchestrator.interrupt()
    turn_result = await turn_task

    return {
        "cycle": cycle_idx,
        "interrupt_delay_s": interrupt_delay_s,
        "heard": heard,
        "turn_result": turn_result,
        "orchestrator": orchestrator,
        "history": history,
    }


async def main_async() -> bool:
    if not PIPER_MODEL_PATH.exists():
        raise SystemExit(f"missing {PIPER_MODEL_PATH} -- run scripts/prepare_tts_models.py first")
    if not INTERRUPT_AUDIO_PATH.exists():
        raise SystemExit(f"missing {INTERRUPT_AUDIO_PATH} -- run scripts/prepare_synthetic_turn_taking_eval.py first")

    voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    list(voice.synthesize("warm up."))  # warmup, excluded from timing

    interrupt_audio, sr = sf.read(INTERRUPT_AUDIO_PATH, dtype="float32")
    assert sr == 16000

    vad_model = load_silero_vad(onnx=False)

    await asyncio.sleep(0)  # let the loop settle before taking the baseline snapshot
    baseline_tasks = asyncio.all_tasks() - {asyncio.current_task()}

    all_ok = True
    results = []

    for i in range(N_CYCLES):
        cycle = await run_cycle(i, voice, vad_model, interrupt_audio, INTERRUPT_DELAYS_S[i])
        results.append(cycle)

        orch = cycle["orchestrator"]
        ok = True

        turn_result_ok = cycle["turn_result"] is None
        ok = ok and turn_result_ok

        tts_cleared_ok = orch._tts_tasks == []
        ok = ok and tts_cleared_ok

        player_idle_ok = orch.player.is_idle()
        ok = ok and player_idle_ok

        # history: 1 user turn + (1 assistant turn IFF something was actually heard)
        expected_turns = 1 + (1 if cycle["heard"] else 0)
        history_ok = len(cycle["history"].turns) == expected_turns
        ok = ok and history_ok

        await asyncio.sleep(0.02)  # let any straggling callbacks settle before snapshotting
        current_tasks = asyncio.all_tasks() - {asyncio.current_task()}
        leaked = current_tasks - baseline_tasks
        pending_leaked = [t for t in leaked if not t.done()]
        no_orphans_ok = not pending_leaked
        ok = ok and no_orphans_ok

        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] cycle {i:2d} delay={cycle['interrupt_delay_s']:.2f}s "
            f"heard={cycle['heard']!r:60s} orphans={len(pending_leaked)}"
        )
        if not ok:
            print(f"       turn_result_ok={turn_result_ok} tts_cleared_ok={tts_cleared_ok} "
                  f"player_idle_ok={player_idle_ok} history_ok={history_ok} (expected {expected_turns} turns, got {len(cycle['history'].turns)})")
        all_ok = all_ok and ok

    return all_ok


def main() -> None:
    ok = asyncio.run(main_async())
    print("\nGATE PASS: 20/20 consecutive interrupts, no state corruption, no orphaned tasks" if ok else "\nGATE FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
