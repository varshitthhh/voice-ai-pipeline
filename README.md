# Streaming Voice Agent — Learned Endpointing

**Thesis:** Conversational latency in voice agents is dominated by endpointing, not inference. This project builds a real-time voice agent and shows that a learned turn-taking model beats fixed-threshold VAD on the latency/interruption tradeoff.

**Domain:** E-commerce customer support.

## 1. Latency Budget

Before any code is written, this section fixes the per-segment millisecond allocation that every later phase is measured against. "Conversational latency" is defined as the interval from **the user's last speech frame** to **the first audible chunk of the agent's spoken response** — the span a human perceives as "did it hear me and respond."

That interval is decomposed into four sequential stages, each owned by a different component in the pipeline (Silero VAD v5 → learned endpointer → faster-whisper distil-large-v3 INT8 → Qwen2.5-7B-Instruct AWQ on vLLM → Kokoro-82M).

### 1.1 Per-segment allocation

| Stage | Component | Budget (ms) | Measured from → to |
|---|---|---|---|
| Endpointing | Silero VAD v5 + learned endpointer | 300 – 700 | `t_speech_start` → `t_endpoint_decision` |
| ASR final | faster-whisper distil-large-v3 INT8 | 80 – 150 | `t_endpoint_decision` → `t_asr_final` |
| LLM TTFT | Qwen2.5-7B-Instruct AWQ (vLLM) | 90 – 300 | `t_asr_final` → `t_llm_first_token` |
| TTS first chunk | Kokoro-82M | 40 – 90 | `t_llm_first_token` (or first sentence boundary) → `t_tts_first_chunk` |

These four spans are exactly the fields the tracing harness (Phase 0.2) records per turn, so every subsequent benchmark ties directly back to this table.

### 1.2 End-to-end target

| | Best case | Worst case |
|---|---|---|
| Sum of stage minimums / maximums | 510 ms | 1,240 ms |
| **Target p50** | **≤ 700 ms** | |
| **Target p95** | **≤ 1,100 ms** | |

500 ms is the rough threshold at which a spoken exchange starts to feel like a real conversation rather than a call-and-response; above ~1.2s, users perceive a stall. The p50/p95 targets above are the numbers Phase 6 (eval suite) and Phase 4.4 (E2E measurement) are graded against.

### 1.3 Why endpointing dominates the budget

Endpointing owns **300–700 ms of the 510–1,240 ms budget — roughly 55–60% of total latency** in both the best and worst case, more than the other three stages combined. It is also the only stage with no compute floor: ASR, LLM, and TTS latency are bounded by model size and hardware and improve predictably with better quantization or a faster GPU. Endpointing latency is instead bounded by an unsolved decision problem — *has the user actually finished speaking* — which a fixed silence threshold answers by waiting, not by understanding. Fixed-threshold VAD trades latency for false interruptions along a single knob; it cannot exploit semantic or prosodic cues to shorten the wait on turns that are actually complete. This is the gap Phase 3's learned endpointer targets, and it is why the project's headline result is an endpointing latency delta, not an inference speedup.

### 1.4 Measurement ownership

Per the compute plan, all headline latency numbers reported against this budget are measured on the **24GB GPU** (serving), never on Colab T4 (training/sweeps only) or the Zenbook (orchestrator/dashboard dev). This keeps every number in this doc, and every number derived from it later, comparable across phases.

### 1.5 Out of scope for this section

No models are loaded and no code is written to produce this budget — the ranges above come from published benchmarks and the component choices already fixed in the stack. Phase 0.2 (tracing harness) instruments these four spans; Phase 0.3 (hardware baseline) validates the ranges against real 24GB/T4 measurements before Phase 2 component benchmarking begins.
