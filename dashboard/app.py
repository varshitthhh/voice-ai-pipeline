"""Phase 7: Streamlit dashboard.

Reads the real trace data that already exists in this repo (traces/*.jsonl,
outputs/*.csv) -- nothing here is synthesized for display. Where the
underlying data doesn't exist yet (per-turn transcript text isn't part of
the TurnTrace schema; see src/tracing/trace.py), the panel says so rather
than fabricating a placeholder that looks real.

Run:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tracing import read_jsonl  # noqa: E402

TRACE_DIR = ROOT / "traces"
OUTPUTS_DIR = ROOT / "outputs"

STAGE_COLORS = {
    "endpointing_ms": "#4C72B0",
    "asr_final_ms": "#DD8452",
    "llm_ttft_ms": "#55A868",
    "tts_first_chunk_ms": "#C44E52",
}
STAGE_LABELS = {
    "endpointing_ms": "Endpointing",
    "asr_final_ms": "ASR final",
    "llm_ttft_ms": "LLM TTFT",
    "tts_first_chunk_ms": "TTS first chunk",
}
BUDGET_MS = {
    "endpointing_ms": (300, 700),
    "asr_final_ms": (80, 150),
    "llm_ttft_ms": (90, 300),
    "tts_first_chunk_ms": (40, 90),
    "e2e_ms": (510, 1240),
}

st.set_page_config(page_title="Voice AI Pipeline Dashboard", layout="wide")
st.title("Voice AI Pipeline -- Learned Endpointing")
st.caption(
    "Every number on this page is read from a real trace/output file already in the repo. "
    "Nothing is simulated for display."
)


# ---------------------------------------------------------------------------
# Sidebar: trace source + run control
# ---------------------------------------------------------------------------

st.sidebar.header("Trace source")
trace_files = sorted(TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
if not trace_files:
    st.sidebar.warning(f"No .jsonl files in {TRACE_DIR}")
    st.stop()

trace_choice = st.sidebar.selectbox("Session trace file", trace_files, format_func=lambda p: p.name)

st.sidebar.header("Run control")
st.sidebar.caption(
    "Regenerates traces/e2e_phase4_4.jsonl by re-running the real Phase 4.4 pipeline "
    "(scripts/run_e2e_traces.py) -- real ASR/TTS compute, several minutes on CPU."
)
if "e2e_process" not in st.session_state:
    st.session_state.e2e_process = None

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("Start", disabled=st.session_state.e2e_process is not None):
    st.session_state.e2e_process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_e2e_traces.py")], cwd=str(ROOT)
    )
if col_stop.button("Stop", disabled=st.session_state.e2e_process is None):
    if st.session_state.e2e_process is not None:
        st.session_state.e2e_process.terminate()
    st.session_state.e2e_process = None

if st.session_state.e2e_process is not None:
    ret = st.session_state.e2e_process.poll()
    if ret is None:
        st.sidebar.info("Running...")
    else:
        st.sidebar.success(f"Finished (exit code {ret}). Reload the trace file above to see new data.")
        st.session_state.e2e_process = None


# ---------------------------------------------------------------------------
# Load traces
# ---------------------------------------------------------------------------

@st.cache_data
def load_traces(path_str: str, mtime: float) -> list:
    """mtime busts the cache when the underlying file changes (e.g. after a re-run)."""
    return list(read_jsonl(Path(path_str)))


traces = load_traces(str(trace_choice), trace_choice.stat().st_mtime)
st.sidebar.metric("Turns loaded", len(traces))

if not traces:
    st.warning("Trace file is empty.")
    st.stop()


# ---------------------------------------------------------------------------
# Summary stats vs. budget
# ---------------------------------------------------------------------------

st.header("Latency summary vs. budget")

deltas_df = pd.DataFrame([tr.deltas_ms() for tr in traces])
summary_rows = []
for stage, (lo, hi) in BUDGET_MS.items():
    vals = deltas_df[stage].dropna()
    if vals.empty:
        continue
    p50, p95 = vals.quantile(0.50), vals.quantile(0.95)
    summary_rows.append({
        "stage": STAGE_LABELS.get(stage, stage),
        "budget_ms": f"{lo}-{hi}",
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "within_budget_p50": lo <= p50 <= hi,
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Latency waterfall
# ---------------------------------------------------------------------------

st.header("Latency waterfall")
n_show = st.slider("Turns to show", min_value=1, max_value=len(traces), value=min(30, len(traces)))
shown = traces[:n_show]

fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(shown))))
for row, tr in enumerate(shown):
    deltas = tr.deltas_ms()
    left = 0.0
    for stage in STAGE_COLORS:
        width = deltas.get(stage) or 0.0
        ax.barh(row, width, left=left, color=STAGE_COLORS[stage], edgecolor="white", height=0.7)
        left += width
ax.set_yticks(range(len(shown)))
ax.set_yticklabels([tr.turn_id for tr in shown])
ax.invert_yaxis()
ax.set_xlabel("ms since t_speech_start")
ax.set_title(f"Session {shown[0].session_id}")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS.values()]
ax.legend(handles, [STAGE_LABELS[s] for s in STAGE_COLORS], loc="lower right", fontsize=8)
fig.tight_layout()
st.pyplot(fig)


# ---------------------------------------------------------------------------
# Turn detail
# ---------------------------------------------------------------------------

st.header("Turn detail")
turn_ids = [tr.turn_id for tr in traces]
selected_turn = st.selectbox("Turn", turn_ids)
tr = next(t for t in traces if t.turn_id == selected_turn)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Stage timestamps")
    st.json({f: getattr(tr, f) for f in ("t_speech_start", "t_vad_trigger", "t_endpoint_decision", "t_asr_final", "t_llm_first_token", "t_tts_first_chunk", "t_audio_out")})
with col_b:
    st.subheader("Stage deltas (ms)")
    st.json(tr.deltas_ms())

st.caption(
    "No transcript panel: TurnTrace (src/tracing/trace.py) records stage timestamps only, not "
    "ASR/LLM text -- there's no real transcript data to show here without changing what Phase 0.2's "
    "tracing schema captures and re-running Phase 4.4 to regenerate traces with it."
)


# ---------------------------------------------------------------------------
# Endpoint decisions over the audio timeline (learned endpointer)
# ---------------------------------------------------------------------------

st.header("Endpoint decisions over the audio timeline")

CORPUS_OPTIONS = {
    "real_ami": {
        "scenarios_path": ROOT / "data" / "turn_taking" / "real_ami" / "scenarios.jsonl",
        "checkpoint_path": OUTPUTS_DIR / "turn_taking_model_real.pt",
        "caption": (
            "Real AMI Meeting Corpus audio (cc-by-4.0) -- real spontaneous speech, real "
            "reconstructed pause timing. Not e-commerce, not CANDOR -- see STATUS.md / "
            "scripts/prepare_ami_turntaking.py for the honest domain-gap discussion."
        ),
    },
    "synthetic": {
        "scenarios_path": ROOT / "data" / "turn_taking" / "synthetic" / "scenarios.jsonl",
        "checkpoint_path": OUTPUTS_DIR / "turn_taking_model.pt",
        "caption": (
            "Synthetic SLURP-splice corpus (real speech clips, scripted silence gaps) -- the "
            "original stand-in, kept here for comparison against the real-data model above."
        ),
    },
}
available_corpora = {
    k: v for k, v in CORPUS_OPTIONS.items()
    if v["scenarios_path"].exists() and v["checkpoint_path"].exists()
}

if not available_corpora:
    st.info(
        "No scenario corpus + checkpoint pair found -- run scripts/prepare_ami_turntaking.py "
        "(real) or the training notebook (synthetic) first."
    )
else:
    corpus_choice = st.radio("Corpus", list(available_corpora.keys()), horizontal=True)
    cfg = available_corpora[corpus_choice]
    st.caption(cfg["caption"])
    scenarios_path = cfg["scenarios_path"]
    checkpoint_path = cfg["checkpoint_path"]

    import json

    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f]
    scenario_ids = [s["scenario_id"] for s in scenarios]
    selected_scenario_id = st.selectbox("Scenario", scenario_ids)
    threshold = st.slider("Firing threshold", 0.0, 1.0, 0.7, 0.05)

    @st.cache_resource
    def load_models(checkpoint_path_str: str):
        # checkpoint_path_str is part of the cache key -- without it, switching the Corpus
        # radio button above would silently keep serving the FIRST-loaded model against the
        # second corpus's scenarios, since Streamlit only differentiates cache_resource by
        # the function's own arguments, not variables closed over from the surrounding scope.
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        from ab_compare_endpointers import load_model_from_checkpoint

        vad_model = load_silero_vad(onnx=False)
        model = load_model_from_checkpoint(Path(checkpoint_path_str))
        asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        return vad_model, model, asr_model

    @st.cache_data
    def compute_trace(scenario_id: str, checkpoint_path_str: str):
        from ab_compare_endpointers import run_learned_trace

        scenario = next(s for s in scenarios if s["scenario_id"] == scenario_id)
        audio, sr = sf.read(ROOT / scenario["audio_path"], dtype="float64")
        vad_model, model, asr_model = load_models(checkpoint_path_str)
        trace = run_learned_trace(model, vad_model, asr_model, audio)
        return audio, sr, scenario, trace

    if st.button("Compute trace"):
        with st.spinner("Running VAD + ASR + GRU inference..."):
            audio, sr, scenario, trace = compute_trace(selected_scenario_id, str(checkpoint_path))

        fired_frame = next((r for r in trace if r["pause_so_far_ms"] > 0 and r["prob"] >= threshold), None)

        fig2, (ax_wave, ax_prob) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        t_audio = np.arange(len(audio)) / sr * 1000
        ax_wave.plot(t_audio, audio, linewidth=0.5, color="#4C72B0")
        ax_wave.axvline(scenario["splice_sample"] / sr * 1000, color="gray", linestyle="--", label="true pause start")
        ax_wave.set_ylabel("amplitude")
        ax_wave.set_title(f"{scenario['scenario_id']} ({scenario['label']})")
        ax_wave.legend(fontsize=8)

        t_prob = [r["t_ms"] for r in trace]
        p_prob = [r["prob"] for r in trace]
        ax_prob.plot(t_prob, p_prob, color="#DD8452", marker="o", markersize=3)
        ax_prob.axhline(threshold, color="gray", linestyle="--", label=f"threshold={threshold}")
        if fired_frame:
            ax_prob.axvline(fired_frame["t_ms"], color="#C44E52", label=f"fired @ {fired_frame['t_ms']:.0f}ms")
        ax_prob.set_ylabel("P(turn_complete)")
        ax_prob.set_xlabel("ms")
        ax_prob.legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

        if fired_frame:
            st.success(f"Fired at t={fired_frame['t_ms']:.0f}ms, prob={fired_frame['prob']:.3f}")
        else:
            st.warning("Never fired at this threshold.")
