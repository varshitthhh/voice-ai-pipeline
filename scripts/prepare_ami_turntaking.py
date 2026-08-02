"""Phase 1.1 (partial, real-data stand-in): turn-taking scenarios built from
REAL AMI Meeting Corpus audio and REAL observed pause timing.

This is NOT CANDOR and NOT e-commerce customer support -- CANDOR access is
still pending manual review, and the domain gap to AMI (corporate meeting
talk, not two-party support calls) is real and stated plainly here, not
hidden. What this buys over the fully-synthetic corpus
(prepare_synthetic_turn_taking_eval.py) is real spontaneous speech and real,
naturally-distributed pause durations, instead of hand-picked silence
lengths spliced onto scripted SLURP commands.

Dataset: edinburghcstr/ami, "ihm" (individual headset mic) config.
Verified before writing this script (not assumed):
  - License: cc-by-4.0 (confirmed via the HF Hub API's dataset tags, not
    just the dataset card text).
  - Fully open: no gating, no access request, downloadable immediately via
    the datasets-server /rows API (same mechanism prepare_slurp_eval.py
    uses) -- no huge parquet shard download required.
  - Schema has `begin_time`/`end_time` (meeting-relative, float seconds)
    and `speaker_id` per utterance -- confirmed by direct API query. This
    is what makes real pause reconstruction possible from a per-utterance-
    segmented dataset: sort one meeting's utterances by begin_time, and the
    gap between consecutive utterances is a real observed pause, labeled
    MID_TURN (same speaker resumes) or a turn-boundary (different speaker
    starts) by comparing speaker_id.
  - Real gap-duration distribution, confirmed on a FULL single meeting
    (EN2001a, 1675 utterances, all fetched -- an earlier check against only
    the first 500 rows gave a misleadingly large p50, because that window
    didn't contain every utterance in the meeting): p50=170ms, p90=2800ms,
    945/1129 gaps under 2s -- this is a plausible natural-pause
    distribution, not an artifact.
  - IMPORTANT: only a FULLY fetched meeting's utterances may be used for
    gap reconstruction. A partial window (e.g. the first N rows of a much
    longer meeting) will be missing utterances that fall in those gaps and
    silently produces wrong pause durations -- this script always fetches
    a meeting to exhaustion before using it, never an arbitrary row slice.

Scenario semantics (matches docs/turn_taking_label_schema.md's TRUE_END /
MID_TURN, and the same JSONL schema as
data/turn_taking/synthetic/scenarios.jsonl, so this is a drop-in
alternative --labels input for baseline_fixed_threshold_vad.py /
ab_compare_endpointers.py / eval.py):
    MID_TURN:  same speaker before and after the gap -- a real within-turn
               pause, the hard negative a fixed-threshold endpointer cuts
               off.
    TRUE_END:  different speaker after the gap -- the first speaker's turn
               genuinely ended. This is an approximation: AMI is a
               multi-party meeting, not a two-party support call, so
               "someone else starts talking" isn't identical to "the agent
               may now respond" -- but it's the closest real proxy
               available for "the turn is actually over."
    Overlapping speech (b.begin_time < a.end_time, real cross-talk, common
    in meetings) is excluded entirely -- it isn't a pause.

Run:
    python scripts/prepare_ami_turntaking.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_AUDIO_DIR = ROOT / "data" / "turn_taking" / "real_ami" / "audio"
OUT_LABELS_PATH = ROOT / "data" / "turn_taking" / "real_ami" / "scenarios.jsonl"

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "edinburghcstr/ami"
CONFIG = "ihm"
SPLIT = "train"
PAGE_SIZE = 100

SAMPLE_RATE = 16000
N_MEETINGS = 3
MAX_SCENARIOS_PER_MEETING = 200
MIN_UTTERANCE_S = 0.3         # drop near-silence/artifact segments
MIN_GAP_MS = 0                # immediate resumption is a valid MID_TURN example
MAX_GAP_MS = 5000             # beyond this, treat as a meeting break, not a turn-taking decision point
RANDOM_SEED = 0


def get_with_retry(url: str, params: dict = None, max_retries: int = 10, timeout: int = 30):
    """HF's datasets-server API rate-limits (429) under sustained request
    volume -- confirmed directly against this exact endpoint, twice: a
    6-attempt/63s-total backoff still weren't enough on the second run
    (this project's own manual verification during research already spent
    a lot of the request budget before this script's first real run).
    Capped exponential backoff up to 10 attempts / ~60s per wait, honoring
    Retry-After when the server sends one."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait_s = float(resp.headers.get("Retry-After", min(2 ** attempt, 60)))
        print(f"  429 rate-limited, waiting {wait_s:.0f}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait_s)
    raise SystemExit(f"gave up after {max_retries} retries against {url} -- still rate-limited")


def fetch_page(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    resp = get_with_retry(ROWS_URL, params=params)
    return resp.json()


def fetch_full_meetings(n_meetings: int, max_pages: int = 400) -> dict:
    """Pages from the start of the split, bucketing utterances by
    meeting_id. The split is stored meeting-grouped (confirmed: 1700+
    consecutive rows shared one meeting_id in a direct check), so paging
    linearly and stopping once n_meetings+1 distinct meetings have
    appeared -- then dropping that (n_meetings+1)-th, necessarily
    incomplete, meeting -- yields n_meetings COMPLETE meetings without
    downloading the full 15GB config."""
    by_meeting = defaultdict(list)
    offset = 0
    for _ in range(max_pages):
        page = fetch_page(offset, PAGE_SIZE)
        rows = page["rows"]
        if not rows:
            break
        for r in rows:
            by_meeting[r["row"]["meeting_id"]].append(r["row"])
        offset += len(rows)
        if len(by_meeting) > n_meetings:
            break
        print(f"  paged {offset} rows, {len(by_meeting)} meeting(s) seen so far")
        time.sleep(0.3)  # pace requests during discovery to avoid bursting the rate limit

    meeting_ids = list(by_meeting.keys())[:n_meetings]
    if len(meeting_ids) < n_meetings:
        print(f"WARNING: only found {len(meeting_ids)} complete meetings, wanted {n_meetings}")
    return {mid: by_meeting[mid] for mid in meeting_ids}


def download_audio(url: str) -> tuple:
    resp = get_with_retry(url)
    import io

    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float64")
    return audio, sr


def build_scenarios_for_meeting(meeting_id: str, utts: list) -> list:
    utts = [u for u in utts if (u["end_time"] - u["begin_time"]) >= MIN_UTTERANCE_S]
    utts.sort(key=lambda u: u["begin_time"])

    scenarios = []
    for i, (a, b) in enumerate(zip(utts, utts[1:])):
        gap_s = b["begin_time"] - a["end_time"]
        if gap_s < 0:
            continue  # overlapping speech -- real cross-talk, not a pause
        gap_ms = gap_s * 1000
        if not (MIN_GAP_MS <= gap_ms <= MAX_GAP_MS):
            continue

        same_speaker = a["speaker_id"] == b["speaker_id"]
        scenarios.append({
            "meeting_id": meeting_id,
            "pair_index": i,
            "label": "MID_TURN" if same_speaker else "TRUE_END",
            "pause_ms": round(gap_ms, 1),
            "clip_a_src": a["audio"][0]["src"],
            "clip_a_text": a["text"],
            "clip_a_speaker": a["speaker_id"],
            "clip_b_src": b["audio"][0]["src"] if same_speaker else None,
            "clip_b_text": b["text"] if same_speaker else None,
            "clip_b_speaker": b["speaker_id"] if same_speaker else None,
        })
        if len(scenarios) >= MAX_SCENARIOS_PER_MEETING:
            break

    return scenarios


def render_scenario(scenario: dict, idx: int) -> dict:
    """Downloads the real audio for one scenario and splices it exactly
    like prepare_synthetic_turn_taking_eval.py does -- clip_a + a REAL-
    duration silence gap (+ clip_b for MID_TURN) -- so this manifest is a
    drop-in --labels alternative for the existing eval scripts."""
    clip_a, sr_a = download_audio(scenario["clip_a_src"])
    assert sr_a == SAMPLE_RATE, f"clip_a is {sr_a}Hz, expected {SAMPLE_RATE}"
    splice_sample = len(clip_a)
    gap = np.zeros(int(SAMPLE_RATE * scenario["pause_ms"] / 1000), dtype=np.float64)

    if scenario["label"] == "MID_TURN":
        clip_b, sr_b = download_audio(scenario["clip_b_src"])
        assert sr_b == SAMPLE_RATE, f"clip_b is {sr_b}Hz, expected {SAMPLE_RATE}"
        audio = np.concatenate([clip_a, gap, clip_b])
        speech_b_start_sample = splice_sample + len(gap)
    else:
        audio = np.concatenate([clip_a, gap])
        speech_b_start_sample = None

    scenario_id = f"ami_{scenario['meeting_id']}_{idx:04d}"
    audio_path = OUT_AUDIO_DIR / f"{scenario_id}.wav"
    sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_16")

    return {
        "scenario_id": scenario_id,
        "label": scenario["label"],
        "pause_ms": scenario["pause_ms"],
        "splice_sample": splice_sample,
        "speech_b_start_sample": speech_b_start_sample,
        "audio_path": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
        "clip_a": f"ami_{scenario['meeting_id']}_{scenario['clip_a_speaker']}_{idx}a",
        "clip_b": f"ami_{scenario['meeting_id']}_{scenario['clip_b_speaker']}_{idx}b" if scenario["clip_b_speaker"] else None,
        # Provenance fields beyond the synthetic schema -- real source, not
        # invented data, and downstream code should be able to tell.
        "source": "ami",
        "meeting_id": scenario["meeting_id"],
        "clip_a_text": scenario["clip_a_text"],
        "clip_b_text": scenario["clip_b_text"],
    }


def main() -> None:
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"discovering {N_MEETINGS} complete meetings from {DATASET}/{CONFIG}...")
    meetings = fetch_full_meetings(N_MEETINGS)
    for mid, utts in meetings.items():
        span_min = (utts[-1]["end_time"] - utts[0]["begin_time"]) / 60 if utts else 0
        print(f"  {mid}: {len(utts)} utterances")

    all_scenarios = []
    for mid, utts in meetings.items():
        pairs = build_scenarios_for_meeting(mid, utts)
        print(f"  {mid}: {len(pairs)} usable turn-boundary pairs (gap in [{MIN_GAP_MS},{MAX_GAP_MS}]ms, no overlap)")
        all_scenarios.extend(pairs)

    if not all_scenarios:
        raise SystemExit("no usable scenarios found -- check meeting discovery / gap filters")

    print(f"\ndownloading + splicing {len(all_scenarios)} real scenarios (this makes one HTTP request per audio clip)...")
    rendered = []
    for i, s in enumerate(all_scenarios):
        rendered.append(render_scenario(s, i))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_scenarios)}")

    with open(OUT_LABELS_PATH, "w", encoding="utf-8") as f:
        for row in rendered:
            f.write(json.dumps(row) + "\n")

    n_true_end = sum(r["label"] == "TRUE_END" for r in rendered)
    n_mid_turn = len(rendered) - n_true_end
    print(f"\nwrote {len(rendered)} real scenarios ({n_true_end} TRUE_END-proxy, {n_mid_turn} MID_TURN) -> {OUT_LABELS_PATH}")
    print(f"audio -> {OUT_AUDIO_DIR}")


if __name__ == "__main__":
    main()
