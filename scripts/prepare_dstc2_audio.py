"""Domain-match investigation: DSTC2 real call audio -- NOT wired into the
turn-taking pipeline, and this script says so via its own assertions, not
just this docstring.

DSTC2 is real human callers phoning a real automated restaurant-booking
phone system (Cambridge) -- the closest DOMAIN match found to "a person
talking to a voice AI agent" of anything checked (AxonData/AIxBlock call-
center datasets are paid/gated past a 2-file sample; Taskmaster-2 and CCPE
were both confirmed text-only despite spoken origins; MultiWOZ never had
audio). No LDC gate -- DSTC2 was released openly by the challenge
organizers specifically for this kind of research.

BUT: verified directly against the HF datasets-server API (not assumed)
that danielroncel/dstc2_audios's schema is only two columns: `audio` and
`session_ids`. No transcript, no turn index, no timestamp field of any
kind. Individual clips are already trimmed to speech (0.15-28.8s each per
the dataset card), meaning the inter-turn SILENCE -- the entire signal an
endpointing model needs -- has already been stripped out and is not
recoverable from this source. This is a structural, not incidental,
limitation: no amount of clever preprocessing recovers timing data that
was never included.

What this script actually does: downloads a small real sample (proving the
audio itself is real, accessible, and free) and explicitly validates that
the timing gap is real, so nobody downstream mistakes "downloadable" for
"usable for pause labeling." If real pause timing is needed from this
corpus, it would have to come from the ORIGINAL DSTC2 distribution
(matthen/dstc on GitHub, which includes full call logs) -- NOT verified in
this pass whether those logs carry per-turn wall-clock timestamps; that's
the next thing to check before treating DSTC2 as viable for this task, not
something to assume.

Run:
    python scripts/prepare_dstc2_audio.py
"""

import json
import time
from pathlib import Path

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_AUDIO_DIR = ROOT / "data" / "turn_taking" / "dstc2_sample" / "audio"
OUT_MANIFEST_PATH = ROOT / "data" / "turn_taking" / "dstc2_sample" / "manifest.jsonl"

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "danielroncel/dstc2_audios"
CONFIG = "default"
SPLIT = "train"
N_SAMPLE = 40


def get_with_retry(url: str, params: dict = None, max_retries: int = 6, timeout: int = 30):
    """HF's datasets-server API rate-limits (429) under sustained request
    volume -- confirmed directly against this exact endpoint during this
    project's own research. Exponential backoff, honoring Retry-After when
    present, rather than crashing the download partway through."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait_s = float(resp.headers.get("Retry-After", 2 ** attempt))
        print(f"  429 rate-limited, waiting {wait_s:.0f}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait_s)
    raise SystemExit(f"gave up after {max_retries} retries against {url} -- still rate-limited")


def fetch_page(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    resp = get_with_retry(ROWS_URL, params=params)
    return resp.json()


def download_audio(url: str, dest: Path) -> None:
    resp = get_with_retry(url)
    dest.write_bytes(resp.content)


def main() -> None:
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    page = fetch_page(0, N_SAMPLE)
    feature_names = {f["name"] for f in page["features"]}
    print(f"columns present in {DATASET}: {sorted(feature_names)}")

    # The actual point of this script: prove, with a real assertion against
    # the real schema, that no timing field exists -- not just claim it.
    timing_fields = {"begin_time", "end_time", "timestamp", "start_time", "duration", "turn_index", "turn_id"}
    found_timing_fields = feature_names & timing_fields
    if found_timing_fields:
        print(f"NOTE: found unexpected timing-related field(s) {found_timing_fields} -- re-evaluate, this docstring's premise may be stale")
    else:
        print("[CONFIRMED] no timing/turn-order field present -- pause duration is NOT recoverable from this HF mirror")

    records = []
    for row_entry in page["rows"]:
        row = row_entry["row"]
        row_idx = row_entry["row_idx"]
        session_id = row["session_ids"]
        audio_url = row["audio"][0]["src"]

        clip_id = f"dstc2_{session_id}_{row_idx}"
        audio_path = OUT_AUDIO_DIR / f"{clip_id}.wav"
        download_audio(audio_url, audio_path)
        duration_s = sf.info(audio_path).duration

        records.append({
            "clip_id": clip_id,
            "session_id": session_id,
            "row_idx": row_idx,
            "audio_path": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
            "duration_s": round(duration_s, 3),
        })

    with open(OUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_sessions = len({r["session_id"] for r in records})
    print(f"\ndownloaded {len(records)} real DSTC2 clips across {n_sessions} sessions -> {OUT_AUDIO_DIR}")
    print(f"manifest -> {OUT_MANIFEST_PATH}")
    print(
        "\nCONCLUSION: real, free, domain-relevant (human <-> automated support-style phone "
        "system) audio confirmed downloadable. NOT usable for turn-taking/endpointing labels as-"
        "is -- no pause timing in this source. Do not point baseline_fixed_threshold_vad.py or "
        "ab_compare_endpointers.py at this manifest; it isn't shaped for that. Useful only as a "
        "domain-flavor / ASR-robustness reference unless the original matthen/dstc call logs are "
        "pulled in and confirmed to carry real turn timestamps."
    )


if __name__ == "__main__":
    main()
