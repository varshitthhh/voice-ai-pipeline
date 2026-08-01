"""Phase 1.2: fetch a small audio+intent eval sample from SLURP.

SLURP's full dataset is 6.75GB with audio; this project only needs "a small
audio+intent eval set" (per README), so this pulls exactly N rows (default
300) from the `test` split via Hugging Face's datasets-server `/rows` API
(server-side row selection — no need to download the underlying 6.75GB of
parquet shards) and downloads each row's individual audio clip.

SLURP's 91-class intent taxonomy is a general-purpose home-assistant domain
with no overlap with Bitext's e-commerce intents — see
docs/intent_corpus_schema.md Section 1. This eval set is NOT merged into the
Bitext label space.

Run:
    python scripts/prepare_slurp_eval.py
"""

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = ROOT / "data" / "intent" / "raw" / "slurp_eval" / "audio"
PROCESSED_DIR = ROOT / "data" / "intent" / "processed"

DATASET = "qmeeus/slurp"
CONFIG = "default"
SPLIT = "test"
N_ROWS = 300
PAGE_SIZE = 100  # datasets-server hard-caps `length` at 100 per request
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_page(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    resp = requests.get(ROWS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_audio(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> None:
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    intent_names = None
    fetched = 0

    while fetched < N_ROWS:
        length = min(PAGE_SIZE, N_ROWS - fetched)
        page = fetch_page(fetched, length)

        if intent_names is None:
            intent_feature = next(f for f in page["features"] if f["name"] == "intent")
            intent_names = intent_feature["type"]["names"]

        for row_entry in page["rows"]:
            row = row_entry["row"]
            row_idx = row_entry["row_idx"]
            slurp_id = row["slurp_id"]
            intent = intent_names[row["intent"]]
            audio_url = row["audio"][0]["src"]

            audio_filename = f"slurp_{slurp_id}_{row_idx}.flac"
            audio_path = RAW_AUDIO_DIR / audio_filename
            download_audio(audio_url, audio_path)

            records.append({
                "id": f"slurp_{slurp_id}_{row_idx}",
                "corpus": "slurp",
                "text": row["sentence"],
                "intent": intent,
                "category": None,
                "audio_path": str(audio_path.relative_to(ROOT)).replace("\\", "/"),
                "split": "eval",
                "source_row_idx": row_idx,
            })

        fetched += len(page["rows"])
        print(f"fetched {fetched}/{N_ROWS} rows")
        if len(page["rows"]) < length:
            break  # ran out of rows in the split before hitting N_ROWS

    out_path = PROCESSED_DIR / "slurp_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total_bytes = sum((RAW_AUDIO_DIR / Path(r["audio_path"]).name).stat().st_size for r in records)
    print(f"wrote {len(records)} rows -> {out_path}")
    print(f"audio on disk: {total_bytes / 1e6:.1f} MB in {RAW_AUDIO_DIR}")
    unique_intents = sorted({r["intent"] for r in records})
    print(f"{len(unique_intents)} distinct SLURP intents present in sample")


if __name__ == "__main__":
    main()
