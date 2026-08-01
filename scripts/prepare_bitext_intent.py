"""Phase 1.2: fetch + normalize the Bitext Customer Support corpus.

Downloads the dataset CSV from Hugging Face (bitext/Bitext-customer-support-
llm-chatbot-training-dataset, CDLA-Sharing-1.0, 19.2MB, 26,872 rows), then
splits it into a training set and a 300-sample held-out gold-candidate set
stratified across all 27 intents, both written in the unified schema from
docs/intent_corpus_schema.md.

The gold-candidate split is NOT gold yet — Bitext's labels are dataset-
generated, not hand-verified. Per docs/intent_corpus_schema.md Section 3, a
human must confirm or correct each of the 300 before Phase 6 treats it as
ground truth.

Run:
    python scripts/prepare_bitext_intent.py
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "intent" / "raw"
PROCESSED_DIR = ROOT / "data" / "intent" / "processed"

REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

GOLD_CANDIDATE_TOTAL = 300
RANDOM_SEED = 0


def fetch_csv() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        local_dir=RAW_DIR,
    )
    return Path(path)


def stratified_gold_indices(df: pd.DataFrame, total: int, seed: int) -> pd.Index:
    """~`total` rows spread evenly across every intent, largest classes absorb the remainder."""
    intents = df["intent"].value_counts().index.tolist()
    base_quota = total // len(intents)
    remainder = total - base_quota * len(intents)

    picked = []
    for i, intent in enumerate(intents):
        quota = base_quota + (1 if i < remainder else 0)
        pool = df[df["intent"] == intent]
        picked.append(pool.sample(n=min(quota, len(pool)), random_state=seed).index)
    return pd.Index([idx for group in picked for idx in group])


def to_record(row: pd.Series, row_idx: int, split: str) -> dict:
    return {
        "id": f"bitext_{row_idx:06d}",
        "corpus": "bitext",
        "text": row["instruction"],
        "intent": row["intent"],
        "category": row["category"],
        "audio_path": None,
        "split": split,
        "source_row_idx": row_idx,
    }


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    csv_path = fetch_csv()
    print(f"fetched {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows, {df['intent'].nunique()} intents, {df['category'].nunique()} categories")

    gold_idx = stratified_gold_indices(df, GOLD_CANDIDATE_TOTAL, RANDOM_SEED)
    gold_df = df.loc[gold_idx]
    train_df = df.drop(index=gold_idx)

    train_records = [to_record(row, idx, "train") for idx, row in train_df.iterrows()]
    gold_records = [to_record(row, idx, "gold_candidate") for idx, row in gold_df.iterrows()]

    write_jsonl(train_records, PROCESSED_DIR / "bitext_train.jsonl")
    write_jsonl(gold_records, PROCESSED_DIR / "bitext_gold_candidates.jsonl")

    print(f"wrote {len(train_records)} train rows -> {PROCESSED_DIR / 'bitext_train.jsonl'}")
    print(f"wrote {len(gold_records)} gold-candidate rows -> {PROCESSED_DIR / 'bitext_gold_candidates.jsonl'}")
    print("gold-candidate intent distribution:")
    print(gold_df["intent"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
