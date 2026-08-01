# Intent Corpus Schema

Phase 1.2 setup for the two corpora Phase 5's intent classifier depends on:
Bitext Customer Support (training) and SLURP (audio eval). This doc fixes the
record format both are normalized into, and is explicit about where the two
corpora's roles diverge — they do **not** share a label space.

## 1. Corpora and their roles

| Corpus | Role | Native taxonomy | Modality |
|---|---|---|---|
| **Bitext Customer Support** (26,872 rows) | Training corpus for the Phase 5 MiniLM intent classifier | 27 intents / 11 categories, e-commerce-support-specific (`cancel_order`, `track_refund`, `payment_issue`, ...) | text only |
| **SLURP** (`qmeeus/slurp`, `test` split sampled) | Audio+intent eval set for the audio pipeline | 91 intents, general-purpose home-assistant domain (`alarm_set`, `weather_query`, `play_music`, ...) | audio + transcript |

### Why there is no intent taxonomy mapping

SLURP's 91 intents are a personal-assistant taxonomy (alarms, calendar, IoT,
music, weather) with essentially zero semantic overlap with Bitext's 27
e-commerce-support intents. Forcing a label mapping between them would be
fabricated, not derived. SLURP is kept in its own label space and used for
what it actually offers: real recorded speech paired with ground-truth intent
labels, i.e. a small out-of-domain check that the audio ingestion side of the
pipeline (real mic audio in, correct structured label out) works end-to-end —
not a second e-commerce eval set. The Bitext-trained classifier's own accuracy
is measured only against the Bitext held-out gold set (Section 3), per the
Phase 6 target (`Intent accuracy... >90%` on the held-out gold set).

## 2. Unified record schema

Both corpora are normalized to the same JSONL record shape so downstream
tooling (Phase 5 data loaders) has one format to read, even though `intent`
values are drawn from two disjoint label sets:

```json
{
  "id": "bitext_000123",
  "corpus": "bitext",
  "text": "I need to cancel my order",
  "intent": "cancel_order",
  "category": "ORDER",
  "audio_path": null,
  "split": "train",
  "source_row_idx": 123
}
```

| Field | Bitext | SLURP |
|---|---|---|
| `id` | `bitext_{row_idx:06d}` | `slurp_{slurp_id}_{row_idx}` |
| `corpus` | `"bitext"` | `"slurp"` |
| `text` | `instruction` column | `sentence` column |
| `intent` | `intent` column (27 classes) | `intent` column, decoded from SLURP's `ClassLabel` int (91 classes) |
| `category` | `category` column (11 classes) | `null` (SLURP has no category grouping) |
| `audio_path` | `null` (text-only corpus) | relative path to the downloaded `.flac` clip |
| `split` | `"train"` or `"gold_candidate"` (Section 3) | `"eval"` |
| `source_row_idx` | original CSV row index | original `test` split row index |

## 3. Held-out gold set (Bitext)

300 utterances are pulled out of Bitext before training, stratified across
its 27 intents (~11 per intent), into `split: "gold_candidate"`. These rows
are **excluded from the training split** entirely.

Bitext's own `intent` labels are dataset-generated, not hand-verified — treat
them as noisy until a human confirms each of the 300. That hand-verification
(confirm or correct each label) is a manual step outside this repo's tooling;
once done, the corrected 300 becomes the Phase 6 "held-out gold set" that
Bitext-trained classifier accuracy (target `>90%`) is measured against.

## 4. File layout

```
data/intent/
  raw/
    bitext_customer_support.csv          # fetched from Hugging Face
    slurp_eval/audio/*.flac                # fetched sample of SLURP test-split clips
  processed/
    bitext_train.jsonl                     # unified schema, split=train
    bitext_gold_candidates.jsonl           # unified schema, split=gold_candidate (needs hand-verification)
    slurp_eval.jsonl                       # unified schema, split=eval
```
