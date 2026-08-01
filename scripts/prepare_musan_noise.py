"""Phase 1.3: fetch a small MUSAN noise sample.

MUSAN's canonical source (openslr.org/17) is a single 11GB tar.gz with no
per-category download, too large for a light sweep. `bilguun/musan-noise`
is a public, ungated Hugging Face mirror of just the noise subset (929 raw
wav files, CC BY 4.0), preserving MUSAN's original free-sound/sound-bible
layout — this fetches a small, evenly-spread sample of individual clips
directly rather than the whole subset.

Run:
    python scripts/prepare_musan_noise.py
"""

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "noise" / "raw" / "musan_noise"

REPO = "bilguun/musan-noise"
RESOLVE_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# Evenly spread indices across each subset's known file count (843 free-sound,
# 87 sound-bible, per the repo's file listing) rather than the first N, so the
# light sample isn't all drawn from one recording session.
FREE_SOUND_INDICES = [0, 140, 280, 420, 560, 700]
SOUND_BIBLE_INDICES = [0, 25, 50, 75]


def clip_paths() -> list:
    paths = [f"noise/free-sound/noise-free-sound-{i:04d}.wav" for i in FREE_SOUND_INDICES]
    paths += [f"noise/sound-bible/noise-sound-bible-{i:04d}.wav" for i in SOUND_BIBLE_INDICES]
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for rel_path in clip_paths():
        url = f"{RESOLVE_BASE}/{rel_path}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        dest = OUT_DIR / Path(rel_path).name
        dest.write_bytes(resp.content)

        manifest.append({
            "clip_id": dest.stem,
            "subset": "free-sound" if "free-sound" in rel_path else "sound-bible",
            "source_repo": REPO,
            "source_path": rel_path,
            "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": len(resp.content),
        })
        print(f"fetched {rel_path} ({len(resp.content) / 1e3:.0f} KB)")

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    total_mb = sum(r["size_bytes"] for r in manifest) / 1e6
    print(f"wrote {len(manifest)} clips ({total_mb:.1f} MB total) -> {OUT_DIR}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
