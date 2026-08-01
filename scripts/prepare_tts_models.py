"""Phase 2.3: fetch the Kokoro and Piper model files needed for the local
(CPU-feasible) half of the TTS benchmark. XTTS v2 is deliberately not
fetched here -- see scripts/benchmark_tts.py's docstring for why.

Sources:
    Kokoro:  github.com/thewh1teagle/kokoro-onnx release "model-files-v1.0"
             kokoro-v1.0.int8.onnx (92.4MB, int8-quantized, CPU-friendly)
             voices-v1.0.bin (28.2MB, all voices packed)
             License: Apache 2.0
    Piper:   huggingface.co/rhasspy/piper-voices, en_US-lessac-medium
             en_US-lessac-medium.onnx (63.2MB) + its .onnx.json config
             License: MIT (per rhasspy/piper-voices)

Run:
    python scripts/prepare_tts_models.py
"""

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "tts" / "models"

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
PIPER_FILES = ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"already have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"fetched {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    kokoro_dir = MODELS_DIR / "kokoro"
    piper_dir = MODELS_DIR / "piper"

    for filename in KOKORO_FILES:
        download(f"{KOKORO_BASE}/{filename}", kokoro_dir / filename)

    for filename in PIPER_FILES:
        download(f"{PIPER_BASE}/{filename}", piper_dir / filename)

    total_mb = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\ntotal on disk: {total_mb:.1f} MB in {MODELS_DIR}")


if __name__ == "__main__":
    main()
