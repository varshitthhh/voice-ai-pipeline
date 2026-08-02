"""Phase 0.3 hardware baseline: the same script, run once per candidate
machine, feeding one CSV row each into the "what runs where" decision.

VRAM is read from the CUDA allocator. TTFT and throughput are measured on a
small synthetic linear-stack workload (representative inference shape, not
a real project model) so the script is fast and runs identically on CPU,
Colab T4, and the 24GB box without pulling in any model weights.

Run once per machine, then keep the CSV in sync across machines:
    python scripts/hardware_baseline.py --label 24gb-gpu
    python scripts/hardware_baseline.py --label colab-t4
    python scripts/hardware_baseline.py --label zenbook-cpu

Each run appends one row to outputs/hardware_baseline.csv.

NOTE: authored and smoke-tested on a CPU-only laptop (no CUDA device
available). Every `device.type == "cuda"` branch below is therefore
unexercised here and is marked TODO — it must be run for real on the
24GB box and on a Colab T4 runtime before Phase 0.3's gate (the CSV
deciding what runs where) is actually satisfied.
"""

import argparse
import csv
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "hardware_baseline.csv"

HIDDEN = 4096
LAYERS = 4
WARMUP_ITERS = 10
BENCH_ITERS = 50
BATCH = 16  # throughput measurement batch size

# Serving the full stack (ASR + LLM + TTS) needs headroom beyond a single
# 7B-AWQ model's footprint; below this we assume train/sweep-only duty.
# Was 20 (sized for a 24GB box that was never actually available this
# project); lowered to 15 now that a 16GB T4 is the only GPU this project
# has and is the one actually serving Phase 2's real numbers -- the
# threshold should describe the hardware the project runs on, not hardware
# that was never provisioned.
SERVING_VRAM_GB_MIN = 15


def build_workload(device: torch.device) -> nn.Module:
    layers = []
    for _ in range(LAYERS):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return nn.Sequential(*layers).to(device).eval()


def sync(device: torch.device) -> None:
    # TODO(gpu-validation): torch.cuda.synchronize is required here so the
    # perf_counter() timings below don't undercount async-dispatched CUDA
    # kernels. Never verified against a real device queue — confirm on the
    # 24GB box / Colab T4 that this actually blocks until GPU work drains.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(device: torch.device) -> dict:
    model = build_workload(device)

    x_warm = torch.randn(1, HIDDEN, device=device)
    for _ in range(WARMUP_ITERS):
        model(x_warm)
    sync(device)

    # TTFT proxy: single-sample forward dispatch after the model is warm.
    x_one = torch.randn(1, HIDDEN, device=device)
    sync(device)
    t0 = time.perf_counter()
    model(x_one)
    sync(device)
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Throughput: batched forward passes.
    x_batch = torch.randn(BATCH, HIDDEN, device=device)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        model(x_batch)
    sync(device)
    elapsed = time.perf_counter() - t0
    throughput_samples_per_sec = (BENCH_ITERS * BATCH) / elapsed

    vram_total_gb = None
    vram_used_gb = None
    # TODO(gpu-validation): VRAM read path is entirely untested locally —
    # no CUDA device on this laptop to exercise get_device_properties /
    # max_memory_allocated against. Confirm the units and that peak
    # allocation (not just current) is what we want on the 24GB box / T4.
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_total_gb = round(props.total_memory / 1e9, 2)
        vram_used_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)

    return {
        "ttft_ms": round(ttft_ms, 3),
        "throughput_samples_per_sec": round(throughput_samples_per_sec, 1),
        "vram_total_gb": vram_total_gb,
        "vram_used_gb": vram_used_gb,
    }


def recommend_role(device_type: str, vram_total_gb) -> str:
    if device_type != "cuda":
        return "orchestrator/dashboard only (no CUDA)"
    # TODO(gpu-validation): SERVING_VRAM_GB_MIN=15 is sized for a 16GB T4,
    # not a measurement. Re-check this threshold once real vram_used_gb
    # numbers exist for the actual serving stack (Phase 2, Sections 12-14 of
    # voice_ai_pipeline_v2.ipynb).
    if vram_total_gb is not None and vram_total_gb >= SERVING_VRAM_GB_MIN:
        return "serving: ASR + LLM + TTS (headline latency measured here)"
    return "training/sweeps only (insufficient VRAM to serve full stack)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="e.g. 24gb-gpu, colab-t4, zenbook-cpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO(gpu-validation): reset_peak_memory_stats / get_device_name have
    # never run against a real CUDA context from this machine. Confirm on
    # the 24GB box / Colab T4 that peak stats are actually zeroed before
    # measure() runs, otherwise vram_used_gb above will read stale.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()

    result = measure(device)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "device_type": device.type,
        "device_name": device_name,
        "torch_version": torch.__version__,
        **result,
        "recommended_role": recommend_role(device.type, result["vram_total_gb"]),
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(f"appended row for label={args.label!r} to {CSV_PATH}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
