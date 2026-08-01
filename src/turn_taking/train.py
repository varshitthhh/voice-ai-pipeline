"""Phase 3.3 training loop + Phase 3.4 ablation/calibration helpers for TurnTakingGRU."""

import time

import torch
import torch.nn as nn


def train_model(model, train_batches: list, val_batches: list, epochs: int = 10, lr: float = 1e-3, device: str = "cpu", verbose: bool = True) -> list:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss_sum, n_train = 0.0, 0.0
        for token_ids, prosody, mask, targets in train_batches:
            token_ids, prosody, mask, targets = (t.to(device) for t in (token_ids, prosody, mask, targets))
            opt.zero_grad()
            logits = model(token_ids, prosody)
            loss = (loss_fn(logits, targets) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * mask.sum().item()
            n_train += mask.sum().item()

        model.eval()
        val_loss_sum, n_val, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            for token_ids, prosody, mask, targets in val_batches:
                token_ids, prosody, mask, targets = (t.to(device) for t in (token_ids, prosody, mask, targets))
                logits = model(token_ids, prosody)
                loss = (loss_fn(logits, targets) * mask).sum() / mask.sum().clamp(min=1)
                val_loss_sum += loss.item() * mask.sum().item()
                n_val += mask.sum().item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += ((preds == targets).float() * mask).sum().item()

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_train, 1),
            "val_loss": val_loss_sum / max(n_val, 1),
            "val_acc": correct / max(n_val, 1),
        }
        history.append(row)
        if verbose:
            print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f}")

    return history


def measure_inference_latency_ms(model, device: str = "cpu", n_reps: int = 200) -> float:
    """Single-frame (T=1) forward pass latency -- matches the <5ms Phase 3.3
    target, which is about per-tick streaming inference cost, not batch
    throughput."""
    model.eval().to(device)
    token_ids = torch.zeros((1, 1, 20), dtype=torch.long, device=device)
    prosody = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(10):  # warmup
            model(token_ids, prosody)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(token_ids, prosody)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return (elapsed / n_reps) * 1000


def calibrate_threshold(model, val_examples: list, thresholds: list, device: str = "cpu") -> list:
    """For each probability threshold, simulate streaming decision-making:
    fire 'turn complete' at the first frame within the labeled pause window
    where P(turn_complete) >= threshold. Mirrors
    scripts/baseline_fixed_threshold_vad.py's response-latency /
    false-interruption-rate definitions exactly, so the learned model's
    curve is directly comparable to the Phase 3.1 fixed-threshold curve on
    the same axes."""
    model.eval().to(device)
    rows = []

    for threshold in thresholds:
        latencies_ms = []
        n_mid_turn = 0
        n_interrupted = 0

        for token_ids, prosody, mask, label in val_examples:
            with torch.no_grad():
                logits = model(token_ids.unsqueeze(0).to(device), prosody.unsqueeze(0).to(device))
                probs = torch.sigmoid(logits).squeeze(0).cpu()

            pause_frames = mask.nonzero(as_tuple=True)[0]
            if len(pause_frames) == 0:
                continue

            fired_frame = None
            for f in pause_frames.tolist():
                if probs[f] >= threshold:
                    fired_frame = f
                    break

            if label == 1:
                if fired_frame is not None:
                    latencies_ms.append((fired_frame - pause_frames[0].item()) * 100)  # 100ms/frame
            else:
                n_mid_turn += 1
                if fired_frame is not None:
                    n_interrupted += 1

        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2] if latencies_ms else float("nan")
        rows.append({
            "threshold": round(threshold, 3),
            "response_latency_p50_ms": p50,
            "false_interruption_rate": round(n_interrupted / n_mid_turn, 4) if n_mid_turn else float("nan"),
            "n_true_end": len(latencies_ms),
            "n_mid_turn": n_mid_turn,
        })

    return rows
