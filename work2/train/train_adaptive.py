"""
Train the adaptive residual module using cached GTCRN features.

GTCRN is frozen; only the residual module is trained.
The degradation estimator can be optionally frozen or co-trained.
"""
import argparse
import json
import sys
import time
import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.data.stft_utils import stft_to_ri, ri_to_istft
from work2.data.degradation import apply_composite_degradation, sample_degradation_config
from work2.data.label_utils import make_degradation_labels
from work2.losses.residual_loss import AdaptiveResidualLoss
from work2.models.gtcrn_adaptive import GTCRNAdaptive, build_degradation_condition_vector
from work2.models.degradation_estimator import DegradationEstimator


def generate_training_batch(
    adaptive_model: GTCRNAdaptive,
    clean_audio: torch.Tensor,
    noise_audio: torch.Tensor,
    severity: str,
    seed: int,
) -> dict:
    """
    Generate one training sample:
    clean → degradation → STFT → GTCRN adaptive forward → labels.

    Returns a dict with enhanced_final, clean_spec, etc. for loss computation.
    """
    seed_everything(seed)

    cfg = sample_degradation_config(severity, seed)
    noise = noise_audio.clone() if cfg.add_noise else None

    result = apply_composite_degradation(clean_audio.clone(), cfg, noise)

    window = torch.hann_window(512).pow(0.5)
    clean_spec = stft_to_ri(result.clean, window=window)
    degraded_spec = stft_to_ri(result.degraded, window=window)

    out = adaptive_model(degraded_spec.unsqueeze(0))

    labels = make_degradation_labels(result)

    return {
        "enhanced_final": out["enhanced_final"].squeeze(0),
        "enhanced_base": out["enhanced_base"].squeeze(0),
        "residual": out["residual"].squeeze(0),
        "clean_spec": clean_spec,
        "bandwidth_limited": (result.cutoff_hz is not None or result.intermediate_rate is not None),
        "noise_present": labels["noise_present"],
        "snr_target": labels["snr_target"],
        "bandwidth_class": labels["bandwidth_class"],
        "bit_class": labels["bit_class"],
        "severity": severity,
        "degradation_conds": out["degradation_conds"].squeeze(0),
        "degradation_preds": out["degradation_preds"],
        "stats": out["stats"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train adaptive residual module")
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default=None,
                        help="Pretrained estimator checkpoint (optional)")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_residual")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--freeze-estimator", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building adaptive GTCRN pipeline...")
    model = GTCRNAdaptive(
        checkpoint_path=args.gtcrn_checkpoint,
        device="cpu",
        freeze_estimator=args.freeze_estimator,
    )

    if args.freeze_estimator and args.estimator_checkpoint:
        ckpt = torch.load(args.estimator_checkpoint, map_location="cpu")
        model.estimator.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded pretrained estimator from {args.estimator_checkpoint}")

    model.train_estimator_and_residual()
    param_info = model.count_params()
    print(f"[INFO] Params: extractor={param_info['extractor_total']} (trainable={param_info['extractor_trainable']}), "
          f"estimator={param_info['estimator_total']} (trainable={param_info['estimator_trainable']}), "
          f"residual={param_info['residual_total']} (trainable={param_info['residual_trainable']})")

    print(f"[INFO] Generating {args.num_samples} synthetic training samples...")
    import random
    rng = random.Random(args.seed)

    datasets = {"train": [], "valid": []}
    severities = ["clean", "light", "medium", "heavy"]
    n_per_sev = args.num_samples // (4 * 5)
    sample_idx = 0

    for i in range(args.num_samples // 4):
        dur = 1.5
        n = int(dur * 16000)
        t = torch.arange(n, dtype=torch.float32) / 16000
        f0 = rng.uniform(80, 400)
        clean = torch.zeros(n)
        for h in range(1, rng.randint(3, 7) + 1):
            clean += (rng.uniform(0.3, 1.0) / h) * torch.sin(2 * torch.pi * f0 * h * t)
        clean = clean / clean.abs().max() * 0.9
        noise_raw = torch.randn(n, dtype=torch.float32) * 0.1
        noise_raw = noise_raw - noise_raw.mean()

        for sev in severities:
            local_seed = args.seed * 100 + i * 10 + severities.index(sev)
            sample = generate_training_batch(model, clean, noise_raw, sev, local_seed)
            bucket = "valid" if sample_idx % 5 == 0 else "train"
            datasets[bucket].append(sample)
            sample_idx += 1

    print(f"[INFO] Dataset: train={len(datasets['train'])}, valid={len(datasets['valid'])}")

    loss_fn = AdaptiveResidualLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.estimator.parameters(), "lr": args.learning_rate * 0.5},
        {"params": model.residual_module.parameters(), "lr": args.learning_rate},
    ])

    history = []
    best_valid = float("inf")
    patience = 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.estimator.train()
        model.residual_module.train()

        train_loss = 0.0
        train_mag = 0.0
        train_hf = 0.0
        train_res = 0.0

        indices = list(range(len(datasets["train"])))
        import random as _random
        _random.Random(epoch + args.seed).shuffle(indices)
        batch_indices = [indices[i:i + args.batch_size] for i in range(0, len(indices), args.batch_size)]

        for batch_idx in batch_indices:
            batch = [datasets["train"][j] for j in batch_idx]
            B = len(batch)

        enhanced_final = torch.stack([b["enhanced_final"].permute(2, 0, 1) for b in batch]).to(device)
        clean_spec = torch.stack([b["clean_spec"].permute(2, 0, 1) for b in batch]).to(device)
        residual = torch.stack([b["residual"].permute(2, 0, 1) for b in batch]).to(device)
        bw_limited = torch.tensor([b["bandwidth_limited"] for b in batch], device=device)

            loss, comps = loss_fn(enhanced_final, clean_spec, residual, bw_limited)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.estimator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.residual_module.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mag += comps["mag"]
            train_hf += comps["hf"]
            train_res += comps["res"]

        n_batches = len(batch_indices)
        train_loss /= n_batches
        train_mag /= n_batches
        train_hf /= n_batches
        train_res /= n_batches

        model.eval()
        model.estimator.eval()
        model.residual_module.eval()

        valid_loss = 0.0
        with torch.no_grad():
            valid_indices = list(range(len(datasets["valid"])))
            valid_batches = [valid_indices[i:i + args.batch_size] for i in range(0, len(valid_indices), args.batch_size)]
            for batch_idx in valid_batches:
                batch = [datasets["valid"][j] for j in batch_idx]
                enhanced_final = torch.stack([b["enhanced_final"].permute(2, 0, 1) for b in batch]).to(device)
                clean_spec = torch.stack([b["clean_spec"].permute(2, 0, 1) for b in batch]).to(device)
                residual = torch.stack([b["residual"].permute(2, 0, 1) for b in batch]).to(device)
                bw_limited = torch.tensor([b["bandwidth_limited"] for b in batch], device=device)

                loss, _ = loss_fn(enhanced_final, clean_spec, residual, bw_limited)
                valid_loss += loss.item()
        valid_loss /= max(len(valid_batches), 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(valid_loss, 6),
            "train_mag": round(train_mag, 6),
            "train_hf": round(train_hf, 6),
            "train_res": round(train_res, 6),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  valid={valid_loss:.4f}  "
                  f"mag={train_mag:.4f}  hf={train_hf:.4f}  res={train_res:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            patience = 0
            torch.save({
                "estimator_state_dict": model.estimator.state_dict(),
                "residual_state_dict": model.residual_module.state_dict(),
                "stats_dim": model.stats_dim,
                "epoch": epoch,
                "seed": args.seed,
            }, out_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    t_total = time.perf_counter() - t_start

    torch.save({
        "estimator_state_dict": model.estimator.state_dict(),
        "residual_state_dict": model.residual_module.state_dict(),
        "stats_dim": model.stats_dim,
        "epoch": epoch,
        "seed": args.seed,
    }, out_dir / "last_model.pt")

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "train_mag", "train_hf", "train_res"])
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "seed": args.seed,
        "epochs": epoch,
        "total_time_s": round(t_total, 2),
        "best_valid_loss": round(best_valid, 6),
        "params": param_info,
        "train_samples": len(datasets["train"]),
        "valid_samples": len(datasets["valid"]),
        "freeze_estimator": args.freeze_estimator,
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Training complete in {t_total:.1f}s")
    print(f"[INFO] Best valid loss: {best_valid:.6f}")
    print(f"[INFO] Outputs saved to {out_dir}")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
