"""
Train degradation estimator from cached GTCRN bottleneck features.

Command-line: python -m work2.train.train_degradation_estimator ...
"""
import argparse
import json
import sys
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.data.label_utils import normalize_snr, denormalize_snr
from work2.datasets.degradation_feature_dataset import DegradationFeatureDataset
from work2.models.degradation_estimator import DegradationEstimator
from work2.losses.degradation_estimator_loss import DegradationEstimatorLoss


def compute_metrics(predictions: dict, targets: dict):
    with torch.no_grad():
        noise_prob = torch.sigmoid(predictions["noise_logit"])
        noise_pred = (noise_prob > 0.5).float()
        noise_acc = (noise_pred == targets["noise_target"]).float().mean().item()

        bw_pred = predictions["bandwidth_logits"].argmax(dim=1)
        bw_acc = (bw_pred == targets["bandwidth_target"]).float().mean().item()

        bit_pred = predictions["bit_logits"].argmax(dim=1)
        bit_acc = (bit_pred == targets["bit_target"]).float().mean().item()

        snr_mae = None
        valid = targets["snr_valid"].bool()
        if valid.any():
            snr_pred_db = denormalize_snr(predictions["snr_pred"][valid])
            snr_true_db = denormalize_snr(targets["snr_target"][valid])
            snr_mae = (snr_pred_db - snr_true_db).abs().mean().item()

    return {
        "noise_acc": noise_acc,
        "bandwidth_acc": bw_acc,
        "bit_acc": bit_acc,
        "snr_mae_db": snr_mae,
    }


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_components = {"noise": 0.0, "snr": 0.0, "bandwidth": 0.0, "bit": 0.0}
    metrics_list = []

    for batch in loader:
        stats = batch["stats"].to(device)
        targets = {
            "noise_target": batch["noise_target"].view(-1, 1).to(device),
            "snr_target": normalize_snr(batch["snr_target"].view(-1, 1).to(device)),
            "snr_valid": batch["snr_valid"].view(-1, 1).to(device),
            "bandwidth_target": batch["bandwidth_target"].to(device),
            "bit_target": batch["bit_target"].to(device),
        }

        preds = model(stats)
        loss, loss_dict = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0.0)

        metrics_list.append(compute_metrics(preds, targets))

    n = len(loader)
    avg_metrics = {k: 0.0 for k in metrics_list[0]}
    for m in metrics_list:
        for k, v in m.items():
            if v is not None:
                avg_metrics[k] += v
    for k in avg_metrics:
        avg_metrics[k] /= n

    return {
        "loss": total_loss / n,
        "loss_noise": loss_components["noise"] / n,
        "loss_snr": loss_components["snr"] / n,
        "loss_bw": loss_components["bandwidth"] / n,
        "loss_bit": loss_components["bit"] / n,
        **avg_metrics,
    }


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    loss_components = {"noise": 0.0, "snr": 0.0, "bandwidth": 0.0, "bit": 0.0}
    metrics_list = []

    for batch in loader:
        stats = batch["stats"].to(device)
        targets = {
            "noise_target": batch["noise_target"].view(-1, 1).to(device),
            "snr_target": normalize_snr(batch["snr_target"].view(-1, 1).to(device)),
            "snr_valid": batch["snr_valid"].view(-1, 1).to(device),
            "bandwidth_target": batch["bandwidth_target"].to(device),
            "bit_target": batch["bit_target"].to(device),
        }

        preds = model(stats)
        loss, loss_dict = loss_fn(preds, targets)

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0.0)

        metrics_list.append(compute_metrics(preds, targets))

    n = len(loader)
    avg_metrics = {k: 0.0 for k in metrics_list[0]}
    nnz = {k: 0 for k in metrics_list[0]}
    for m in metrics_list:
        for k, v in m.items():
            if v is not None:
                avg_metrics[k] += v
                nnz[k] += 1
    for k in avg_metrics:
        avg_metrics[k] /= max(nnz[k], 1)

    return {
        "loss": total_loss / n,
        "loss_noise": loss_components["noise"] / n,
        "loss_snr": loss_components["snr"] / n,
        "loss_bw": loss_components["bandwidth"] / n,
        "loss_bit": loss_components["bit"] / n,
        **avg_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/estimator_cache")
    parser.add_argument("--output-dir", type=str, default="outputs/degradation_estimator")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Resume from last_model.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"

    train_ds = DegradationFeatureDataset(str(index_path), str(cache_dir), split="train")
    valid_ds = DegradationFeatureDataset(str(index_path), str(cache_dir), split="valid")
    test_ds = DegradationFeatureDataset(str(index_path), str(cache_dir), split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stats_dim = train_ds.stats_dim

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_valid_loss = float("inf")
    history = []

    if args.resume:
        ckpt_path = out_dir / "last_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model = DegradationEstimator(input_dim=ckpt["input_dim"], hidden_dim=ckpt.get("hidden_dim", 32)).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if (out_dir / "history.csv").exists():
                import csv
                with open(out_dir / "history.csv", "r") as f:
                    for row in csv.DictReader(f):
                        history.append({k: float(v) if v else 0.0 for k, v in row.items()})
                        history[-1]["epoch"] = int(history[-1]["epoch"])
                if "valid_loss" in history[-1]:
                    best_valid_loss = min(h["valid_loss"] for h in history)
            print(f"[INFO] Resumed from epoch {start_epoch}, best_valid_loss={best_valid_loss:.4f}")
        else:
            print("[WARN] --resume specified but no last_model.pt found, starting fresh")
            model = DegradationEstimator(input_dim=stats_dim).to(device)
    else:
        model = DegradationEstimator(input_dim=stats_dim).to(device)

    n_params = model.count_trainable_params()

    print(f"[INFO] Stats dimension: {stats_dim}")
    print(f"[INFO] Trainable params: {n_params}")
    print(f"[INFO] Dataset: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    loss_fn = DegradationEstimatorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6
    )

    config = {
        "input_dim": stats_dim,
        "hidden_dim": model.hidden_dim,
        "trainable_params": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
    }
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    patience_counter = 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_metrics = eval_epoch(model, valid_loader, loss_fn, device)

        scheduler.step(valid_metrics["loss"])

        epoch_info = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(epoch_info)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"valid_loss={valid_metrics['loss']:.4f} | "
              f"noise_acc={valid_metrics['noise_acc']:.3f} | "
              f"bw_acc={valid_metrics['bandwidth_acc']:.3f} | "
              f"bit_acc={valid_metrics['bit_acc']:.3f}"
              + (f" | snr_mae={valid_metrics['snr_mae_db']:.2f}dB" if valid_metrics.get('snr_mae_db') is not None else ""))

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": stats_dim,
                "hidden_dim": model.hidden_dim,
                "epoch": epoch,
                "validation_metrics": valid_metrics,
                "seed": args.seed,
            }, out_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    t_total = time.perf_counter() - t_start

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": stats_dim,
        "hidden_dim": model.hidden_dim,
        "epoch": epoch,
        "seed": args.seed,
    }, out_dir / "last_model.pt")

    test_metrics = eval_epoch(model, test_loader, loss_fn, device)
    print(f"[INFO] Test: noise_acc={test_metrics['noise_acc']:.4f}, "
          f"bw_acc={test_metrics['bandwidth_acc']:.4f}, "
          f"bit_acc={test_metrics['bit_acc']:.4f}"
          + (f", snr_mae={test_metrics['snr_mae_db']:.2f}dB" if test_metrics.get('snr_mae_db') is not None else ""))

    # History CSV
    hist_path = out_dir / "history.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "valid_loss",
            "train_noise_acc", "valid_noise_acc",
            "train_bw_acc", "valid_bw_acc",
            "train_bit_acc", "valid_bit_acc",
            "train_snr_mae_db", "valid_snr_mae_db",
        ])
        writer.writeheader()
        for h in history:
            writer.writerow({
                "epoch": h["epoch"],
                "train_loss": h["train"]["loss"],
                "valid_loss": h["valid"]["loss"],
                "train_noise_acc": h["train"].get("noise_acc", ""),
                "valid_noise_acc": h["valid"].get("noise_acc", ""),
                "train_bw_acc": h["train"].get("bandwidth_acc", ""),
                "valid_bw_acc": h["valid"].get("bandwidth_acc", ""),
                "train_bit_acc": h["train"].get("bit_acc", ""),
                "valid_bit_acc": h["valid"].get("bit_acc", ""),
                "train_snr_mae_db": h["train"].get("snr_mae_db", ""),
                "valid_snr_mae_db": h["valid"].get("snr_mae_db", ""),
            })

    summary = {
        **config,
        "best_valid_loss": best_valid_loss,
        "final_epoch": epoch,
        "total_training_time_s": round(t_total, 2),
        "test_metrics": test_metrics,
        "early_stopped": patience_counter >= args.patience,
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Total training time: {t_total:.1f}s")
    print(f"[INFO] Outputs saved to {out_dir}")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
