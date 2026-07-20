"""
Fast evaluation for shard-based degradation estimator cache.
"""
import argparse
import json
import sys
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
)
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.data.label_utils import denormalize_snr
from work2.datasets.degradation_feature_shard_dataset import DegradationFeatureShardDataset
from work2.models.degradation_estimator import DegradationEstimator


def plot_confusion(cm, labels, title, savepath):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_scatter(y_true, y_pred, xlabel, ylabel, title, savepath):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=6)
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_training_curves(history_csv: Path, savepath: Path):
    if not history_csv.exists():
        return
    rows = []
    with open(history_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    valid_loss = [float(r["valid_loss"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, valid_loss, label="Valid")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metric_pairs = [
        ("valid_noise_acc", "Noise Acc"),
        ("valid_bw_acc", "BW Acc"),
        ("valid_bit_acc", "Bit Acc"),
    ]
    for key, label in metric_pairs:
        if key in rows[0] and rows[0][key] not in ("", None):
            vals = [float(r[key]) for r in rows]
            axes[1].plot(epochs, vals, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Valid Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "valid_snr_mae_db" in rows[0] and rows[0]["valid_snr_mae_db"] not in ("", None):
        vals = [float(r["valid_snr_mae_db"]) for r in rows]
        axes[2].plot(epochs, vals)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE (dB)")
        axes[2].set_title("SNR MAE")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def collate_fn(batch):
    return {
        "stats": torch.stack([b["stats"] for b in batch], dim=0),
        "noise_target": torch.stack([b["noise_target"] for b in batch], dim=0),
        "snr_target": torch.stack([b["snr_target"] for b in batch], dim=0),
        "snr_valid": torch.stack([b["snr_valid"] for b in batch], dim=0),
        "bandwidth_target": torch.tensor([int(b["bandwidth_target"]) for b in batch], dtype=torch.long),
        "bit_target": torch.tensor([int(b["bit_target"]) for b in batch], dtype=torch.long),
        "severity": [b["severity"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/estimator_cache_sharded")
    parser.add_argument("--checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/degradation_estimator_fast/evaluation")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"
    test_ds = DegradationFeatureShardDataset(str(index_path), str(cache_dir), split="test")
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DegradationEstimator(input_dim=ckpt["input_dim"], hidden_dim=ckpt.get("hidden_dim", 32))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_noise_prob, all_noise_true = [], []
    all_bw_pred, all_bw_true = [], []
    all_bit_pred, all_bit_true = [], []
    all_snr_pred, all_snr_true, all_snr_valid = [], [], []
    all_sev = []

    with torch.no_grad():
        for batch in test_loader:
            stats = batch["stats"].to(device)
            preds = model(stats)
            all_noise_prob.append(torch.sigmoid(preds["noise_logit"]).cpu().view(-1))
            all_noise_true.append(batch["noise_target"].view(-1))
            all_bw_pred.append(preds["bandwidth_logits"].argmax(dim=1).cpu())
            all_bw_true.append(batch["bandwidth_target"])
            all_bit_pred.append(preds["bit_logits"].argmax(dim=1).cpu())
            all_bit_true.append(batch["bit_target"])
            all_snr_pred.append(preds["snr_pred"].cpu().view(-1))
            all_snr_true.append(batch["snr_target"].view(-1))
            all_snr_valid.append(batch["snr_valid"].view(-1))
            all_sev.extend(batch["severity"])

    noise_prob = torch.cat(all_noise_prob).numpy().flatten()
    noise_true = torch.cat(all_noise_true).numpy().flatten().astype(int)
    noise_pred = (noise_prob > 0.5).astype(int)

    bw_pred = torch.cat(all_bw_pred).numpy()
    bw_true = torch.cat(all_bw_true).numpy()
    bit_pred = torch.cat(all_bit_pred).numpy()
    bit_true = torch.cat(all_bit_true).numpy()
    snr_valid = torch.cat(all_snr_valid).numpy().flatten().astype(bool)

    noise_acc = accuracy_score(noise_true, noise_pred)
    noise_prec = precision_score(noise_true, noise_pred, zero_division=0)
    noise_rec = recall_score(noise_true, noise_pred, zero_division=0)
    noise_f1 = f1_score(noise_true, noise_pred, zero_division=0)

    bw_acc = accuracy_score(bw_true, bw_pred)
    bw_f1 = f1_score(bw_true, bw_pred, average="macro", zero_division=0)

    bit_acc = accuracy_score(bit_true, bit_pred)
    bit_f1 = f1_score(bit_true, bit_pred, average="macro", zero_division=0)

    snr_mae_db = snr_rmse_db = snr_pearson = None
    snr_pred_db = snr_true_db = None
    if snr_valid.any():
        snr_p = torch.cat(all_snr_pred).numpy().flatten()
        snr_t = torch.cat(all_snr_true).numpy().flatten()
        snr_pred_db = denormalize_snr(torch.tensor(snr_p[snr_valid])).numpy()
        snr_true_db = snr_t[snr_valid]
        snr_mae_db = mean_absolute_error(snr_true_db, snr_pred_db)
        snr_rmse_db = np.sqrt(mean_squared_error(snr_true_db, snr_pred_db))
        if len(snr_true_db) > 1:
            snr_pearson, _ = pearsonr(snr_true_db, snr_pred_db)

    plot_confusion(confusion_matrix(noise_true, noise_pred, labels=[0,1]), ["Clean", "Noisy"], "Noise Confusion Matrix", out_dir / "noise_confusion_matrix.png")
    plot_confusion(confusion_matrix(bw_true, bw_pred, labels=[0,1,2]), ["Full", "6k", "4k"], "Bandwidth Confusion Matrix", out_dir / "bandwidth_confusion_matrix.png")
    plot_confusion(confusion_matrix(bit_true, bit_pred, labels=[0,1,2,3]), ["16b", "12b", "10b", "8b"], "Bit-depth Confusion Matrix", out_dir / "bit_confusion_matrix.png")
    if snr_pred_db is not None and len(snr_true_db) > 1:
        plot_scatter(snr_true_db, snr_pred_db, "True SNR (dB)", "Predicted SNR (dB)", "SNR Prediction Scatter", out_dir / "snr_scatter.png")

    history_csv = Path(args.output_dir).parent / "history.csv"
    if history_csv.exists():
        plot_training_curves(history_csv, out_dir / "training_curves.png")

    sev_metrics = []
    for sev in ["clean", "light", "medium", "heavy"]:
        mask = np.array([s == sev for s in all_sev])
        if not mask.any():
            continue
        sev_metrics.append({
            "severity": sev,
            "n_samples": int(mask.sum()),
            "noise_accuracy": round(float(accuracy_score(noise_true[mask], noise_pred[mask])), 4),
            "bandwidth_accuracy": round(float(accuracy_score(bw_true[mask], bw_pred[mask])), 4),
            "bit_accuracy": round(float(accuracy_score(bit_true[mask], bit_pred[mask])), 4),
        })

    with open(out_dir / "metrics_by_severity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "n_samples", "noise_accuracy", "bandwidth_accuracy", "bit_accuracy"])
        writer.writeheader()
        writer.writerows(sev_metrics)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["severity", "noise_true", "noise_pred_prob", "bw_true", "bw_pred", "bit_true", "bit_pred", "snr_true_db", "snr_pred_db"])
        snr_p_all = torch.cat(all_snr_pred).numpy().flatten()
        snr_t_all = torch.cat(all_snr_true).numpy().flatten()
        for i in range(len(all_sev)):
            snr_t = snr_t_all[i] if snr_valid[i] else ""
            snr_p = float(denormalize_snr(torch.tensor([snr_p_all[i]])).item()) if snr_valid[i] else ""
            writer.writerow([all_sev[i], noise_true[i], round(float(noise_prob[i]), 4), bw_true[i], bw_pred[i], bit_true[i], bit_pred[i], snr_t, snr_p])

    metrics = {
        "noise": {"accuracy": round(float(noise_acc), 4), "precision": round(float(noise_prec), 4), "recall": round(float(noise_rec), 4), "f1": round(float(noise_f1), 4)},
        "bandwidth": {"accuracy": round(float(bw_acc), 4), "macro_f1": round(float(bw_f1), 4)},
        "bit": {"accuracy": round(float(bit_acc), 4), "macro_f1": round(float(bit_f1), 4)},
        "snr": {"mae_db": round(float(snr_mae_db), 4) if snr_mae_db is not None else None, "rmse_db": round(float(snr_rmse_db), 4) if snr_rmse_db is not None else None, "pearson_r": round(float(snr_pearson), 4) if snr_pearson is not None else None},
        "by_severity": sev_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[INFO] Evaluation complete.")
    print(f"[INFO] noise_acc={noise_acc:.4f}, bw_acc={bw_acc:.4f}, bit_acc={bit_acc:.4f}" + (f", snr_mae={snr_mae_db:.2f}dB" if snr_mae_db is not None else ""))
    print(f"[INFO] Outputs saved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
