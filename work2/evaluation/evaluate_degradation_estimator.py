"""
Evaluate trained degradation estimator and produce metrics + plots.
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
from work2.data.label_utils import normalize_snr, denormalize_snr
from work2.datasets.degradation_feature_dataset import DegradationFeatureDataset
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
    ax.scatter(y_true, y_pred, alpha=0.5, s=8, color="steelblue")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
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

    acc_keys = [
        ("valid_noise_acc", "Noise Acc"),
        ("valid_bw_acc", "BW Acc"),
        ("valid_bit_acc", "Bit Acc"),
    ]
    for key, label in acc_keys:
        if key in rows[0] and rows[0][key]:
            vals = [float(r[key]) for r in rows]
            axes[1].plot(epochs, vals, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Valid Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "valid_snr_mae_db" in rows[0] and rows[0]["valid_snr_mae_db"]:
        vals = [float(r["valid_snr_mae_db"]) for r in rows]
        axes[2].plot(epochs, vals)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE (dB)")
        axes[2].set_title("SNR MAE")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/estimator_cache")
    parser.add_argument("--checkpoint", type=str, default="outputs/degradation_estimator/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/degradation_estimator/evaluation")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"

    test_ds = DegradationFeatureDataset(str(index_path), str(cache_dir), split="test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try alternative
        ckpt_path = Path("outputs/degradation_estimator/best_model.pt")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = DegradationEstimator(input_dim=ckpt["input_dim"], hidden_dim=ckpt.get("hidden_dim", 32))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_preds = {"noise_logit": [], "snr_pred": [], "bandwidth_logits": [], "bit_logits": []}
    all_targets = {"noise_target": [], "snr_target": [], "snr_valid": [],
                   "bandwidth_target": [], "bit_target": [], "severity": []}

    with torch.no_grad():
        for batch in test_loader:
            stats = batch["stats"].to(device)
            preds = model(stats)

            for k in all_preds:
                all_preds[k].append(preds[k].cpu())

            for k in ["noise_target", "snr_target", "snr_valid", "bandwidth_target", "bit_target"]:
                val = batch[k]
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                all_targets[k].append(val)
            all_targets["severity"].extend(batch["severity"])

    for k in all_preds:
        all_preds[k] = torch.cat(all_preds[k], dim=0)
    for k in ["noise_target", "snr_target", "snr_valid", "bandwidth_target", "bit_target"]:
        all_targets[k] = torch.cat([t if t.dim() > 0 else t.unsqueeze(0) for t in all_targets[k]], dim=0)

    # Noise metrics
    noise_prob = torch.sigmoid(all_preds["noise_logit"]).numpy().flatten()
    noise_pred_bin = (noise_prob > 0.5).astype(int)
    noise_true = all_targets["noise_target"].numpy().flatten().astype(int)

    noise_acc = accuracy_score(noise_true, noise_pred_bin)
    noise_prec = precision_score(noise_true, noise_pred_bin, zero_division=0)
    noise_rec = recall_score(noise_true, noise_pred_bin, zero_division=0)
    noise_f1 = f1_score(noise_true, noise_pred_bin, zero_division=0)

    # Bandwidth
    bw_pred = all_preds["bandwidth_logits"].argmax(dim=1).numpy()
    bw_true = all_targets["bandwidth_target"].numpy()
    bw_acc = accuracy_score(bw_true, bw_pred)
    bw_f1 = f1_score(bw_true, bw_pred, average="macro", zero_division=0)

    # Bit
    bit_pred = all_preds["bit_logits"].argmax(dim=1).numpy()
    bit_true = all_targets["bit_target"].numpy()
    bit_acc = accuracy_score(bit_true, bit_pred)
    bit_f1 = f1_score(bit_true, bit_pred, average="macro", zero_division=0)

    # SNR
    snr_valid_mask = all_targets["snr_valid"].numpy().flatten().astype(bool)
    snr_mae_db = None
    snr_rmse_db = None
    snr_pearson = None
    snr_pred_db = None
    snr_true_db = None

    if snr_valid_mask.any():
        snr_pred_norm = all_preds["snr_pred"].numpy().flatten()
        snr_true_norm = all_targets["snr_target"].numpy().flatten()
        snr_pred_db = denormalize_snr(torch.tensor(snr_pred_norm[snr_valid_mask])).numpy()
        snr_true_db = snr_true_norm[snr_valid_mask]
        snr_mae_db = mean_absolute_error(snr_true_db, snr_pred_db)
        snr_rmse_db = np.sqrt(mean_squared_error(snr_true_db, snr_pred_db))
        snr_pearson, _ = pearsonr(snr_true_db, snr_pred_db)

    # Confusion matrices
    bw_labels = ["Full (0)", "6k (1)", "4k (2)"]
    cm_bw = confusion_matrix(bw_true, bw_pred, labels=[0, 1, 2])
    plot_confusion(cm_bw, bw_labels, "Bandwidth Confusion Matrix",
                   out_dir / "bandwidth_confusion_matrix.png")

    bit_labels = ["16-bit (0)", "12-bit (1)", "10-bit (2)", "8-bit (3)"]
    cm_bit = confusion_matrix(bit_true, bit_pred, labels=[0, 1, 2, 3])
    plot_confusion(cm_bit, bit_labels, "Bit-depth Confusion Matrix",
                   out_dir / "bit_confusion_matrix.png")

    noise_labels = ["Clean (0)", "Noisy (1)"]
    cm_noise = confusion_matrix(noise_true, noise_pred_bin, labels=[0, 1])
    plot_confusion(cm_noise, noise_labels, "Noise Confusion Matrix",
                   out_dir / "noise_confusion_matrix.png")

    if snr_pred_db is not None and len(snr_pred_db) > 1:
        plot_scatter(snr_true_db, snr_pred_db,
                     "True SNR (dB)", "Predicted SNR (dB)",
                     "SNR Prediction Scatter", out_dir / "snr_scatter.png")

    # Training curves
    history_csv = Path(args.output_dir).parent / "history.csv"
    if not history_csv.exists():
        history_csv = Path("outputs/degradation_estimator/history.csv")
    if history_csv.exists():
        plot_training_curves(history_csv, out_dir / "training_curves.png")

    # Severity breakdown
    sevs = all_targets["severity"]
    sev_metrics = []
    for sev in ["clean", "light", "medium", "heavy"]:
        mask = np.array([s == sev for s in sevs])
        if not mask.any():
            continue
        sev_noise_acc = accuracy_score(noise_true[mask], noise_pred_bin[mask]) if mask.any() else 0
        sev_bw_acc = accuracy_score(bw_true[mask], bw_pred[mask]) if mask.any() else 0
        sev_bit_acc = accuracy_score(bit_true[mask], bit_pred[mask]) if mask.any() else 0
        sev_metrics.append({
            "severity": sev,
            "n_samples": int(mask.sum()),
            "noise_accuracy": round(sev_noise_acc, 4),
            "bandwidth_accuracy": round(sev_bw_acc, 4),
            "bit_accuracy": round(sev_bit_acc, 4),
        })

    sev_csv = out_dir / "metrics_by_severity.csv"
    with open(sev_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "n_samples",
                                                "noise_accuracy", "bandwidth_accuracy", "bit_accuracy"])
        writer.writeheader()
        for row in sev_metrics:
            writer.writerow(row)

    # Predictions CSV
    pred_csv = out_dir / "predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["severity", "noise_true", "noise_pred_prob",
                         "bw_true", "bw_pred", "bit_true", "bit_pred",
                         "snr_true_db", "snr_pred_db"])
        for i in range(len(sev_metrics)):
            idx = np.where(np.array(sevs) == sev_metrics[i]["severity"])[0][0]
            # Write all rows
        for i in range(len(all_targets["severity"])):
            snr_t = denormalize_snr(all_targets["snr_target"][i]).item() if snr_valid_mask[i] else ""
            snr_p = snr_pred_db[np.where(snr_valid_mask)[0]][np.searchsorted(np.where(snr_valid_mask)[0], i)] if snr_valid_mask[i] and snr_pred_db is not None else ""
            writer.writerow([
                all_targets["severity"][i],
                noise_true[i],
                round(float(noise_prob[i]), 4),
                bw_true[i],
                bw_pred[i],
                bit_true[i],
                bit_pred[i],
                snr_t,
                snr_p,
            ])

    metrics = {
        "noise": {
            "accuracy": round(noise_acc, 4),
            "precision": round(noise_prec, 4),
            "recall": round(noise_rec, 4),
            "f1": round(noise_f1, 4),
        },
        "bandwidth": {
            "accuracy": round(bw_acc, 4),
            "macro_f1": round(bw_f1, 4),
        },
        "bit": {
            "accuracy": round(bit_acc, 4),
            "macro_f1": round(bit_f1, 4),
        },
        "snr": {
            "mae_db": round(float(snr_mae_db), 4) if snr_mae_db is not None else None,
            "rmse_db": round(float(snr_rmse_db), 4) if snr_rmse_db is not None else None,
            "pearson_r": round(float(snr_pearson), 4) if snr_pearson is not None else None,
        },
        "by_severity": sev_metrics,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[INFO] Evaluation complete.")
    for sev in sev_metrics:
        print(f"  {sev['severity']:8s}: n={sev['n_samples']}, "
              f"noise_acc={sev['noise_accuracy']:.3f}, "
              f"bw_acc={sev['bandwidth_accuracy']:.3f}, "
              f"bit_acc={sev['bit_accuracy']:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
