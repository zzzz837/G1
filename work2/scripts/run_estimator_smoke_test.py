"""
End-to-end smoke test: synthetic pseudo-speech -> GTCRN features -> train estimator.

This uses SYNTHETIC data only. Results DO NOT represent real speech performance.
The purpose is to verify the training/evaluation/plotting pipeline works end-to-end.
"""
import argparse
import json
import sys
import time
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error,
)
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import ensure_mono, peak_normalize
from work2.data.stft_utils import stft_to_ri
from work2.data.degradation import (
    apply_composite_degradation,
    sample_degradation_config,
)
from work2.data.label_utils import normalize_snr, denormalize_snr, make_degradation_labels
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.losses.degradation_estimator_loss import DegradationEstimatorLoss


def generate_pseudo_speech(n_samples: int, duration: float, fs: int, seed: int):
    """Generate synthetic pseudo-speech with harmonic structure."""
    import random
    rng = random.Random(seed)
    sigs = []
    for i in range(n_samples):
        n = int(duration * fs)
        t = torch.arange(n, dtype=torch.float32) / fs
        f0 = rng.uniform(80, 400)
        n_h = rng.randint(3, 8)
        sig = torch.zeros(n)
        for h in range(1, n_h + 1):
            amp = rng.uniform(0.3, 1.0) / h
            phase = rng.uniform(0, 2 * np.pi)
            freq_mod = 1.0 + 0.01 * torch.sin(2 * np.pi * rng.uniform(2, 8) * t)
            sig += amp * torch.sin(2 * np.pi * f0 * h * t * freq_mod + phase)

        n_seg = rng.randint(2, 5)
        seg_len = n // n_seg
        env = torch.ones(n)
        for s in range(n_seg):
            start = s * seg_len
            end = min((s + 1) * seg_len, n)
            if rng.random() < 0.3:
                env[start:end] = 0.0
            else:
                fade = min(400, (end - start) // 2)
                if fade > 0:
                    env[start:start + fade] = torch.linspace(0, 1, fade)
                    env[end - fade:end] = torch.linspace(1, 0, fade)
        sig = sig * env
        if sig.abs().max() < 1e-6:
            sig = torch.sin(2 * np.pi * f0 * t)
        sig = peak_normalize(sig, peak=0.9)
        sigs.append(sig)
    return sigs


def cache_from_synthetic(sigs, noises, extractor, out_dir: Path, seed: int):
    severities = ["clean", "light", "medium", "heavy"]
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    n_sources = len(sigs)

    for src_idx in range(n_sources):
        import hashlib
        h = hashlib.sha256(f"split_{seed}_{src_idx}".encode()).hexdigest()
        bucket = int(h, 16) % 10
        split = "train" if bucket < 8 else ("valid" if bucket < 9 else "test")

        for sev_idx, sev in enumerate(severities):
            local_seed = seed * 10000 + src_idx * 100 + sev_idx
            seed_everything(local_seed)

            cfg = sample_degradation_config(sev, local_seed)
            noise = noises[src_idx % len(noises)] if cfg.add_noise else None

            result = apply_composite_degradation(
                sigs[src_idx].clone(), cfg,
                noise.clone() if noise is not None else None,
            )

            window = torch.hann_window(512).pow(0.5)
            spec = stft_to_ri(result.degraded, window=window)

            feats = extractor(spec.unsqueeze(0))
            labels = make_degradation_labels(result)

            sample = {
                "stats": feats["stats"].squeeze(0).cpu(),
                "noise_target": torch.tensor([labels["noise_present"]]),
                "snr_target": torch.tensor([labels["snr_target"]]),
                "snr_valid": torch.tensor([labels["snr_valid"]]),
                "bandwidth_target": torch.tensor([labels["bandwidth_class"]]),
                "bit_target": torch.tensor([labels["bit_class"]]),
                "severity": sev,
            }

            fname = f"smoke_{src_idx * 4 + sev_idx:05d}.pt"
            torch.save(sample, out_dir / fname)

            entries.append({
                "cache_file": fname,
                "split": split,
                "source_path": f"synth_{src_idx:04d}",
                "severity": sev,
            })

    with open(out_dir / "index.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return entries


def train_model(cache_dir: Path, output_dir: Path, epochs: int, batch_size: int, lr: float, seed: int):
    from work2.datasets.degradation_feature_dataset import DegradationFeatureDataset

    seed_everything(seed)
    device = torch.device("cpu")

    idx = cache_dir / "index.jsonl"
    train_ds = DegradationFeatureDataset(str(idx), str(cache_dir), split="train")
    valid_ds = DegradationFeatureDataset(str(idx), str(cache_dir), split="valid")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dim = train_ds.stats_dim
    model = DegradationEstimator(input_dim=dim).to(device)
    loss_fn = DegradationEstimatorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            stats = batch["stats"].to(device)
            targets = {
                "noise_target": batch["noise_target"].view(-1, 1).to(device),
                "snr_target": normalize_snr(batch["snr_target"].view(-1, 1).to(device)),
                "snr_valid": batch["snr_valid"].view(-1, 1).to(device),
                "bandwidth_target": batch["bandwidth_target"].to(device),
                "bit_target": batch["bit_target"].to(device),
            }
            preds = model(stats)
            loss, _ = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                stats = batch["stats"].to(device)
                targets = {
                    "noise_target": batch["noise_target"].view(-1, 1).to(device),
                    "snr_target": normalize_snr(batch["snr_target"].view(-1, 1).to(device)),
                    "snr_valid": batch["snr_valid"].view(-1, 1).to(device),
                    "bandwidth_target": batch["bandwidth_target"].to(device),
                    "bit_target": batch["bit_target"].to(device),
                }
                preds = model(stats)
                loss, _ = loss_fn(preds, targets)
                valid_loss += loss.item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "valid_loss": valid_loss / len(valid_loader),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={history[-1]['train_loss']:.4f}  valid_loss={history[-1]['valid_loss']:.4f}")

    t_total = time.perf_counter() - t0

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": dim,
        "hidden_dim": model.hidden_dim,
        "epoch": epochs,
        "seed": seed,
    }, output_dir / "best_model.pt")

    with open(output_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss"])
        w.writeheader()
        w.writerows(history)

    return model, dim, t_total


def evaluate_and_plot(model, cache_dir: Path, output_dir: Path, seed: int):
    from work2.datasets.degradation_feature_dataset import DegradationFeatureDataset

    device = torch.device("cpu")
    idx = cache_dir / "index.jsonl"
    test_ds = DegradationFeatureDataset(str(idx), str(cache_dir), split="test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model.eval()

    all_noise_prob = []
    all_noise_true = []
    all_bw_pred = []
    all_bw_true = []
    all_bit_pred = []
    all_bit_true = []
    all_snr_pred = []
    all_snr_true = []
    all_snr_valid = []
    all_sev = []

    with torch.no_grad():
        for batch in test_loader:
            stats = batch["stats"].to(device)
            preds = model(stats)

            all_noise_prob.append(torch.sigmoid(preds["noise_logit"]).cpu().view(-1))
            all_noise_true.append(batch["noise_target"].view(-1))
            all_bw_pred.append(preds["bandwidth_logits"].argmax(dim=1).cpu())
            all_bw_true.append(batch["bandwidth_target"].view(-1))
            all_bit_pred.append(preds["bit_logits"].argmax(dim=1).cpu())
            all_bit_true.append(batch["bit_target"].view(-1))
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
    noise_f1 = f1_score(noise_true, noise_pred, zero_division=0)
    bw_acc = accuracy_score(bw_true, bw_pred)
    bw_f1 = f1_score(bw_true, bw_pred, average="macro", zero_division=0)
    bit_acc = accuracy_score(bit_true, bit_pred)
    bit_f1 = f1_score(bit_true, bit_pred, average="macro", zero_division=0)

    snr_mae = None
    if snr_valid.any():
        snr_p = torch.cat(all_snr_pred).numpy().flatten()
        snr_t = torch.cat(all_snr_true).numpy().flatten()
        snr_pred_db = denormalize_snr(torch.tensor(snr_p[snr_valid])).numpy()
        snr_true_db = snr_t[snr_valid]
        snr_mae = mean_absolute_error(snr_true_db, snr_pred_db)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _plot_cm(true, pred, labels, title, fp):
        cm = confusion_matrix(true, pred, labels=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(fp, dpi=150); plt.close(fig)

    _plot_cm(noise_true, noise_pred, ["Clean", "Noisy"], "Noise Confusion", fig_dir / "noise_cm.png")
    _plot_cm(bw_true, bw_pred, ["Full", "6k", "4k"], "Bandwidth Confusion", fig_dir / "bw_cm.png")
    _plot_cm(bit_true, bit_pred, ["16b", "12b", "10b", "8b"], "Bit-depth Confusion", fig_dir / "bit_cm.png")

    if snr_mae is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(snr_true_db, snr_pred_db, alpha=0.5, s=8, color="steelblue")
        ax.plot([0, 40], [0, 40], "r--", alpha=0.5)
        ax.set_xlabel("True SNR (dB)"); ax.set_ylabel("Predicted SNR (dB)")
        ax.set_title("SNR Prediction")
        fig.tight_layout(); fig.savefig(fig_dir / "snr_scatter.png", dpi=150); plt.close(fig)

    # Training curves
    hist_csv = output_dir / "history.csv"
    if hist_csv.exists():
        rows = []
        with open(hist_csv, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if rows:
            epochs_v = [int(r["epoch"]) for r in rows]
            t_loss = [float(r["train_loss"]) for r in rows]
            v_loss = [float(r["valid_loss"]) for r in rows]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs_v, t_loss, label="Train")
            ax.plot(epochs_v, v_loss, label="Valid")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Curves")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=150); plt.close(fig)

    # Severity breakdown
    sev_breakdown = []
    for sev in ["clean", "light", "medium", "heavy"]:
        mask = np.array([s == sev for s in all_sev])
        if not mask.any():
            continue
        sev_breakdown.append({
            "severity": sev,
            "n": int(mask.sum()),
            "noise_acc": round(accuracy_score(noise_true[mask], noise_pred[mask]), 4),
            "bw_acc": round(accuracy_score(bw_true[mask], bw_pred[mask]), 4),
            "bit_acc": round(accuracy_score(bit_true[mask], bit_pred[mask]), 4),
        })

    metrics = {
        "note": "SYNTHETIC DATA ONLY - does not represent real speech performance",
        "noise_accuracy": round(noise_acc, 4),
        "noise_f1": round(noise_f1, 4),
        "bandwidth_accuracy": round(bw_acc, 4),
        "bandwidth_f1": round(bw_f1, 4),
        "bit_accuracy": round(bit_acc, 4),
        "bit_f1": round(bit_f1, 4),
        "snr_mae_db": round(float(snr_mae), 4) if snr_mae is not None else None,
        "by_severity": sev_breakdown,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["severity", "noise_true", "noise_pred_prob", "bw_true", "bw_pred", "bit_true", "bit_pred"])
        for i in range(len(all_sev)):
            w.writerow([all_sev[i], noise_true[i], round(float(noise_prob[i]), 4),
                        bw_true[i], bw_pred[i], bit_true[i], bit_pred[i]])

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/estimator_smoke")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    ckpt_dir = out_dir / "checkpoints"

    print(f"[INFO] GTCRN Feature Extractor smoke test")
    print(f"[INFO] num_samples={args.num_samples}, seed={args.seed}")
    print("[WARN] Using SYNTHETIC pseudo-speech. Results do NOT represent real performance.")

    seed_everything(args.seed)

    print("[INFO] Loading GTCRN...")
    extractor = FrozenGTCRNFeatureExtractor("checkpoints/model_trained_on_dns3.tar", "cpu")
    stats_dim = extractor.get_stats_dim()
    print(f"[INFO] Stats dimension: {stats_dim}")

    print("[INFO] Generating synthetic data...")
    n_src = args.num_samples // 4
    sigs = generate_pseudo_speech(n_src, 2.0, 16000, args.seed)
    noises = [torch.randn(s.numel(), dtype=torch.float32) * 0.1 for s in sigs]
    print(f"[INFO] Generated {len(sigs)} source signals")

    print("[INFO] Caching features...")
    t_cache = time.perf_counter()
    entries = cache_from_synthetic(sigs, noises, extractor, cache_dir, args.seed)
    t_cache = time.perf_counter() - t_cache
    total = len(entries)
    splits = {}
    for e in entries:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
    print(f"[INFO] Cached {total} samples in {t_cache:.1f}s (train={splits.get('train',0)}, valid={splits.get('valid',0)}, test={splits.get('test',0)})")

    print("[INFO] Training...")
    model, dim, t_train = train_model(cache_dir, ckpt_dir, epochs=30, batch_size=64, lr=0.001, seed=args.seed)
    print(f"[INFO] Training complete in {t_train:.1f}s")

    print("[INFO] Evaluating...")
    metrics = evaluate_and_plot(model, cache_dir, ckpt_dir, args.seed)

    print("\n=== SMOKE TEST RESULTS (synthetic only) ===")
    print(f"  Noise accuracy: {metrics['noise_accuracy']:.4f}")
    print(f"  Noise F1: {metrics['noise_f1']:.4f}")
    print(f"  Bandwidth accuracy: {metrics['bandwidth_accuracy']:.4f}")
    print(f"  Bandwidth F1: {metrics['bandwidth_f1']:.4f}")
    print(f"  Bit accuracy: {metrics['bit_accuracy']:.4f}")
    print(f"  Bit F1: {metrics['bit_f1']:.4f}")
    if metrics.get('snr_mae_db') is not None:
        print(f"  SNR MAE: {metrics['snr_mae_db']:.2f} dB")
    print(f"  Caching time: {t_cache:.1f}s")
    print(f"  Training time: {t_train:.1f}s")

    # Single sample inference time
    model.eval()
    dummy_stats = torch.randn(1, dim)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_stats)
        times.append(time.perf_counter() - t0)
    t_infer = np.mean(times) * 1000
    print(f"  Single sample inference: {t_infer:.3f} ms")
    print(f"[DONE] Outputs in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
