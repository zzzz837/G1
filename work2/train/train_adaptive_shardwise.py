"""
Shard-wise training for adaptive residual refinement.

This version avoids per-sample Dataset/DataLoader indexing for large spectrogram tensors.
Instead, it:
  1. loads one adaptive shard file at a time
  2. optionally shuffles shard order
  3. splits each shard tensor into mini-batches in-memory
  4. trains sequentially on those mini-batches

This is much faster on Windows + CPU for large spectrogram training.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from work2.data.random_utils import seed_everything
from work2.models.gtcrn_adaptive import GTCRNAdaptive
from work2.losses.residual_loss import AdaptiveResidualLoss


def load_index(index_path: Path, split: str):
    shard_to_entries = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("split") != split:
                continue
            shard_to_entries.setdefault(e["shard_file"], []).append(e["offset"])
    if not shard_to_entries:
        raise ValueError(f"No entries for split={split} in {index_path}")
    return shard_to_entries


def split_minibatches(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        yield start, end


def run_split_epoch(model, shard_map, cache_dir: Path, loss_fn, optimizer, device, batch_size: int, train: bool, log_interval: int, max_shards=None, phase_name="train"):
    if train:
        model.train()
        model.estimator.eval()
        model.residual_module.train()
    else:
        model.eval()

    shard_files = list(shard_map.keys())
    if train:
        random.shuffle(shard_files)
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    total_loss = 0.0
    comp_sum = {"mag": 0.0, "hf": 0.0, "res": 0.0}
    processed_batches = 0
    processed_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for shard_idx, shard_file in enumerate(shard_files, start=1):
            try:
                shard = torch.load(cache_dir / shard_file, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[FATAL] Failed to load shard {shard_file}: {e}", file=sys.stderr)
                raise

            offsets = sorted(shard_map[shard_file])
            # gather this split's samples from current shard
            clean_spec = shard["clean_spec"][offsets]
            degraded_spec = shard["degraded_spec"][offsets]
            bandwidth_limited = shard["bandwidth_limited"][offsets]

            n_items = clean_spec.shape[0]
            if n_items == 0:
                continue

            for start, end in split_minibatches(n_items, batch_size):
                try:
                    clean_mb = clean_spec[start:end].to(device)
                    degraded_mb = degraded_spec[start:end].to(device)
                    bw_mb = bandwidth_limited[start:end].to(device)

                    out = model(degraded_mb)
                    enhanced_final = out["enhanced_final"].permute(0, 3, 2, 1)
                    residual = out["residual"].permute(0, 3, 2, 1)
                    clean_perm = clean_mb.permute(0, 3, 2, 1)

                    loss, comps = loss_fn(enhanced_final, clean_perm, residual, bw_mb)

                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.residual_module.parameters(), 1.0)
                        optimizer.step()

                    total_loss += loss.item()
                    for k in comp_sum:
                        comp_sum[k] += comps[k]
                    processed_batches += 1
                    processed_samples += (end - start)

                    if processed_batches % log_interval == 0 or processed_batches == 1:
                        print(f"[{phase_name}] shard {shard_idx}/{len(shard_files)} batch {processed_batches} samples={processed_samples} loss={loss.item():.4f} mag={comps['mag']:.4f} hf={comps['hf']:.4f} res={comps['res']:.6f}", flush=True)
                except Exception as batch_err:
                    print(f"[FATAL] {phase_name} failed in shard {shard_file} batch offsets {start}:{end}: {batch_err}", file=sys.stderr, flush=True)
                    raise

    if processed_batches == 0:
        raise RuntimeError(f"No batch processed in split {phase_name}")

    return {
        "loss": total_loss / processed_batches,
        "mag": comp_sum["mag"] / processed_batches,
        "hf": comp_sum["hf"] / processed_batches,
        "res": comp_sum["res"] / processed_batches,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Shard-wise adaptive residual training")
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_residual_shardwise")
    parser.add_argument("--residual-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--max-valid-shards", type=int, default=None)
    parser.add_argument("--max-test-shards", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")
    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"

    train_map = load_index(index_path, "train")
    valid_map = load_index(index_path, "valid")
    test_map = load_index(index_path, "test")

    def count_samples(m):
        return sum(len(v) for v in m.values())

    model = GTCRNAdaptive(checkpoint_path=args.gtcrn_checkpoint, device="cpu", freeze_gtcrn=True, freeze_estimator=True, residual_version=args.residual_version)
    if Path(args.estimator_checkpoint).exists():
        est_ckpt = torch.load(args.estimator_checkpoint, map_location="cpu")
        model.estimator.load_state_dict(est_ckpt["model_state_dict"])
    model.train_residual_only()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_valid = float("inf")
    history = []
    if args.resume and (out_dir / "last_model.pt").exists():
        ckpt = torch.load(out_dir / "last_model.pt", map_location="cpu")
        model.residual_module.load_state_dict(ckpt["residual_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO] Resumed from epoch {start_epoch}")

    counts = model.count_params()
    print(f"[INFO] Train shards={len(train_map)} valid shards={len(valid_map)} test shards={len(test_map)}")
    print(f"[INFO] Train samples={count_samples(train_map)} valid samples={count_samples(valid_map)} test samples={count_samples(test_map)}")
    print(f"[INFO] Residual params: {counts['residual_trainable']}")

    loss_fn = AdaptiveResidualLoss()
    optimizer = torch.optim.AdamW(model.residual_module.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6)

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({
            "epochs": args.epochs,
            "mini_batch_size": args.mini_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "train_samples": count_samples(train_map),
            "valid_samples": count_samples(valid_map),
            "test_samples": count_samples(test_map),
            "train_shards": len(train_map),
            "valid_shards": len(valid_map),
            "test_shards": len(test_map),
            "residual_params": counts['residual_trainable'],
            "residual_version": args.residual_version,
            "cache_dir": str(cache_dir),
            "cache_mode": "adaptive_shardwise",
        }, f, indent=2)

    patience_counter = 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        print("[INFO] Starting train epoch...", flush=True)
        train_metrics = run_split_epoch(model, train_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=True, log_interval=args.log_interval, max_shards=args.max_train_shards, phase_name="train")
        print(f"[INFO] Train epoch done: loss={train_metrics['loss']:.4f}", flush=True)
        print("[INFO] Starting valid epoch...", flush=True)
        valid_metrics = run_split_epoch(model, valid_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=False, log_interval=max(args.log_interval, 1000), max_shards=args.max_valid_shards, phase_name="valid")
        print(f"[INFO] Valid epoch done: loss={valid_metrics['loss']:.4f}", flush=True)

        scheduler.step(valid_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_metrics['loss']:.4f} | valid_loss={valid_metrics['loss']:.4f} | mag={valid_metrics['mag']:.4f} | hf={valid_metrics['hf']:.4f} | res={valid_metrics['res']:.6f}", flush=True)

        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            patience_counter = 0
            torch.save({
                "residual_state_dict": model.residual_module.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "validation_metrics": valid_metrics,
            }, out_dir / "best_model.pt")
            print("[INFO] Saved best_model.pt", flush=True)
        else:
            patience_counter += 1

        torch.save({
            "residual_state_dict": model.residual_module.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
        }, out_dir / "last_model.pt")
        print("[INFO] Saved last_model.pt", flush=True)

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}", flush=True)
            break

    t_total = time.perf_counter() - t_start
    print("[INFO] Starting test epoch...", flush=True)
    test_metrics = run_split_epoch(model, test_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=False, log_interval=10**9, max_shards=args.max_test_shards, phase_name="test")
    print(f"[INFO] Test: loss={test_metrics['loss']:.4f} mag={test_metrics['mag']:.4f} hf={test_metrics['hf']:.4f} res={test_metrics['res']:.6f}", flush=True)

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "train_mag", "valid_mag", "train_hf", "valid_hf", "train_res", "valid_res"])
        writer.writeheader()
        for h in history:
            writer.writerow({
                "epoch": h["epoch"],
                "train_loss": h["train"]["loss"],
                "valid_loss": h["valid"]["loss"],
                "train_mag": h["train"]["mag"],
                "valid_mag": h["valid"]["mag"],
                "train_hf": h["train"]["hf"],
                "valid_hf": h["valid"]["hf"],
                "train_res": h["train"]["res"],
                "valid_res": h["valid"]["res"],
            })

    with open(out_dir / "train_summary.json", "w") as f:
        json.dump({
            "epochs": epoch,
            "total_training_time_s": round(t_total, 2),
            "best_valid_loss": best_valid,
            "test_metrics": test_metrics,
            "residual_params": counts['residual_trainable'],
            "residual_version": args.residual_version,
            "early_stopped": patience_counter >= args.patience,
        }, f, indent=2)

    print(f"[INFO] Total training time: {t_total:.1f}s")
    print(f"[INFO] Outputs saved to {out_dir}")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
