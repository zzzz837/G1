"""
Fast shard-based training for adaptive residual refinement.
"""
import argparse
import json
import sys
import time
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.models.gtcrn_adaptive import GTCRNAdaptive
from work2.losses.residual_loss import AdaptiveResidualLoss
from work2.datasets.adaptive_residual_shard_dataset import AdaptiveResidualShardDataset


def collate_fn(batch):
    return {
        "clean_spec": torch.stack([b["clean_spec"] for b in batch], dim=0),
        "degraded_spec": torch.stack([b["degraded_spec"] for b in batch], dim=0),
        "enhanced_base": torch.stack([b["enhanced_base"] for b in batch], dim=0),
        "noise_target": torch.stack([b["noise_target"] for b in batch], dim=0),
        "snr_target": torch.stack([b["snr_target"] for b in batch], dim=0),
        "snr_valid": torch.stack([b["snr_valid"] for b in batch], dim=0),
        "bandwidth_target": torch.tensor([int(b["bandwidth_target"]) for b in batch], dtype=torch.long),
        "bit_target": torch.tensor([int(b["bit_target"]) for b in batch], dtype=torch.long),
        "bandwidth_limited": torch.tensor([bool(b["bandwidth_limited"]) for b in batch], dtype=torch.bool),
        "severity": [b["severity"] for b in batch],
    }


def run_epoch(model, loader, loss_fn, optimizer, device, train=True, log_interval=50, max_batches=None, phase_name=None):
    if train:
        model.train()
        model.estimator.eval()
        model.residual_module.train()
    else:
        model.eval()

    total_loss = 0.0
    comp_sum = {"mag": 0.0, "hf": 0.0, "res": 0.0}
    processed = 0
    phase = phase_name or ("train" if train else "valid")

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            if max_batches is not None and step > max_batches:
                break
            try:
                degraded_spec = batch["degraded_spec"].to(device)
                clean_spec = batch["clean_spec"].to(device)
                bw_limited = batch["bandwidth_limited"].to(device)

                out = model(degraded_spec)
                enhanced_final = out["enhanced_final"].permute(0, 3, 2, 1)
                residual = out["residual"].permute(0, 3, 2, 1)
                clean_perm = clean_spec.permute(0, 3, 2, 1)

                loss, comps = loss_fn(enhanced_final, clean_perm, residual, bw_limited)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.residual_module.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                processed += 1
                for k in comp_sum:
                    comp_sum[k] += comps[k]

                if step % log_interval == 0 or step == 1:
                    denom = min(len(loader), max_batches) if max_batches is not None else len(loader)
                    print(f"[{phase}] batch {step}/{denom} loss={loss.item():.4f} mag={comps['mag']:.4f} hf={comps['hf']:.4f} res={comps['res']:.6f}")
            except Exception as batch_err:
                print(f"[FATAL] {phase} batch {step} failed: {batch_err}", file=sys.stderr)
                raise

    if processed == 0:
        raise RuntimeError(f"No batch processed in run_epoch ({phase})")
    return {"loss": total_loss / processed, **{k: v / processed for k, v in comp_sum.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_residual_fast")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--no-shuffle-train", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")
    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"

    train_ds = AdaptiveResidualShardDataset(str(index_path), str(cache_dir), split="train")
    valid_ds = AdaptiveResidualShardDataset(str(index_path), str(cache_dir), split="valid")
    test_ds = AdaptiveResidualShardDataset(str(index_path), str(cache_dir), split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(not args.no_shuffle_train), num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = GTCRNAdaptive(checkpoint_path=args.gtcrn_checkpoint, device="cpu", freeze_gtcrn=True, freeze_estimator=True)
    if Path(args.estimator_checkpoint).exists():
        ckpt_est = torch.load(args.estimator_checkpoint, map_location="cpu")
        model.estimator.load_state_dict(ckpt_est["model_state_dict"])
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
    print(f"[INFO] Dataset: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    print(f"[INFO] Residual params: {counts['residual_trainable']}")
    print(f"[INFO] Train loader batches: {len(train_loader)} | Valid loader batches: {len(valid_loader)} | Test loader batches: {len(test_loader)}")

    loss_fn = AdaptiveResidualLoss()
    optimizer = torch.optim.AdamW(model.residual_module.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6)

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "train_samples": len(train_ds),
            "valid_samples": len(valid_ds),
            "test_samples": len(test_ds),
            "residual_params": counts['residual_trainable'],
            "cache_dir": str(cache_dir),
            "cache_mode": "adaptive_sharded",
        }, f, indent=2)

    patience_counter = 0
    t_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        try:
            print("[INFO] Starting train epoch...", flush=True)
            train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, log_interval=args.log_interval, max_batches=args.max_train_batches, phase_name="train")
            print(f"[INFO] Train epoch done: loss={train_metrics['loss']:.4f}", flush=True)
            print("[INFO] Starting valid epoch...", flush=True)
            valid_metrics = run_epoch(model, valid_loader, loss_fn, optimizer, device, train=False, log_interval=max(args.log_interval, 1000), max_batches=args.max_valid_batches, phase_name="valid")
            print(f"[INFO] Valid epoch done: loss={valid_metrics['loss']:.4f}", flush=True)
        except BaseException as epoch_err:
            print(f"[FATAL] Epoch {epoch} failed: {repr(epoch_err)}", file=sys.stderr, flush=True)
            raise
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
    test_metrics = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False, log_interval=10**9, phase_name="test")
    print(f"[INFO] Test: loss={test_metrics['loss']:.4f} mag={test_metrics['mag']:.4f} hf={test_metrics['hf']:.4f} res={test_metrics['res']:.6f}")

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
