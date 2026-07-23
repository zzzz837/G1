"""
Fast cached training for adaptive residual modules using existing adaptive_cache_sharded.

Uses ONLY cached tensors:
- clean_spec
- enhanced_base
- oracle degradation labels -> condition vector
- bandwidth_limited

No GTCRN / estimator forward is executed during training.
This is the recommended fast training path for V1/V2 residual experiments.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

from work2.data.random_utils import seed_everything
from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2Formal
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


def build_oracle_condition(noise_target, snr_target, bandwidth_target, bit_target):
    """
    Build 9-dim oracle condition vector:
      [noise(1), snr_norm(1), bw_onehot(3), bit_onehot(4)]
    Inputs are batch tensors from cached shard.
    """
    B = noise_target.shape[0]
    bw_oh = torch.zeros(B, 3, dtype=torch.float32, device=noise_target.device)
    bw_oh.scatter_(1, bandwidth_target.view(-1, 1), 1.0)
    bit_oh = torch.zeros(B, 4, dtype=torch.float32, device=noise_target.device)
    bit_oh.scatter_(1, bit_target.view(-1, 1), 1.0)
    snr_norm = torch.clamp(snr_target / 40.0, 0.0, 1.0)
    return torch.cat([noise_target, snr_norm, bw_oh, bit_oh], dim=1)


def run_split_epoch(residual_module, shard_map, cache_dir: Path, loss_fn, optimizer, device, batch_size: int, train: bool, log_interval: int, max_shards=None, phase_name="train"):
    if train:
        residual_module.train()
    else:
        residual_module.eval()

    shard_files = list(shard_map.keys())
    if train:
        random.shuffle(shard_files)
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    total_loss = 0.0
    comp_sum = {"mag": 0.0, "hf": 0.0, "res": 0.0}
    alpha_sum = None
    residual_ratio_sum = 0.0
    processed_batches = 0
    processed_samples = 0

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for shard_idx, shard_file in enumerate(shard_files, start=1):
            shard = torch.load(cache_dir / shard_file, map_location="cpu", weights_only=False)
            offsets = list(shard_map[shard_file])
            if train:
                random.shuffle(offsets)
            else:
                offsets = sorted(offsets)

            clean_spec = shard["clean_spec"][offsets]          # (N,F,T,2)
            enhanced_base = shard["enhanced_base"][offsets]    # (N,F,T,2)
            noise_target = shard["noise_target"][offsets].to(torch.float32)   # (N,1)
            snr_target = shard["snr_target"][offsets].to(torch.float32)       # (N,1)
            bandwidth_target = shard["bandwidth_target"][offsets].to(torch.long)
            bit_target = shard["bit_target"][offsets].to(torch.long)
            bandwidth_limited = shard["bandwidth_limited"][offsets].to(torch.bool)

            n_items = clean_spec.shape[0]
            if n_items == 0:
                continue

            for start, end in split_minibatches(n_items, batch_size):
                try:
                    clean_mb = clean_spec[start:end].to(device)           # (B,F,T,2)
                    base_mb = enhanced_base[start:end].to(device)         # (B,F,T,2)
                    noise_mb = noise_target[start:end].to(device)         # (B,1)
                    snr_mb = snr_target[start:end].to(device)             # (B,1)
                    bw_mb_cls = bandwidth_target[start:end].to(device)    # (B,)
                    bit_mb_cls = bit_target[start:end].to(device)         # (B,)
                    bw_limited_mb = bandwidth_limited[start:end].to(device)

                    cond_mb = build_oracle_condition(noise_mb, snr_mb, bw_mb_cls, bit_mb_cls)

                    # residual module expects (B,2,T,F)
                    clean_perm = clean_mb.permute(0, 3, 2, 1).contiguous()
                    base_perm = base_mb.permute(0, 3, 2, 1).contiguous()

                    out = residual_module(base_perm, cond_mb)
                    loss, comps = loss_fn(out["enhanced_final"], clean_perm, out["residual"], bw_limited_mb)

                    alpha_tensor = out["alpha"].detach()
                    alpha_mean = alpha_tensor.mean(dim=0)
                    if alpha_sum is None:
                        alpha_sum = torch.zeros_like(alpha_mean)
                    alpha_sum += alpha_mean.cpu()
                    residual_rms = out["residual"].detach().pow(2).mean().sqrt()
                    base_rms = base_perm.detach().pow(2).mean().sqrt()
                    residual_ratio = float((residual_rms / (base_rms + 1e-8)).item())

                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(residual_module.parameters(), 1.0)
                        optimizer.step()

                    total_loss += loss.item()
                    for k in comp_sum:
                        comp_sum[k] += comps[k]
                    residual_ratio_sum += residual_ratio
                    processed_batches += 1
                    processed_samples += (end - start)

                    if processed_batches % log_interval == 0 or processed_batches == 1:
                        if alpha_mean.numel() == 3:
                            print(
                                f"[{phase_name}] shard {shard_idx}/{len(shard_files)} batch {processed_batches} samples={processed_samples} "
                                f"loss={loss.item():.4f} mag={comps['mag']:.4f} hf={comps['hf']:.4f} res={comps['res']:.6f} "
                                f"alpha_low={alpha_mean[0].item():.4f} alpha_mid={alpha_mean[1].item():.4f} alpha_high={alpha_mean[2].item():.4f} "
                                f"res_ratio={residual_ratio:.4f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[{phase_name}] shard {shard_idx}/{len(shard_files)} batch {processed_batches} samples={processed_samples} "
                                f"loss={loss.item():.4f} mag={comps['mag']:.4f} hf={comps['hf']:.4f} res={comps['res']:.6f} res_ratio={residual_ratio:.4f}",
                                flush=True,
                            )
                except Exception as batch_err:
                    print(f"[FATAL] {phase_name} failed in shard {shard_file} batch offsets {start}:{end}: {batch_err}", file=sys.stderr, flush=True)
                    raise

    if processed_batches == 0:
        raise RuntimeError(f"No batch processed in split {phase_name}")

    alpha_avg = alpha_sum / processed_batches if alpha_sum is not None else None
    out = {
        "loss": total_loss / processed_batches,
        "mag": comp_sum["mag"] / processed_batches,
        "hf": comp_sum["hf"] / processed_batches,
        "res": comp_sum["res"] / processed_batches,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "residual_ratio": residual_ratio_sum / processed_batches,
    }
    if alpha_avg is not None and alpha_avg.numel() == 3:
        out.update({
            "alpha_low": float(alpha_avg[0].item()),
            "alpha_mid": float(alpha_avg[1].item()),
            "alpha_high": float(alpha_avg[2].item()),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Fast cached training for adaptive residual V1/V2")
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_residual_cached")
    parser.add_argument("--residual-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--condition-source", type=str, default="oracle", choices=["oracle"], help="Current cached training uses oracle conditions from cached labels")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--max-valid-shards", type=int, default=None)
    parser.add_argument("--max-test-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    seed_everything(args.seed)
    device = torch.device("cpu")

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"
    train_map = load_index(index_path, "train")
    valid_map = load_index(index_path, "valid")
    test_map = load_index(index_path, "test")

    def count_samples(m):
        return sum(len(v) for v in m.values())

    if args.residual_version == "v1":
        residual_module = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    else:
        residual_module = AdaptiveResidualModuleV2Formal(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_valid = float("inf")
    history = []
    if args.resume and (out_dir / "last_model.pt").exists():
        ckpt = torch.load(out_dir / "last_model.pt", map_location="cpu")
        residual_module.load_state_dict(ckpt["residual_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO] Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in residual_module.parameters() if p.requires_grad)
    print(f"[INFO] Residual version={args.residual_version}")
    print(f"[INFO] Condition source={args.condition_source}")
    print(f"[INFO] Train shards={len(train_map)} valid shards={len(valid_map)} test shards={len(test_map)}")
    print(f"[INFO] Train samples={count_samples(train_map)} valid samples={count_samples(valid_map)} test samples={count_samples(test_map)}")
    print(f"[INFO] Residual trainable params: {n_params}")
    print(f"[INFO] CPU threads: {torch.get_num_threads()}")

    loss_fn = AdaptiveResidualLoss()
    optimizer = torch.optim.AdamW(residual_module.parameters(), lr=args.learning_rate)
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
            "residual_params": n_params,
            "residual_version": args.residual_version,
            "condition_source": args.condition_source,
            "cache_dir": str(cache_dir),
            "cache_mode": "adaptive_cached_oracle",
        }, f, indent=2)

    patience_counter = 0
    t_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        print("[INFO] Starting train epoch...", flush=True)
        train_metrics = run_split_epoch(residual_module, train_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=True, log_interval=args.log_interval, max_shards=args.max_train_shards, phase_name="train")
        print(f"[INFO] Train epoch done: loss={train_metrics['loss']:.4f}", flush=True)
        print("[INFO] Starting valid epoch...", flush=True)
        valid_metrics = run_split_epoch(residual_module, valid_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=False, log_interval=max(args.log_interval, 1000), max_shards=args.max_valid_shards, phase_name="valid")
        print(f"[INFO] Valid epoch done: loss={valid_metrics['loss']:.4f}", flush=True)

        scheduler.step(valid_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        extra = ""
        if "alpha_low" in valid_metrics:
            extra += f" | alpha=({valid_metrics['alpha_low']:.3f},{valid_metrics['alpha_mid']:.3f},{valid_metrics['alpha_high']:.3f})"
        extra += f" | res_ratio={valid_metrics['residual_ratio']:.4f}"
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_metrics['loss']:.4f} | valid_loss={valid_metrics['loss']:.4f} | mag={valid_metrics['mag']:.4f} | hf={valid_metrics['hf']:.4f} | res={valid_metrics['res']:.6f}{extra}", flush=True)

        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            patience_counter = 0
            torch.save({
                "residual_state_dict": residual_module.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "validation_metrics": valid_metrics,
                "residual_version": args.residual_version,
            }, out_dir / "best_model.pt")
            print("[INFO] Saved best_model.pt", flush=True)
        else:
            patience_counter += 1

        torch.save({
            "residual_state_dict": residual_module.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "residual_version": args.residual_version,
        }, out_dir / "last_model.pt")
        print("[INFO] Saved last_model.pt", flush=True)

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}", flush=True)
            break

    t_total = time.perf_counter() - t_start
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    residual_module.load_state_dict(best_ckpt["residual_state_dict"])
    print(f"[INFO] Loaded best model from epoch {best_ckpt.get('epoch')} for final test", flush=True)
    print("[INFO] Starting test epoch...", flush=True)
    test_metrics = run_split_epoch(residual_module, test_map, cache_dir, loss_fn, optimizer, device, args.mini_batch_size, train=False, log_interval=10**9, max_shards=args.max_test_shards, phase_name="test")
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
            "residual_params": n_params,
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
