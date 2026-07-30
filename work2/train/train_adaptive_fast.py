"""
Fast shard-based training for adaptive residual refinement.
"""
import argparse
import faulthandler
import inspect
import json
import sys
import time
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from work2.data.random_utils import seed_everything
from work2.data.shard_batch_sampler import ShardBatchSampler
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


def run_epoch(model, loader, loss_fn, optimizer, device, residual_scale: float | None, train=True, log_interval=50, max_batches=None, phase_name=None, force_residual_scale: float | None = None):
    if train:
        model.eval()
        model.residual_module.train()
    else:
        model.eval()

    total_loss = 0.0
    component_names = ["mag", "hf", "complex", "protect", "res", "gate", "ratio", "si_sdr", "delta", "scale", "scale_smooth", "scale_order", "scale_mean", "scale_min", "scale_max"]
    comp_sum = {name: 0.0 for name in component_names}
    processed = 0
    phase = phase_name or ("train" if train else "valid")
    scale_by_severity = {"clean": [], "light": [], "medium": [], "heavy": []}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        loader_iter = iter(loader)
        step = 0
        while True:
            if max_batches is not None and step >= max_batches:
                break

            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except BaseException as load_err:
                print(
                    f"[FATAL] {phase} failed while loading batch {step + 1}: {repr(load_err)}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

            step += 1
            try:
                degraded_spec = batch["degraded_spec"].to(device)
                clean_spec = batch["clean_spec"].to(device)
                bw_limited = batch["bandwidth_limited"].to(device)

                out = model(degraded_spec, force_residual_scale=force_residual_scale)
                enhanced_base = out["enhanced_base"].permute(0, 3, 2, 1)
                enhanced_final = out["enhanced_final"].permute(0, 3, 2, 1)
                residual = enhanced_final - enhanced_base
                dynamic_scale = out.get("dynamic_scale")
                clean_perm = clean_spec.permute(0, 3, 2, 1)

                loss, comps = loss_fn(
                    enhanced_final,
                    clean_perm,
                    residual,
                    bw_limited,
                    enhanced_base=enhanced_base,
                    severity=batch["severity"],
                    global_alpha=out.get("global_alpha"),
                    dynamic_scale=dynamic_scale,
                )
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.residual_module.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                processed += 1
                for k in comp_sum:
                    if k in comps:
                        value = comps[k]
                        if torch.is_tensor(value):
                            value = value.detach().item()
                        comp_sum[k] += float(value)

                if dynamic_scale is not None:
                    scales = dynamic_scale.detach().view(-1).cpu()
                    for sev, scale in zip(batch["severity"], scales):
                        scale_by_severity[str(sev)].append(float(scale.item()))

                if step % log_interval == 0 or step == 1:
                    denom = min(len(loader), max_batches) if max_batches is not None else len(loader)
                    print(
                        f"[{phase}] batch {step}/{denom} "
                        f"loss={loss.item():.4f} "
                        f"mag={comps['mag']:.4f} "
                        f"hf={comps['hf']:.4f} "
                        f"complex={comps['complex']:.4f} "
                        f"protect={comps['protect']:.6f} "
                        f"ratio={comps['ratio']:.6f} "
                        f"gate={comps['gate']:.6f} "
                        f"si_sdr={comps['si_sdr']:.4f} "
                        f"delta={comps['delta']:.4f} "
                        f"scale_mean={comps.get('scale_mean', 0.0):.3f} "
                        f"scale_min={comps.get('scale_min', 0.0):.3f} "
                        f"scale_max={comps.get('scale_max', 0.0):.3f} "
                        f"scale_loss={comps.get('scale', 0.0):.5f} "
                        f"scale_order={comps.get('scale_order', 0.0):.5f}"
                    )
            except Exception as batch_err:
                print(f"[FATAL] {phase} batch {step} failed: {batch_err}", file=sys.stderr)
                raise

    if processed == 0:
        raise RuntimeError(f"No batch processed in run_epoch ({phase})")

    for severity, values in scale_by_severity.items():
        if values:
            mean_scale = sum(values) / len(values)
            print(f"[SCALE][{phase}] {severity}: {mean_scale:.4f}")

    return {"loss": total_loss / processed, **{k: v / processed for k, v in comp_sum.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--train-cache-dir", type=str, default=None)
    parser.add_argument("--valid-cache-dir", type=str, default=None)
    parser.add_argument("--test-cache-dir", type=str, default=None)
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_residual_fast")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--residual-scale", type=float, default=0.65)
    parser.add_argument("--dynamic-scale-stage", type=str, default="fixed_pretrain", choices=["fixed_pretrain", "scale_only", "joint"])
    parser.add_argument("--fixed-scale", type=float, default=0.65)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    parser.add_argument("--residual-checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--no-shuffle-train", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")

    if args.cache_dir is not None:
        train_cache_dir = Path(args.cache_dir)
        valid_cache_dir = Path(args.cache_dir)
        test_cache_dir = Path(args.cache_dir)
    else:
        if args.train_cache_dir is None or args.valid_cache_dir is None:
            raise ValueError("Either --cache-dir or both --train-cache-dir and --valid-cache-dir are required")
        train_cache_dir = Path(args.train_cache_dir)
        valid_cache_dir = Path(args.valid_cache_dir)
        test_cache_dir = Path(args.test_cache_dir) if args.test_cache_dir is not None else None

    train_index_path = train_cache_dir / "index.jsonl"
    valid_index_path = valid_cache_dir / "index.jsonl"
    test_index_path = test_cache_dir / "index.jsonl" if test_cache_dir is not None else None

    train_ds = AdaptiveResidualShardDataset(str(train_index_path), str(train_cache_dir), split="train")
    valid_ds = AdaptiveResidualShardDataset(str(valid_index_path), str(valid_cache_dir), split="valid")
    test_ds = AdaptiveResidualShardDataset(str(test_index_path), str(test_cache_dir), split="test") if test_cache_dir is not None else None

    train_batch_sampler = ShardBatchSampler(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle_shards=not args.no_shuffle_train,
        shuffle_within_shard=not args.no_shuffle_train,
        drop_last=False,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn) if test_ds is not None else None

    model = GTCRNAdaptive(checkpoint_path=args.gtcrn_checkpoint, device="cpu", freeze_gtcrn=True, freeze_estimator=True)
    estimator_path = Path(args.estimator_checkpoint)
    if not estimator_path.exists():
        raise FileNotFoundError(f"Estimator checkpoint not found: {estimator_path}")
    ckpt_est = torch.load(estimator_path, map_location="cpu")
    model.estimator.load_state_dict(ckpt_est["model_state_dict"])
    model.train_residual_only()

    if args.dynamic_scale_stage == "fixed_pretrain":
        for param in model.residual_module.parameters():
            param.requires_grad = True
        for param in model.residual_module.scale_head.parameters():
            param.requires_grad = False
    elif args.dynamic_scale_stage == "scale_only":
        if args.residual_checkpoint is None:
            raise ValueError("--residual-checkpoint is required for scale_only")
        residual_path = Path(args.residual_checkpoint)
        if not residual_path.exists():
            raise FileNotFoundError(f"Residual checkpoint not found: {residual_path}")
        residual_ckpt = torch.load(residual_path, map_location="cpu")
        state_dict = residual_ckpt.get("residual_state_dict", residual_ckpt)
        model.residual_module.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded residual checkpoint: {residual_path.resolve()}")
        for param in model.residual_module.parameters():
            param.requires_grad = False
        for param in model.residual_module.scale_head.parameters():
            param.requires_grad = True
    elif args.dynamic_scale_stage == "joint":
        if args.residual_checkpoint is None:
            raise ValueError("--residual-checkpoint is required for joint")
        residual_path = Path(args.residual_checkpoint)
        if not residual_path.exists():
            raise FileNotFoundError(f"Residual checkpoint not found: {residual_path}")
        residual_ckpt = torch.load(residual_path, map_location="cpu")
        state_dict = residual_ckpt.get("residual_state_dict", residual_ckpt)
        model.residual_module.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded residual checkpoint: {residual_path.resolve()}")
        for param in model.residual_module.parameters():
            param.requires_grad = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fault_log = open(out_dir / "native_crash.log", "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=fault_log, all_threads=True)
    print(f"[INFO] Native crash log: {(out_dir / 'native_crash.log').resolve()}")

    start_epoch = 1
    best_valid = float("inf")
    history = []
    if args.resume and (out_dir / "last_model.pt").exists():
        ckpt = torch.load(out_dir / "last_model.pt", map_location="cpu")
        model.residual_module.load_state_dict(ckpt["residual_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO] Resumed from epoch {start_epoch}")

    counts = model.count_params()
    print(f"[INFO] Dataset: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds) if test_ds is not None else 0}")
    print(f"[INFO] Residual params: {counts['residual_trainable']}")
    print(f"[INFO] Train loader batches: {len(train_loader)} | Valid loader batches: {len(valid_loader)} | Test loader batches: {len(test_loader) if test_loader is not None else 0}")
    print(f"[INFO] Train shard sampler: shards={len(train_batch_sampler.shard_names)}, batches={len(train_batch_sampler)}, batch_size={args.batch_size}, drop_last={train_batch_sampler.drop_last}")
    print(f"[INFO] residual_scale={args.residual_scale}")
    print(f"[INFO] dynamic_scale_stage={args.dynamic_scale_stage}")
    print(f"[INFO] fixed_scale={args.fixed_scale}")
    print(f"[INFO] joint_learning_rate={args.joint_learning_rate}")
    print(f"[INFO] residual_module_type={type(model.residual_module).__name__}")
    print(f"[INFO] Estimator checkpoint: {estimator_path.resolve()}")
    print(f"[INFO] Loss source: {inspect.getfile(AdaptiveResidualLoss)}")

    loss_fn = AdaptiveResidualLoss(
        lambda_scale=0.0 if args.dynamic_scale_stage == "fixed_pretrain" else 0.02,
        lambda_scale_smooth=0.0 if args.dynamic_scale_stage == "fixed_pretrain" else 0.005,
        lambda_scale_order=0.0 if args.dynamic_scale_stage == "fixed_pretrain" else 0.01,
    )
    if args.dynamic_scale_stage == "fixed_pretrain":
        print(f"[INFO] force_residual_scale={args.fixed_scale}")
        print("[INFO] scale supervision disabled")
    print(
        f"[LOSS_CFG] lambda_mag={loss_fn.lambda_mag}, "
        f"lambda_hf={loss_fn.lambda_hf}, "
        f"lambda_complex={loss_fn.lambda_complex}, "
        f"lambda_protect={loss_fn.lambda_protect}, "
        f"lambda_res={loss_fn.lambda_res}, "
        f"lambda_ratio={loss_fn.lambda_ratio}, "
        f"lambda_si_sdr={loss_fn.lambda_si_sdr}, "
        f"lambda_delta={loss_fn.lambda_delta}, "
        f"lambda_gate={loss_fn.lambda_gate}, "
        f"lambda_scale={loss_fn.lambda_scale}, "
        f"lambda_scale_smooth={loss_fn.lambda_scale_smooth}, "
        f"hf_start_bin={loss_fn.hf_start_bin}"
    )
    if args.dynamic_scale_stage == "joint":
        optimizer_lr = args.joint_learning_rate
    else:
        optimizer_lr = args.learning_rate
    print(f"[INFO] optimizer_lr={optimizer_lr}")
    optimizer = torch.optim.AdamW([p for p in model.residual_module.parameters() if p.requires_grad], lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6)

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "joint_learning_rate": args.joint_learning_rate,
            "optimizer_lr": optimizer_lr,
            "seed": args.seed,
            "train_samples": len(train_ds),
            "valid_samples": len(valid_ds),
            "test_samples": len(test_ds) if test_ds is not None else 0,
            "residual_params": counts['residual_trainable'],
            "train_cache_dir": str(train_cache_dir),
            "valid_cache_dir": str(valid_cache_dir),
            "test_cache_dir": str(test_cache_dir) if test_cache_dir is not None else None,
            "cache_mode": "adaptive_sharded_split_dirs" if args.cache_dir is None else "adaptive_sharded",
            "dynamic_scale_stage": args.dynamic_scale_stage,
            "residual_scale": args.residual_scale,
            "fixed_scale": args.fixed_scale,
            "residual_checkpoint": args.residual_checkpoint,
        }, f, indent=2)

    patience_counter = 0
    t_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        try:
            print("[INFO] Starting train epoch...", flush=True)
            train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, residual_scale=args.residual_scale, train=True, log_interval=args.log_interval, max_batches=args.max_train_batches, phase_name="train", force_residual_scale=args.fixed_scale if args.dynamic_scale_stage == "fixed_pretrain" else None)
            print(f"[INFO] Train epoch done: loss={train_metrics['loss']:.4f}", flush=True)
            print("[INFO] Starting valid epoch...", flush=True)
            valid_metrics = run_epoch(model, valid_loader, loss_fn, optimizer, device, residual_scale=args.residual_scale, train=False, log_interval=max(args.log_interval, 1000), max_batches=args.max_valid_batches, phase_name="valid", force_residual_scale=args.fixed_scale if args.dynamic_scale_stage == "fixed_pretrain" else None)
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

    best_path = out_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
    best_ckpt = torch.load(best_path, map_location=device)
    model.residual_module.load_state_dict(best_ckpt["residual_state_dict"])
    print(f"[INFO] Loaded best checkpoint: epoch={best_ckpt.get('epoch')}, valid_loss={best_ckpt.get('validation_metrics', {}).get('loss')}")

    t_total = time.perf_counter() - t_start
    test_metrics = None
    if test_loader is not None:
        test_metrics = run_epoch(
            model,
            test_loader,
            loss_fn,
            optimizer=None,
            device=device,
            residual_scale=args.residual_scale,
            train=False,
            log_interval=10**9,
            phase_name="test",
            force_residual_scale=args.fixed_scale if args.dynamic_scale_stage == "fixed_pretrain" else None,
        )
        print(f"[INFO] Test: loss={test_metrics['loss']:.4f} mag={test_metrics['mag']:.4f} hf={test_metrics['hf']:.4f} res={test_metrics['res']:.6f}")

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "train_mag", "valid_mag", "train_hf", "valid_hf", "train_complex", "valid_complex", "train_protect", "valid_protect", "train_res", "valid_res", "train_gate", "valid_gate", "train_ratio", "valid_ratio", "train_si_sdr", "valid_si_sdr", "train_delta", "valid_delta", "train_scale", "valid_scale", "train_scale_order", "valid_scale_order", "train_scale_mean", "valid_scale_mean", "train_scale_min", "valid_scale_min", "train_scale_max", "valid_scale_max"])
        writer.writeheader()
        for h in history:
            writer.writerow({
                "epoch": h["epoch"],
                "train_loss": h["train"]["loss"],
                "valid_loss": h["valid"]["loss"],
                "train_mag": h["train"].get("mag", ""),
                "valid_mag": h["valid"].get("mag", ""),
                "train_hf": h["train"].get("hf", ""),
                "valid_hf": h["valid"].get("hf", ""),
                "train_complex": h["train"].get("complex", ""),
                "valid_complex": h["valid"].get("complex", ""),
                "train_protect": h["train"].get("protect", ""),
                "valid_protect": h["valid"].get("protect", ""),
                "train_res": h["train"].get("res", ""),
                "valid_res": h["valid"].get("res", ""),
                "train_gate": h["train"].get("gate", ""),
                "valid_gate": h["valid"].get("gate", ""),
                "train_ratio": h["train"].get("ratio", ""),
                "valid_ratio": h["valid"].get("ratio", ""),
                "train_si_sdr": h["train"].get("si_sdr", ""),
                "valid_si_sdr": h["valid"].get("si_sdr", ""),
                "train_delta": h["train"].get("delta", ""),
                "valid_delta": h["valid"].get("delta", ""),
                "train_scale": h["train"].get("scale", ""),
                "valid_scale": h["valid"].get("scale", ""),
                "train_scale_order": h["train"].get("scale_order", ""),
                "valid_scale_order": h["valid"].get("scale_order", ""),
                "train_scale_mean": h["train"].get("scale_mean", ""),
                "valid_scale_mean": h["valid"].get("scale_mean", ""),
                "train_scale_min": h["train"].get("scale_min", ""),
                "valid_scale_min": h["valid"].get("scale_min", ""),
                "train_scale_max": h["train"].get("scale_max", ""),
                "valid_scale_max": h["valid"].get("scale_max", ""),
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
    except BaseException as e:
        print(f"[FATAL][TOP] {type(e).__name__}: {repr(e)}", file=sys.stderr, flush=True)
        raise
