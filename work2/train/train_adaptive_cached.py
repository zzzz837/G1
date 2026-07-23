"""
Cached V1 / V2-Refined-B training.

Important properties:
- uses only cached tensors; no GTCRN or estimator forward
- uses oracle conditions from cached labels
- shuffles offsets inside each training shard
- averages every loss/statistic by processed sample count
- uses a deterministic shard subset when --max-*-shards is set
- logs low/mid/high gates, global gate, residual ratio, and severity-wise
  global gate/residual ratio
- saves full resume state (model, optimizer, scheduler, patience, history)
- reloads best_model.pt before final test
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

from work2.data.random_utils import seed_everything
from work2.losses.residual_loss import AdaptiveResidualLoss
from work2.models.adaptive_residual import (
    AdaptiveResidualModule,
    AdaptiveResidualModuleV2Formal,
)


COMPONENT_KEYS = ("mag", "hf", "complex", "protect", "res", "gate", "ratio")
GATE_KEYS = (
    "alpha_low",
    "alpha_mid",
    "alpha_high",
    "global_alpha",
    "residual_ratio",
)
SEVERITY_NAMES = ("clean", "light", "medium", "heavy")
SEVERITY_GATE_KEYS = tuple(
    f"{metric}_{severity}"
    for metric in ("global_alpha", "residual_ratio")
    for severity in SEVERITY_NAMES
)


def load_index(index_path: Path, split: str) -> dict[str, list[int]]:
    shard_to_entries: dict[str, list[int]] = {}
    with open(index_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("split") != split:
                continue
            shard_to_entries.setdefault(entry["shard_file"], []).append(
                int(entry["offset"])
            )
    if not shard_to_entries:
        raise ValueError(f"No entries for split={split} in {index_path}")
    return shard_to_entries


def split_minibatches(n_items: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def select_shards(
    shard_map: dict[str, list[int]],
    max_shards: int | None,
    train: bool,
) -> list[str]:
    """
    Select a deterministic subset, then optionally shuffle its order.

    This is important for smoke/medium experiments: slicing after random.shuffle
    would train on a different subset every epoch and make comparisons invalid.
    """
    shard_files = sorted(shard_map.keys())
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive when provided.")
        shard_files = shard_files[:max_shards]
    if train:
        random.shuffle(shard_files)
    return shard_files


def build_oracle_condition(
    noise_target: torch.Tensor,
    snr_target: torch.Tensor,
    bandwidth_target: torch.Tensor,
    bit_target: torch.Tensor,
) -> torch.Tensor:
    """Build [noise(1), SNR(1), bandwidth one-hot(3), bit one-hot(4)]."""
    batch_size = noise_target.shape[0]
    noise_target = noise_target.view(batch_size, 1).to(torch.float32)
    snr_target = snr_target.view(batch_size, 1).to(torch.float32)
    bandwidth_target = bandwidth_target.view(batch_size).to(torch.long)
    bit_target = bit_target.view(batch_size).to(torch.long)

    if bandwidth_target.min().item() < 0 or bandwidth_target.max().item() > 2:
        raise ValueError("bandwidth_target must be in {0,1,2}.")
    if bit_target.min().item() < 0 or bit_target.max().item() > 3:
        raise ValueError("bit_target must be in {0,1,2,3}.")

    bw_oh = torch.zeros(
        batch_size, 3, dtype=torch.float32, device=noise_target.device
    )
    bw_oh.scatter_(1, bandwidth_target.view(-1, 1), 1.0)
    bit_oh = torch.zeros(
        batch_size, 4, dtype=torch.float32, device=noise_target.device
    )
    bit_oh.scatter_(1, bit_target.view(-1, 1), 1.0)
    snr_norm = torch.clamp(snr_target / 40.0, 0.0, 1.0)
    return torch.cat([noise_target, snr_norm, bw_oh, bit_oh], dim=1)


def severity_to_indices(values) -> torch.Tensor:
    mapping = {name: idx for idx, name in enumerate(SEVERITY_NAMES)}
    if isinstance(values, torch.Tensor):
        if values.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            result = values.to(torch.long).view(-1)
        else:
            values = values.tolist()
            result = torch.tensor(
                [mapping[str(value)] for value in values], dtype=torch.long
            )
    else:
        result = torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    if result.numel() and (result.min().item() < 0 or result.max().item() > 3):
        raise ValueError("Severity indices must be in {0,1,2,3}.")
    return result


def gather_gate_stats(
    output: dict[str, torch.Tensor],
    severity: torch.Tensor,
) -> tuple[dict[str, float], dict[str, tuple[float, int]]]:
    stats = {key: 0.0 for key in GATE_KEYS}

    alpha = output.get("alpha")
    if alpha is not None:
        alpha_det = alpha.detach()
        if alpha_det.ndim >= 2 and alpha_det.shape[1] >= 3:
            stats["alpha_low"] = float(alpha_det[:, 0].mean().item())
            stats["alpha_mid"] = float(alpha_det[:, 1].mean().item())
            stats["alpha_high"] = float(alpha_det[:, 2].mean().item())
        else:
            mean_alpha = float(alpha_det.mean().item())
            stats["alpha_low"] = mean_alpha
            stats["alpha_mid"] = mean_alpha
            stats["alpha_high"] = mean_alpha

    global_alpha = output.get("global_alpha")
    if global_alpha is None:
        global_values = torch.ones_like(severity, dtype=torch.float32)
        stats["global_alpha"] = 1.0
    else:
        global_values = global_alpha.detach().view(-1)
        stats["global_alpha"] = float(global_values.mean().item())

    residual_ratio = output.get("residual_ratio")
    if residual_ratio is None:
        residual = output["residual"].detach().flatten(1)
        base = (
            output["enhanced_final"] - output["residual"]
        ).detach().flatten(1)
        ratio_values = torch.linalg.vector_norm(residual, dim=1) / (
            torch.linalg.vector_norm(base, dim=1) + 1e-8
        )
    else:
        ratio_values = residual_ratio.detach().view(-1)
    stats["residual_ratio"] = float(ratio_values.mean().item())

    by_severity: dict[str, tuple[float, int]] = {}
    severity = severity.detach().view(-1)
    for idx, name in enumerate(SEVERITY_NAMES):
        mask = severity == idx
        count = int(mask.sum().item())
        if count:
            by_severity[f"global_alpha_{name}"] = (
                float(global_values[mask].sum().item()),
                count,
            )
            by_severity[f"residual_ratio_{name}"] = (
                float(ratio_values[mask].sum().item()),
                count,
            )
    return stats, by_severity


def run_split_epoch(
    residual_module,
    shard_map,
    cache_dir: Path,
    loss_fn,
    optimizer,
    device,
    batch_size: int,
    train: bool,
    log_interval: int,
    max_shards=None,
    phase_name="train",
):
    residual_module.train(train)
    shard_files = select_shards(shard_map, max_shards, train=train)

    total_loss = 0.0
    comp_sum = {key: 0.0 for key in COMPONENT_KEYS}
    gate_sum = {key: 0.0 for key in GATE_KEYS}
    severity_sum = {key: 0.0 for key in SEVERITY_GATE_KEYS}
    severity_count = {key: 0 for key in SEVERITY_GATE_KEYS}
    processed_batches = 0
    processed_samples = 0

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for shard_idx, shard_file in enumerate(shard_files, start=1):
            shard_path = cache_dir / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)

            offsets = list(shard_map[shard_file])
            if train:
                random.shuffle(offsets)
            else:
                offsets.sort()

            required = (
                "clean_spec",
                "enhanced_base",
                "noise_target",
                "snr_target",
                "bandwidth_target",
                "bit_target",
                "bandwidth_limited",
                "severity",
            )
            missing = [key for key in required if key not in shard]
            if missing:
                raise KeyError(f"{shard_file} is missing keys: {missing}")

            clean_spec = shard["clean_spec"][offsets]
            enhanced_base = shard["enhanced_base"][offsets]
            noise_target = shard["noise_target"][offsets].to(torch.float32)
            snr_target = shard["snr_target"][offsets].to(torch.float32)
            bandwidth_target = shard["bandwidth_target"][offsets].to(torch.long)
            bit_target = shard["bit_target"][offsets].to(torch.long)
            bandwidth_limited = shard["bandwidth_limited"][offsets].to(torch.bool)
            severity_all = severity_to_indices(
                [shard["severity"][offset] for offset in offsets]
            )

            n_items = clean_spec.shape[0]
            if enhanced_base.shape[0] != n_items:
                raise ValueError(f"Batch-size mismatch in {shard_file}.")
            if n_items == 0:
                continue

            for start, end in split_minibatches(n_items, batch_size):
                current_bs = end - start
                try:
                    clean_mb = clean_spec[start:end].to(
                        device=device, dtype=torch.float32
                    )
                    base_mb = enhanced_base[start:end].to(
                        device=device, dtype=torch.float32
                    )
                    noise_mb = noise_target[start:end].to(device)
                    snr_mb = snr_target[start:end].to(device)
                    bw_mb = bandwidth_target[start:end].to(device)
                    bit_mb = bit_target[start:end].to(device)
                    bw_limited_mb = bandwidth_limited[start:end].to(device)
                    severity_mb = severity_all[start:end].to(device)

                    if clean_mb.ndim != 4 or clean_mb.shape[-1] != 2:
                        raise ValueError(
                            f"clean_spec must be (B,F,T,2), got {tuple(clean_mb.shape)}"
                        )
                    if base_mb.shape != clean_mb.shape:
                        raise ValueError(
                            f"enhanced_base shape {tuple(base_mb.shape)} does not match "
                            f"clean_spec {tuple(clean_mb.shape)}"
                        )

                    cond_mb = build_oracle_condition(
                        noise_mb, snr_mb, bw_mb, bit_mb
                    )
                    clean_perm = clean_mb.permute(0, 3, 2, 1).contiguous()
                    base_perm = base_mb.permute(0, 3, 2, 1).contiguous()

                    output = residual_module(base_perm, cond_mb)
                    loss, components = loss_fn(
                        enhanced_final=output["enhanced_final"],
                        clean_spec=clean_perm,
                        residual=output["residual"],
                        bandwidth_limited=bw_limited_mb,
                        enhanced_base=base_perm,
                        severity=severity_mb,
                        global_alpha=output.get("global_alpha"),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {loss.item()}")

                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            residual_module.parameters(), 1.0
                        )
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError(
                                f"Non-finite gradient norm: {grad_norm.item()}"
                            )
                        optimizer.step()

                    gate_stats, by_severity = gather_gate_stats(
                        output, severity_mb
                    )
                    total_loss += float(loss.item()) * current_bs
                    for key in COMPONENT_KEYS:
                        comp_sum[key] += components[key] * current_bs
                    for key in GATE_KEYS:
                        gate_sum[key] += gate_stats[key] * current_bs
                    for key, (value_sum, count) in by_severity.items():
                        severity_sum[key] += value_sum
                        severity_count[key] += count

                    processed_batches += 1
                    processed_samples += current_bs

                    if processed_batches == 1 or processed_batches % log_interval == 0:
                        print(
                            f"[{phase_name}] shard={shard_idx}/{len(shard_files)} "
                            f"batch={processed_batches} samples={processed_samples} "
                            f"loss={loss.item():.4f} mag={components['mag']:.4f} "
                            f"hf={components['hf']:.4f} "
                            f"complex={components['complex']:.4f} "
                            f"protect={components['protect']:.4f} "
                            f"res={components['res']:.6f} "
                            f"gate={components['gate']:.5f} "
                            f"ratio={components['ratio']:.5f} "
                            f"a=({gate_stats['alpha_low']:.4f},"
                            f"{gate_stats['alpha_mid']:.4f},"
                            f"{gate_stats['alpha_high']:.4f}) "
                            f"g={gate_stats['global_alpha']:.4f} "
                            f"rho={gate_stats['residual_ratio']:.6f}",
                            flush=True,
                        )
                except Exception as batch_error:
                    print(
                        f"[FATAL] {phase_name} failed in {shard_file}, "
                        f"batch {start}:{end}: {batch_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

    if processed_samples == 0:
        raise RuntimeError(f"No sample processed in split {phase_name}")

    result = {
        "loss": total_loss / processed_samples,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
    }
    result.update(
        {key: comp_sum[key] / processed_samples for key in COMPONENT_KEYS}
    )
    result.update(
        {key: gate_sum[key] / processed_samples for key in GATE_KEYS}
    )
    result.update(
        {
            key: (
                severity_sum[key] / severity_count[key]
                if severity_count[key] > 0
                else float("nan")
            )
            for key in SEVERITY_GATE_KEYS
        }
    )
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(
        description="Cached training for V1 and V2-Refined-B"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="outputs/adaptive_cache_sharded"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/adaptive_residual_v2_refined_b",
    )
    parser.add_argument(
        "--residual-version", choices=["v1", "v2"], default="v2"
    )
    parser.add_argument(
        "--condition-source", choices=["oracle"], default="oracle"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--max-valid-shards", type=int, default=None)
    parser.add_argument("--max-test-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--alpha-max", type=float, default=0.2)
    parser.add_argument("--candidate-scale", type=float, default=1.0)
    parser.add_argument("--global-gate-init", type=float, default=-2.0)
    parser.add_argument("--lambda-mag", type=float, default=1.0)
    parser.add_argument("--lambda-hf", type=float, default=0.5)
    parser.add_argument("--lambda-complex", type=float, default=0.1)
    parser.add_argument("--lambda-protect", type=float, default=0.1)
    parser.add_argument("--lambda-res", type=float, default=0.05)
    parser.add_argument("--lambda-gate", type=float, default=0.02)
    parser.add_argument("--lambda-ratio", type=float, default=0.2)
    parser.add_argument("--hf-cutoff-ratio", type=float, default=0.55)
    parser.add_argument("--clean-protect-weight", type=float, default=1.0)
    parser.add_argument("--light-protect-weight", type=float, default=0.5)
    parser.add_argument("--medium-protect-weight", type=float, default=0.1)
    parser.add_argument("--heavy-protect-weight", type=float, default=0.0)
    args = parser.parse_args()

    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(args.seed)
    device = resolve_device(args.device)

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    train_map = load_index(index_path, "train")
    valid_map = load_index(index_path, "valid")
    test_map = load_index(index_path, "test")

    def count_samples(split_map):
        return sum(len(values) for values in split_map.values())

    if args.residual_version == "v1":
        residual_module = AdaptiveResidualModule(
            n_freqs=257, cond_dim=9, hidden_dim=32
        ).to(device)
    else:
        residual_module = AdaptiveResidualModuleV2Formal(
            n_freqs=257,
            cond_dim=9,
            hidden_dim=32,
            alpha_max=args.alpha_max,
            candidate_scale=args.candidate_scale,
            global_gate_init=args.global_gate_init,
        ).to(device)

    loss_fn = AdaptiveResidualLoss(
        lambda_mag=args.lambda_mag,
        lambda_hf=args.lambda_hf,
        lambda_res=args.lambda_res,
        lambda_complex=args.lambda_complex,
        lambda_protect=args.lambda_protect,
        lambda_gate=args.lambda_gate,
        lambda_ratio=args.lambda_ratio,
        hf_cutoff_ratio=args.hf_cutoff_ratio,
        clean_protect_weight=args.clean_protect_weight,
        light_protect_weight=args.light_protect_weight,
        medium_protect_weight=args.medium_protect_weight,
        heavy_protect_weight=args.heavy_protect_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        residual_module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 2),
        min_lr=1e-6,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out_dir / "last_model.pt").exists():
        raise FileExistsError(
            f"{out_dir} already contains last_model.pt. Use a new output directory "
            "or pass --resume intentionally."
        )

    start_epoch = 1
    best_valid = float("inf")
    history: list[dict] = []
    patience_counter = 0

    if args.resume:
        last_path = out_dir / "last_model.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume: {last_path} does not exist.")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("model_variant") not in (None, "v2_refined_b"):
            raise ValueError(
                f"Unexpected model_variant in resume checkpoint: "
                f"{checkpoint.get('model_variant')}"
            )
        residual_module.load_state_dict(checkpoint["residual_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid = float(checkpoint.get("best_valid", best_valid))
        history = checkpoint.get("history", [])
        patience_counter = int(checkpoint.get("patience_counter", 0))
        print(f"[INFO] Resumed from epoch {start_epoch}")

    n_params = residual_module.count_trainable_params()
    config = {
        **vars(args),
        "resolved_device": str(device),
        "train_samples": count_samples(train_map),
        "valid_samples": count_samples(valid_map),
        "test_samples": count_samples(test_map),
        "train_shards": len(train_map),
        "valid_shards": len(valid_map),
        "test_shards": len(test_map),
        "selected_train_shards": len(
            select_shards(train_map, args.max_train_shards, train=False)
        ),
        "selected_valid_shards": len(
            select_shards(valid_map, args.max_valid_shards, train=False)
        ),
        "selected_test_shards": len(
            select_shards(test_map, args.max_test_shards, train=False)
        ),
        "residual_params": n_params,
        "cache_mode": "adaptive_cached_oracle_v2_refined_b",
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print(f"[INFO] Residual version: {args.residual_version}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Trainable parameters: {n_params}")
    print(
        f"[INFO] train/valid/test samples: "
        f"{config['train_samples']}/{config['valid_samples']}/{config['test_samples']}"
    )

    start_time = time.perf_counter()
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_metrics = run_split_epoch(
            residual_module,
            train_map,
            cache_dir,
            loss_fn,
            optimizer,
            device,
            args.mini_batch_size,
            train=True,
            log_interval=args.log_interval,
            max_shards=args.max_train_shards,
            phase_name="train",
        )
        valid_metrics = run_split_epoch(
            residual_module,
            valid_map,
            cache_dir,
            loss_fn,
            optimizer,
            device,
            args.mini_batch_size,
            train=False,
            log_interval=max(args.log_interval, 1000),
            max_shards=args.max_valid_shards,
            phase_name="valid",
        )

        scheduler.step(valid_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_metrics['loss']:.4f} | "
            f"valid={valid_metrics['loss']:.4f} | "
            f"mag={valid_metrics['mag']:.4f} | "
            f"hf={valid_metrics['hf']:.4f} | "
            f"complex={valid_metrics['complex']:.4f} | "
            f"protect={valid_metrics['protect']:.4f} | "
            f"res={valid_metrics['res']:.6f} | "
            f"gate={valid_metrics['gate']:.5f} | "
            f"ratio={valid_metrics['ratio']:.5f} | "
            f"a=({valid_metrics['alpha_low']:.4f},"
            f"{valid_metrics['alpha_mid']:.4f},"
            f"{valid_metrics['alpha_high']:.4f}) | "
            f"g={valid_metrics['global_alpha']:.4f} | "
            f"rho={valid_metrics['residual_ratio']:.6f} | "
            f"g_clean={valid_metrics['global_alpha_clean']:.4f} | "
            f"g_heavy={valid_metrics['global_alpha_heavy']:.4f}",
            flush=True,
        )

        improved = valid_metrics["loss"] < best_valid - args.min_delta
        if improved:
            best_valid = valid_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "residual_state_dict": residual_module.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "validation_metrics": valid_metrics,
                    "residual_version": args.residual_version,
                    "model_variant": "v2_refined_b",
                    "alpha_max": args.alpha_max,
                    "loss_config": {
                        "lambda_mag": args.lambda_mag,
                        "lambda_hf": args.lambda_hf,
                        "lambda_complex": args.lambda_complex,
                        "lambda_protect": args.lambda_protect,
                        "lambda_res": args.lambda_res,
                        "lambda_gate": args.lambda_gate,
                        "lambda_ratio": args.lambda_ratio,
                        "hf_cutoff_ratio": args.hf_cutoff_ratio,
                        "clean_protect_weight": args.clean_protect_weight,
                        "light_protect_weight": args.light_protect_weight,
                        "medium_protect_weight": args.medium_protect_weight,
                        "heavy_protect_weight": args.heavy_protect_weight,
                    },
                },
                out_dir / "best_model.pt",
            )
            print("[INFO] Saved best_model.pt")
        else:
            patience_counter += 1

        torch.save(
            {
                "residual_state_dict": residual_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "residual_version": args.residual_version,
                "model_variant": "v2_refined_b",
                "best_valid": best_valid,
                "patience_counter": patience_counter,
                "history": history,
            },
            out_dir / "last_model.pt",
        )

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    training_time = time.perf_counter() - start_time
    best_path = out_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint was saved at {best_path}")

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    residual_module.load_state_dict(best_checkpoint["residual_state_dict"])
    residual_module.eval()
    print(f"[INFO] Testing best checkpoint from epoch {best_checkpoint.get('epoch')}")

    test_metrics = run_split_epoch(
        residual_module,
        test_map,
        cache_dir,
        loss_fn,
        optimizer,
        device,
        args.mini_batch_size,
        train=False,
        log_interval=10**9,
        max_shards=args.max_test_shards,
        phase_name="test",
    )

    metric_keys = ("loss", *COMPONENT_KEYS, *GATE_KEYS, *SEVERITY_GATE_KEYS)
    fieldnames = ["epoch", "lr"] + [
        f"{phase}_{key}" for phase in ("train", "valid") for key in metric_keys
    ]
    with open(
        out_dir / "history.csv", "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            row = {"epoch": item["epoch"], "lr": item["lr"]}
            for phase in ("train", "valid"):
                for key in metric_keys:
                    row[f"{phase}_{key}"] = item[phase][key]
            writer.writerow(row)

    summary = {
        "epochs_completed": last_epoch,
        "best_epoch": best_checkpoint.get("epoch"),
        "total_training_time_s": round(training_time, 2),
        "best_valid_loss": best_valid,
        "test_metrics": test_metrics,
        "residual_params": n_params,
        "residual_version": args.residual_version,
        "model_variant": "v2_refined_b",
        "early_stopped": patience_counter >= args.patience,
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(
        f"[INFO] Test best checkpoint: loss={test_metrics['loss']:.4f}, "
        f"mag={test_metrics['mag']:.4f}, hf={test_metrics['hf']:.4f}, "
        f"complex={test_metrics['complex']:.4f}, "
        f"protect={test_metrics['protect']:.4f}, "
        f"res={test_metrics['res']:.6f}, "
        f"gate={test_metrics['gate']:.5f}, "
        f"ratio={test_metrics['ratio']:.5f}"
    )
    print(f"[DONE] Outputs saved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        raise
