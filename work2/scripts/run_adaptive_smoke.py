"""
Smoke test for the full adaptive GTCRN pipeline.

Generates synthetic data, caches features, trains the residual module,
and evaluates final enhancement quality vs. baseline GTCRN.

Results are for pipeline verification only — NOT real speech performance.
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
import soundfile as sf

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import ensure_mono, peak_normalize, rms, calculate_snr_db
from work2.data.stft_utils import stft_to_ri, ri_to_istft
from work2.data.degradation import apply_composite_degradation, sample_degradation_config
from work2.data.label_utils import make_degradation_labels
from work2.models.gtcrn_adaptive import GTCRNAdaptive


def generate_pseudo_speech(n_samples: int, duration: float, fs: int, seed: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_smoke")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building adaptive GTCRN pipeline...")
    model = GTCRNAdaptive("checkpoints/model_trained_on_dns3.tar", "cpu", freeze_estimator=False)
    param_info = model.count_params()
    print(f"[INFO] Residual trainable params: {param_info['residual_trainable']}")

    print(f"[INFO] Generating {args.num_samples} synthetic speech samples...")
    sigs = generate_pseudo_speech(args.num_samples // 4, 2.0, 16000, args.seed)
    noises = [torch.randn(s.numel(), dtype=torch.float32) * 0.1 for s in sigs]

    severities = ["clean", "light", "medium", "heavy"]

    train_data = []
    valid_data = []

    for i, (clean, noise_raw) in enumerate(zip(sigs, noises)):
        for si, sev in enumerate(severities):
            local_seed = args.seed * 100 + i * 10 + si
            seed_everything(local_seed)
            cfg = sample_degradation_config(sev, local_seed)
            noise = noise_raw.clone() if cfg.add_noise else None

            result = apply_composite_degradation(clean.clone(), cfg, noise)

            window = torch.hann_window(512).pow(0.5)
            clean_spec = stft_to_ri(result.clean, window=window)
            degraded_spec = stft_to_ri(result.degraded, window=window)

            item = {
                "clean_spec": clean_spec,
                "degraded_spec": degraded_spec,
                "clean_audio": result.clean,
                "degraded_audio": result.degraded,
                "bandwidth_limited": result.cutoff_hz is not None or result.intermediate_rate is not None,
                "seed": local_seed,
                "severity": sev,
            }

            if i % 5 == 0:
                valid_data.append(item)
            else:
                train_data.append(item)

    print(f"[INFO] Dataset: train={len(train_data)}, valid={len(valid_data)}")

    print(f"[INFO] Training for {args.epochs} epochs...")
    from work2.losses.residual_loss import AdaptiveResidualLoss

    loss_fn = AdaptiveResidualLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.estimator.parameters(), "lr": 0.0005},
        {"params": model.residual_module.parameters(), "lr": 0.001},
    ])

    history = []
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.estimator.train()
        model.residual_module.train()

        import random as _random
        _random.Random(epoch + args.seed).shuffle(train_data)

        total_loss = 0.0
        total_mag = 0.0
        total_hf = 0.0
        total_res = 0.0
        bs = 8

        for b_start in range(0, len(train_data), bs):
            batch = train_data[b_start:b_start + bs]

            clean_specs = torch.stack([b["clean_spec"] for b in batch])
            degraded_specs = torch.stack([b["degraded_spec"] for b in batch])
            bw_limited = torch.tensor([b["bandwidth_limited"] for b in batch])

            out = model(degraded_specs)
            enhanced_final = out["enhanced_final"].permute(0, 3, 2, 1)
            residual = out["residual"].permute(0, 3, 2, 1)
            clean_permuted = clean_specs.permute(0, 3, 2, 1)

            loss, comps = loss_fn(enhanced_final, clean_permuted, residual, bw_limited)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.estimator.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.residual_module.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mag += comps["mag"]
            total_hf += comps["hf"]
            total_res += comps["res"]

        n_batches = max(1, len(train_data) // bs + (1 if len(train_data) % bs else 0))
        total_loss /= n_batches
        total_mag /= n_batches
        total_hf /= n_batches
        total_res /= n_batches

        model.eval()
        model.estimator.eval()
        model.residual_module.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for b_start in range(0, len(valid_data), bs):
                batch = valid_data[b_start:b_start + bs]
                clean_specs = torch.stack([b["clean_spec"] for b in batch])
                degraded_specs = torch.stack([b["degraded_spec"] for b in batch])
                bw_limited = torch.tensor([b["bandwidth_limited"] for b in batch])
                out = model(degraded_specs)
                enhanced_final_val = out["enhanced_final"].permute(0, 3, 2, 1)
                residual_val = out["residual"].permute(0, 3, 2, 1)
                clean_permuted_val = clean_specs.permute(0, 3, 2, 1)
                loss, _ = loss_fn(enhanced_final_val, clean_permuted_val, residual_val, bw_limited)
                valid_loss += loss.item()
        valid_loss /= max(1, len(valid_data) // bs + 1)

        history.append({"epoch": epoch, "train_loss": total_loss, "valid_loss": valid_loss})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_loss={total_loss:.4f}  valid_loss={valid_loss:.4f}")

    t_train = time.perf_counter() - t_start
    print(f"[INFO] Training complete in {t_train:.1f}s")

    # Evaluation: compare GTCRN base vs adaptive on test samples
    print("[INFO] Evaluating base GTCRN vs adaptive...")
    results = []

    for i, item in enumerate(valid_data):
        clean_spec = item["clean_spec"].unsqueeze(0)
        degraded_spec = item["degraded_spec"].unsqueeze(0)

        with torch.no_grad():
            out = model(degraded_spec)

        enhanced_base = out["enhanced_base"]
        enhanced_final = out["enhanced_final"]

        base_mag = torch.sqrt(enhanced_base[..., 0]**2 + enhanced_base[..., 1]**2 + 1e-12)
        final_mag = torch.sqrt(enhanced_final[..., 0]**2 + enhanced_final[..., 1]**2 + 1e-12)
        clean_mag = torch.sqrt(clean_spec[..., 0]**2 + clean_spec[..., 1]**2 + 1e-12)

        base_l1 = (base_mag - clean_mag).abs().mean().item()
        final_l1 = (final_mag - clean_mag).abs().mean().item()

        results.append({
            "severity": item["severity"],
            "base_l1": base_l1,
            "adaptive_l1": final_l1,
            "improvement": base_l1 - final_l1,
        })

    # Summary
    sev_summary = {}
    for r in results:
        sev = r["severity"]
        if sev not in sev_summary:
            sev_summary[sev] = {"base": [], "adaptive": []}
        sev_summary[sev]["base"].append(r["base_l1"])
        sev_summary[sev]["adaptive"].append(r["adaptive_l1"])

    print("\n=== Adaptive Residual Comparison (synthetic only) ===")
    print(f"{'Severity':<10} {'Base L1':>10} {'Adaptive L1':>12} {'Delta':>10}")
    print("-" * 44)
    for sev in severities:
        if sev in sev_summary:
            b = np.mean(sev_summary[sev]["base"])
            a = np.mean(sev_summary[sev]["adaptive"])
            d = b - a
            print(f"{sev:<10} {b:>10.4f} {a:>12.4f} {d:>+10.4f}")

    # Save results
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump({
            "note": "SYNTHETIC DATA ONLY — NOT real speech performance",
            "results": results,
            "severity_summary": {s: {"base_mean": float(np.mean(v["base"])), "adaptive_mean": float(np.mean(v["adaptive"]))}
                                for s, v in sev_summary.items()},
        }, f, indent=2)

    # Training curves
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = [h["epoch"] for h in history]
    t_loss = [h["train_loss"] for h in history]
    v_loss = [h["valid_loss"] for h in history]
    ax.plot(epochs, t_loss, label="Train")
    ax.plot(epochs, v_loss, label="Valid")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Adaptive Residual Training")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=150); plt.close(fig)

    # Improvement bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = [s for s in severities if s in sev_summary]
    improvements = [float(np.mean(sev_summary[s]["base"]) - np.mean(sev_summary[s]["adaptive"])) for s in x_labels]
    colors = ["green" if v > 0 else "red" for v in improvements]
    ax.bar(x_labels, improvements, color=colors)
    ax.set_ylabel("L1 Improvement (positive = better)"); ax.set_title("Adaptive vs Base GTCRN")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout(); fig.savefig(fig_dir / "improvement_bar.png", dpi=150); plt.close(fig)

    # Single sample waveform comparison
    sample = valid_data[0]
    with torch.no_grad():
        out = model(sample["degraded_spec"].unsqueeze(0))
    window = torch.hann_window(512).pow(0.5)
    base_audio = ri_to_istft(out["enhanced_base"][0], window=window)
    final_audio = ri_to_istft(out["enhanced_final"][0], window=window)
    clean_audio = sample["clean_audio"]
    degraded_audio = sample["degraded_audio"]

    sf.write(str(out_dir / "sample_clean.wav"), clean_audio.numpy(), 16000)
    sf.write(str(out_dir / "sample_degraded.wav"), degraded_audio.numpy(), 16000)
    sf.write(str(out_dir / "sample_gtcrn_base.wav"), base_audio.numpy(), 16000)
    sf.write(str(out_dir / "sample_adaptive.wav"), final_audio.numpy(), 16000)

    # Waveform plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, data, title in zip(axes,
                                [clean_audio, degraded_audio, base_audio, final_audio],
                                ["Clean", "Degraded", "GTCRN Base", "GTCRN Adaptive"]):
        ax.plot(data[:2000].numpy(), linewidth=0.5, color="steelblue")
        ax.set_title(title); ax.set_ylabel("Amp")
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("Samples")
    fig.tight_layout(); fig.savefig(fig_dir / "waveform_comparison.png", dpi=150); plt.close(fig)

    print(f"\n[DONE] All outputs in {out_dir}")
    print("[WARN] Results are from SYNTHETIC data — do NOT cite as real speech performance.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
