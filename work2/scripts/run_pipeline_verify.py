"""
End-to-end pipeline verification on VCTK real speech samples.

Tests the complete chain:
  clean VCTK → composite degradation → STFT → Frozen GTCRN
  → Degradation Estimator → Adaptive Residual → compare base vs adaptive
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import ensure_mono, peak_normalize
from work2.data.stft_utils import stft_to_ri, ri_to_istft
from work2.data.degradation import apply_composite_degradation, sample_degradation_config
from work2.data.label_utils import make_degradation_labels
from work2.models.gtcrn_adaptive import GTCRNAdaptive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=str, default="data/vctk_16k")
    parser.add_argument("--output-dir", type=str, default="outputs/pipeline_verify")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--estimator-checkpoint", type=str, default=None,
                        help="Trained estimator checkpoint (optional)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = Path(args.clean_dir)
    wav_files = sorted(clean_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files in {clean_dir}")

    print(f"[INFO] Found {len(wav_files)} VCTK files")

    print("[INFO] Building adaptive GTCRN pipeline...")
    model = GTCRNAdaptive("checkpoints/model_trained_on_dns3.tar", "cpu")
    param_info = model.count_params()
    print(f"[INFO] Params: GTCRN={param_info['extractor_total']} (frozen), "
          f"Estimator={param_info['estimator_total']}, Residual={param_info['residual_total']}")

    if args.estimator_checkpoint and Path(args.estimator_checkpoint).exists():
        ckpt = torch.load(args.estimator_checkpoint, map_location="cpu")
        model.estimator.load_state_dict(ckpt["model_state_dict"])
        model.freeze_estimator = True
        model._apply_freezing()
        print(f"[INFO] Loaded trained estimator from {args.estimator_checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[WARN] No estimator checkpoint — using untrained weights")

    severities = ["clean", "light", "medium", "heavy"]
    results = []

    for fi, wav_path in enumerate(wav_files):
        clean_audio, fs = sf.read(str(wav_path), dtype="float32")
        clean_audio = ensure_mono(torch.from_numpy(clean_audio))

        if fs != 16000:
            raise RuntimeError(f"Sample rate {fs} != 16000 for {wav_path.name}")

        duration = len(clean_audio) / fs
        print(f"\n[INFO] File {fi+1}/{len(wav_files)}: {wav_path.name} ({duration:.1f}s)")

        for sev in severities:
            local_seed = args.seed * 100 + fi * 10 + severities.index(sev)
            seed_everything(local_seed)

            cfg = sample_degradation_config(sev, local_seed)
            noise = torch.randn(len(clean_audio), dtype=torch.float32) * 0.1
            noise = noise - noise.mean()

            result = apply_composite_degradation(clean_audio.clone(), cfg, noise.clone())

            window = torch.hann_window(512).pow(0.5)
            degraded_spec = stft_to_ri(result.degraded, window=window)
            clean_spec = stft_to_ri(result.clean, window=window)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(degraded_spec.unsqueeze(0))
            t_infer = time.perf_counter() - t0

            labels = make_degradation_labels(result)

            degradation_preds = out["degradation_preds"]
            noise_prob = float(torch.sigmoid(degradation_preds["noise_logit"]).item())
            snr_pred_norm = float(degradation_preds["snr_pred"].item())
            bw_pred = int(degradation_preds["bandwidth_logits"].argmax().item())
            bit_pred = int(degradation_preds["bit_logits"].argmax().item())

            from work2.data.label_utils import denormalize_snr
            snr_pred_db = float(denormalize_snr(torch.tensor([snr_pred_norm])).item())

            enhanced_base_mag = torch.sqrt(out["enhanced_base"][0, ..., 0]**2 + out["enhanced_base"][0, ..., 1]**2 + 1e-12)
            enhanced_final_mag = torch.sqrt(out["enhanced_final"][0, ..., 0]**2 + out["enhanced_final"][0, ..., 1]**2 + 1e-12)
            clean_mag = torch.sqrt(clean_spec[..., 0]**2 + clean_spec[..., 1]**2 + 1e-12)

            base_l1 = float((enhanced_base_mag - clean_mag).abs().mean().item())
            adaptive_l1 = float((enhanced_final_mag - clean_mag).abs().mean().item())

            residual_power = float((out["residual"] ** 2).mean().item())

            r = {
                "file": wav_path.name,
                "severity": sev,
                "true_noise": int(labels["noise_present"]),
                "pred_noise_prob": round(noise_prob, 4),
                "true_bw_class": labels["bandwidth_class"],
                "pred_bw_class": bw_pred,
                "true_bit_class": labels["bit_class"],
                "pred_bit_class": bit_pred,
                "true_snr_db": labels["snr_target"],
                "pred_snr_db": round(snr_pred_db, 2),
                "base_l1": round(base_l1, 6),
                "adaptive_l1": round(adaptive_l1, 6),
                "delta_l1": round(base_l1 - adaptive_l1, 6),
                "residual_power": round(residual_power, 8),
                "inference_time_s": round(t_infer, 4),
            }
            results.append(r)

    # Summary by severity
    sev_summary = {}
    for s in severities:
        items = [r for r in results if r["severity"] == s]
        if not items:
            continue
        base_mean = np.mean([r["base_l1"] for r in items])
        adaptive_mean = np.mean([r["adaptive_l1"] for r in items])
        noise_acc = np.mean([r["true_noise"] == int(r["pred_noise_prob"] > 0.5) for r in items])
        bw_acc = np.mean([r["true_bw_class"] == r["pred_bw_class"] for r in items])
        bit_acc = np.mean([r["true_bit_class"] == r["pred_bit_class"] for r in items])
        sev_summary[s] = {
            "n": len(items),
            "base_l1_mean": round(float(base_mean), 6),
            "adaptive_l1_mean": round(float(adaptive_mean), 6),
            "delta_l1": round(float(base_mean - adaptive_mean), 6),
            "noise_accuracy": round(float(noise_acc), 4),
            "bandwidth_accuracy": round(float(bw_acc), 4),
            "bit_accuracy": round(float(bit_acc), 4),
        }

    print("\n" + "=" * 75)
    print("PIPELINE VERIFICATION RESULTS (5 VCTK samples, estimator untrained)")
    print("=" * 75)
    print(f"{'Severity':<10} {'N':>3} {'Base L1':>10} {'Adaptive L1':>12} {'Delta':>10} {'NoiseAcc':>10} {'BW Acc':>10} {'Bit Acc':>10}")
    print("-" * 75)
    for s in severities:
        if s in sev_summary:
            m = sev_summary[s]
            print(f"{s:<10} {m['n']:>3} {m['base_l1_mean']:>10.4f} {m['adaptive_l1_mean']:>12.4f} "
                  f"{m['delta_l1']:>+10.4f} {m['noise_accuracy']:>10.2%} {m['bandwidth_accuracy']:>10.2%} {m['bit_accuracy']:>10.2%}")

    t_avg = np.mean([r["inference_time_s"] for r in results])
    print(f"\n[INFO] Avg inference time: {t_avg*1000:.1f} ms (STFT+GTCRN+Estimator+Residual)")

    with open(out_dir / "pipeline_results.json", "w") as f:
        json.dump({"results": results, "severity_summary": sev_summary}, f, indent=2)

    print(f"\n[DONE] Results saved to {out_dir / 'pipeline_results.json'}")
    print("[NOTE] Estimator and residual are UNTRAINED — accuracies expected to be random.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
