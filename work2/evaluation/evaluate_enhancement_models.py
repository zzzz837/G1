"""
Unified enhancement evaluation for Degraded / Base / EDCR-V1 / EDCR-V2.

Uses cached adaptive shards and oracle conditions from labels.
Outputs paper-ready tables and curves.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pystoi import stoi

from work2.data.stft_utils import ri_to_istft
from work2.models.adaptive_residual import AdaptiveResidualModule
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator

try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except Exception:
    HAS_PESQ = False

SEVERITIES = ["clean", "light", "medium", "heavy"]
SAMPLE_RATE = 16000
N_FFT = 512
HF_START_HZ = 4000
HF_START_BIN = round(HF_START_HZ / (SAMPLE_RATE / N_FFT))


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
    return shard_to_entries


def build_oracle_condition(noise_target, snr_target, bandwidth_target, bit_target):
    B = noise_target.shape[0]
    bw_oh = torch.zeros(B, 3, dtype=torch.float32, device=noise_target.device)
    bw_oh.scatter_(1, bandwidth_target.view(-1, 1), 1.0)
    bit_oh = torch.zeros(B, 4, dtype=torch.float32, device=noise_target.device)
    bit_oh.scatter_(1, bit_target.view(-1, 1), 1.0)
    snr_norm = torch.clamp(snr_target / 40.0, 0.0, 1.0)
    return torch.cat([noise_target, snr_norm, bw_oh, bit_oh], dim=1)


def si_sdr(ref, est, eps=1e-8):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + eps
    proj = np.dot(est, ref) / ref_energy
    s_target = proj * ref
    e_noise = est - s_target
    num = np.sum(s_target ** 2) + eps
    den = np.sum(e_noise ** 2) + eps
    return float(10 * np.log10(num / den))


def lsd(clean_spec, test_spec, hf_only=False):
    clean_mag = np.sqrt(clean_spec[...,0]**2 + clean_spec[...,1]**2 + 1e-12)
    test_mag = np.sqrt(test_spec[...,0]**2 + test_spec[...,1]**2 + 1e-12)
    if hf_only:
        start = HF_START_BIN
        clean_mag = clean_mag[start:]
        test_mag = test_mag[start:]
    val = np.sqrt(np.mean((20*np.log10(clean_mag+1e-12)-20*np.log10(test_mag+1e-12))**2, axis=0)).mean()
    return float(val)


def mean_dict(list_dicts, keys):
    return {k: float(np.mean([d[k] for d in list_dicts])) for k in keys}


def parse_scale_list(scale_text: str):
    values = []
    for part in scale_text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("At least one residual scale must be provided")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--v1-checkpoint", type=str, default="outputs/adaptive_residual_cached_v1_oracle_fixed/best_model.pt")
    parser.add_argument("--v2-checkpoint", type=str, default="outputs/adaptive_residual_cached_v2_oracle_fixed/best_model.pt")
    parser.add_argument("--v2-architecture", type=str, default="initial", choices=["initial", "refined"])
    parser.add_argument("--output-dir", type=str, default="outputs/paper_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--fast-metrics-only", action="store_true", help="Only compute SI-SDR, LSD, HF-LSD")
    parser.add_argument("--disable-pesq", action="store_true", help="Skip PESQ even if the package is installed")
    parser.add_argument("--condition-mode", type=str, default="predicted", choices=["predicted", "oracle"], help="Condition source for V1 evaluation")
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--residual-scale", type=float, default=0.75, help="Single residual scale (legacy option)")
    parser.add_argument("--residual-scales", type=str, default=None, help="Comma-separated residual scales, e.g. 0.5,0.75,0.9,1.0")
    parser.add_argument("--progress-interval", type=int, default=5, help="Print progress every N shards")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    scales = parse_scale_list(args.residual_scales) if args.residual_scales else [args.residual_scale]
    scale_names = [f"v1_s{str(s).replace('.', 'p')}" for s in scales]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    shard_map = load_index(cache_dir / "index.jsonl", args.split)
    shard_files = list(shard_map.keys())
    if args.max_shards is not None:
        shard_files = shard_files[:args.max_shards]

    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1_ckpt = torch.load(args.v1_checkpoint, map_location="cpu")
    print(f"[CONFIG] residual_scale={scales}")
    print(f"[CONFIG] condition_mode={args.condition_mode}")
    print(f"[CHECKPOINT] V1={Path(args.v1_checkpoint).resolve()}")
    print(f"[EPOCH] V1={v1_ckpt.get('epoch')}")
    print(f"[INFO] V1 checkpoint epoch={v1_ckpt.get('epoch')} valid={v1_ckpt.get('validation_metrics')}")
    v1.load_state_dict(v1_ckpt["residual_state_dict"])
    v1.eval()

    extractor = None
    estimator = None
    if args.condition_mode == "predicted":
        extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.gtcrn_checkpoint, device="cpu")
        estimator_ckpt = torch.load(args.estimator_checkpoint, map_location="cpu")
        estimator = DegradationEstimator(input_dim=estimator_ckpt["input_dim"], hidden_dim=estimator_ckpt.get("hidden_dim", 32)).to(device)
        estimator.load_state_dict(estimator_ckpt["model_state_dict"])
        estimator.eval()

    rows = []
    window = torch.hann_window(512).pow(0.5)

    with torch.inference_mode():
        for shard_idx, shard_file in enumerate(shard_files, start=1):
            shard = torch.load(cache_dir / shard_file, map_location="cpu", weights_only=False)
            offsets = sorted(shard_map[shard_file])

            clean_batch = shard["clean_spec"][offsets].float()
            degraded_batch = shard["degraded_spec"][offsets].float()
            base_batch = shard["enhanced_base"][offsets].float()
            noise_batch = shard["noise_target"][offsets].float()
            snr_batch = shard["snr_target"][offsets].float()
            bw_batch = shard["bandwidth_target"][offsets].long()
            bit_batch = shard["bit_target"][offsets].long()
            severity_batch = [shard["severity"][off] for off in offsets]

            if args.condition_mode == "oracle":
                cond_batch = build_oracle_condition(noise_batch, snr_batch, bw_batch, bit_batch).to(device)
            else:
                pred_stats = extractor(degraded_batch.to(device))["stats"]
                preds = estimator(pred_stats)
                noise_p = torch.sigmoid(preds["noise_logit"])
                bw_p = torch.softmax(preds["bandwidth_logits"], dim=-1)
                bit_p = torch.softmax(preds["bit_logits"], dim=-1)
                cond_batch = torch.cat([noise_p, preds["snr_pred"], bw_p, bit_p], dim=-1)
                assert cond_batch[:, 1].min().item() >= -1e-6
                assert cond_batch[:, 1].max().item() <= 1.0 + 1e-6
            base_perm = base_batch.permute(0, 3, 2, 1).contiguous().to(device)

            v1_out = v1(base_perm, cond_batch)
            residual = v1_out["residual"]
            v1_batches = {}
            for scale, scale_name in zip(scales, scale_names):
                v1_final_perm = base_perm + scale * residual
                v1_batches[scale_name] = v1_final_perm.permute(0, 3, 2, 1).contiguous().cpu()

            all_specs = [clean_batch, degraded_batch, base_batch] + [v1_batches[name] for name in scale_names]
            all_specs_cat = torch.cat(all_specs, dim=0)
            all_wavs_cat = ri_to_istft(all_specs_cat, window=window)
            chunks = list(all_wavs_cat.chunk(len(all_specs), dim=0))
            clean_wavs = chunks[0].cpu().numpy()
            degraded_wavs = chunks[1].cpu().numpy()
            base_wavs = chunks[2].cpu().numpy()
            v1_wavs_map = {name: chunks[idx + 3].cpu().numpy() for idx, name in enumerate(scale_names)}

            for local_idx, off in enumerate(offsets):
                clean_spec_t = clean_batch[local_idx]
                degraded_spec_t = degraded_batch[local_idx]
                base_spec_t = base_batch[local_idx]
                severity = severity_batch[local_idx]

                clean_spec = clean_spec_t.numpy()
                degraded_spec = degraded_spec_t.numpy()
                base_spec = base_spec_t.numpy()
                clean_wav = clean_wavs[local_idx]
                degraded_wav = degraded_wavs[local_idx]
                base_wav = base_wavs[local_idx]

                def metrics(est_wav, est_spec):
                    out = {
                        "pesq": float('nan'),
                        "stoi": float('nan'),
                        "estoi": float('nan'),
                        "si_sdr": float(si_sdr(clean_wav, est_wav)),
                        "lsd": lsd(clean_spec, est_spec, hf_only=False),
                        "hf_lsd": lsd(clean_spec, est_spec, hf_only=True),
                    }
                    if not args.fast_metrics_only:
                        out["stoi"] = float(stoi(clean_wav, est_wav, 16000, extended=False))
                        out["estoi"] = float(stoi(clean_wav, est_wav, 16000, extended=True))
                        if HAS_PESQ and (not args.disable_pesq):
                            out["pesq"] = float(pesq_fn(16000, clean_wav, est_wav, 'wb'))
                    return out

                degraded_m = metrics(degraded_wav, degraded_spec)
                base_m = metrics(base_wav, base_spec)
                row = {
                    "sample_id": f"{shard_file}:{off}",
                    "shard_file": shard_file,
                    "offset": int(off),
                    "severity": severity,
                    "noise_target": float(noise_batch[local_idx].item()),
                    "snr_target": float(snr_batch[local_idx].item()),
                    "bandwidth_target": int(bw_batch[local_idx].item()),
                    "bit_target": int(bit_batch[local_idx].item()),
                    "degraded": degraded_m,
                    "base": base_m,
                }

                for scale_name in scale_names:
                    v1_spec_t = v1_batches[scale_name][local_idx]
                    v1_spec = v1_spec_t.numpy()
                    v1_wav = v1_wavs_map[scale_name][local_idx]
                    row[scale_name] = metrics(v1_wav, v1_spec)

                rows.append(row)

            if shard_idx % args.progress_interval == 0 or shard_idx == 1 or shard_idx == len(shard_files):
                print(f"[INFO] Processed shard {shard_idx}/{len(shard_files)}")

    methods = ["degraded", "base"] + scale_names
    metric_keys = ["pesq", "stoi", "estoi", "si_sdr", "lsd", "hf_lsd"]

    overall = {m: mean_dict([r[m] for r in rows], metric_keys) for m in methods}
    with open(out_dir / "table_enhancement_overall.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "PESQ", "STOI", "ESTOI", "SI-SDR", "LSD", "HF-LSD"])
        for m in methods:
            vals = overall[m]
            writer.writerow([m, vals["pesq"], vals["stoi"], vals["estoi"], vals["si_sdr"], vals["lsd"], vals["hf_lsd"]])

    with open(out_dir / "table_enhancement_by_severity.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Severity", "Method", "PESQ", "STOI", "ESTOI", "SI-SDR", "LSD", "HF-LSD"])
        for sev in SEVERITIES:
            group = [r for r in rows if r["severity"] == sev]
            for m in methods:
                vals = mean_dict([r[m] for r in group], metric_keys)
                writer.writerow([sev, m, vals["pesq"], vals["stoi"], vals["estoi"], vals["si_sdr"], vals["lsd"], vals["hf_lsd"]])

    with open(out_dir / "sentence_level_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "shard_file", "offset", "severity", "noise_target", "snr_target", "bandwidth_target", "bit_target", "method", "pesq", "stoi", "estoi", "si_sdr", "lsd", "hf_lsd", "delta_stoi_vs_base", "delta_estoi_vs_base", "delta_si_sdr_vs_base", "delta_lsd_vs_base", "delta_hf_lsd_vs_base"])
        for r in rows:
            base_ref = r["base"]
            for m in methods:
                mm = r[m]
                writer.writerow([
                    r["sample_id"], r["shard_file"], r["offset"], r["severity"], r["noise_target"], r["snr_target"], r["bandwidth_target"], r["bit_target"],
                    m, mm["pesq"], mm["stoi"], mm["estoi"], mm["si_sdr"], mm["lsd"], mm["hf_lsd"],
                    (mm["stoi"] - base_ref["stoi"]) if (not math.isnan(mm["stoi"]) and not math.isnan(base_ref["stoi"])) else float('nan'),
                    (mm["estoi"] - base_ref["estoi"]) if (not math.isnan(mm["estoi"]) and not math.isnan(base_ref["estoi"])) else float('nan'),
                    mm["si_sdr"] - base_ref["si_sdr"],
                    mm["lsd"] - base_ref["lsd"],
                    mm["hf_lsd"] - base_ref["hf_lsd"],
                ])

    def plot_severity_metric(metric_key, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(SEVERITIES))
        width = 0.8 / max(len(methods) - 1, 1)
        base_vals = []
        for sev in SEVERITIES:
            group = [r for r in rows if r["severity"] == sev]
            base_vals.append(np.mean([r["base"][metric_key] for r in group]))
        ax.bar(x - width * (len(scale_names) / 2), base_vals, width, label="Base")
        for idx, scale_name in enumerate(scale_names):
            vals = []
            for sev in SEVERITIES:
                group = [r for r in rows if r["severity"] == sev]
                vals.append(np.mean([r[scale_name][metric_key] for r in group]))
            ax.bar(x + width * (idx - (len(scale_names) - 1) / 2), vals, width, label=scale_name)
        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITIES)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Severity")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    if HAS_PESQ and all(not math.isnan(overall[m]["pesq"]) for m in methods):
        plot_severity_metric("pesq", "PESQ", "severity_pesq_curve.png")
    plot_severity_metric("si_sdr", "SI-SDR", "severity_si_sdr_curve.png")
    plot_severity_metric("hf_lsd", "HF-LSD", "severity_hf_lsd_curve.png")

    base_params = 48245
    est_params = 1803
    v1_params = 2184
    def file_mb(p):
        return round(Path(p).stat().st_size / (1024*1024), 4) if Path(p).exists() else float('nan')

    with open(out_dir / "table_complexity.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Total Params", "Added Params", "Residual-stage Trainable Params", "Total Learned Params", "Checkpoint Size MB"])
        writer.writerow(["Base Enhancer", base_params, 0, 0, 0, "n/a"])
        writer.writerow(["EDCR-V1", base_params + est_params + v1_params, est_params + v1_params, v1_params, est_params + v1_params, file_mb(args.v1_checkpoint)])

    print(f"[DONE] Enhancement evaluation finished. Results saved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
