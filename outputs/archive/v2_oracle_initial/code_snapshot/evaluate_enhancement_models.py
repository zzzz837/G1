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
from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2Formal

try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except Exception:
    HAS_PESQ = False

SEVERITIES = ["clean", "light", "medium", "heavy"]


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
    # robust SI-SDR implementation with explicit length alignment
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
    # input: (F,T,2)
    clean_mag = np.sqrt(clean_spec[...,0]**2 + clean_spec[...,1]**2 + 1e-12)
    test_mag = np.sqrt(test_spec[...,0]**2 + test_spec[...,1]**2 + 1e-12)
    if hf_only:
        start = int(clean_mag.shape[0] * 0.55)
        clean_mag = clean_mag[start:]
        test_mag = test_mag[start:]
    val = np.sqrt(np.mean((20*np.log10(clean_mag+1e-12)-20*np.log10(test_mag+1e-12))**2, axis=0)).mean()
    return float(val)


def enhance_with_module(base_spec_ri, cond, module, device):
    base_perm = torch.tensor(base_spec_ri, dtype=torch.float32, device=device).unsqueeze(0).permute(0,3,2,1)
    cond = cond.to(device)
    with torch.inference_mode():
        out = module(base_perm, cond)
    final_ri = out["enhanced_final"].permute(0,3,2,1)[0].cpu().numpy()
    return final_ri, out


def mean_dict(list_dicts, keys):
    return {k: float(np.mean([d[k] for d in list_dicts])) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--v1-checkpoint", type=str, default="outputs/adaptive_residual_cached_v1_oracle_fixed/best_model.pt")
    parser.add_argument("--v2-checkpoint", type=str, default="outputs/adaptive_residual_cached_v2_oracle_fixed/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/paper_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    shard_map = load_index(cache_dir / "index.jsonl", args.split)
    shard_files = list(shard_map.keys())
    if args.max_shards is not None:
        shard_files = shard_files[:args.max_shards]

    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v2 = AdaptiveResidualModuleV2Formal(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1_ckpt = torch.load(args.v1_checkpoint, map_location="cpu")
    v2_ckpt = torch.load(args.v2_checkpoint, map_location="cpu")
    print(f"[INFO] V1 checkpoint epoch={v1_ckpt.get('epoch')} valid={v1_ckpt.get('validation_metrics')}")
    print(f"[INFO] V2 checkpoint epoch={v2_ckpt.get('epoch')} valid={v2_ckpt.get('validation_metrics')}")
    v1.load_state_dict(v1_ckpt["residual_state_dict"])
    v2.load_state_dict(v2_ckpt["residual_state_dict"])
    v1.eval(); v2.eval()

    rows = []
    window = torch.hann_window(512).pow(0.5)

    for shard_idx, shard_file in enumerate(shard_files, start=1):
        shard = torch.load(cache_dir / shard_file, map_location="cpu", weights_only=False)
        offsets = sorted(shard_map[shard_file])
        for off in offsets:
            clean_spec = shard["clean_spec"][off].numpy()       # (F,T,2)
            degraded_spec = shard["degraded_spec"][off].numpy() # (F,T,2)
            base_spec = shard["enhanced_base"][off].numpy()     # (F,T,2)
            noise_t = shard["noise_target"][off].view(1,1).float()
            snr_t = shard["snr_target"][off].view(1,1).float()
            bw_t = shard["bandwidth_target"][off].view(1).long()
            bit_t = shard["bit_target"][off].view(1).long()
            severity = shard["severity"][off]
            cond = build_oracle_condition(noise_t, snr_t, bw_t, bit_t)

            v1_spec, _ = enhance_with_module(base_spec, cond, v1, device)
            v2_spec, _ = enhance_with_module(base_spec, cond, v2, device)

            clean_wav = ri_to_istft(torch.tensor(clean_spec), window=window).numpy()
            degraded_wav = ri_to_istft(torch.tensor(degraded_spec), window=window).numpy()
            base_wav = ri_to_istft(torch.tensor(base_spec), window=window).numpy()
            v1_wav = ri_to_istft(torch.tensor(v1_spec), window=window).numpy()
            v2_wav = ri_to_istft(torch.tensor(v2_spec), window=window).numpy()

            def metrics(est_wav, est_spec):
                out = {
                    "pesq": float('nan') if not HAS_PESQ else float(pesq_fn(16000, clean_wav, est_wav, 'wb')),
                    "stoi": float(stoi(clean_wav, est_wav, 16000, extended=False)),
                    "estoi": float(stoi(clean_wav, est_wav, 16000, extended=True)),
                    "si_sdr": float(si_sdr(clean_wav, est_wav)),
                    "lsd": lsd(clean_spec, est_spec, hf_only=False),
                    "hf_lsd": lsd(clean_spec, est_spec, hf_only=True),
                }
                return out

            degraded_m = metrics(degraded_wav, degraded_spec)
            base_m = metrics(base_wav, base_spec)
            v1_m = metrics(v1_wav, v1_spec)
            v2_m = metrics(v2_wav, v2_spec)

            rows.append({
                "severity": severity,
                "degraded": degraded_m,
                "base": base_m,
                "v1": v1_m,
                "v2": v2_m,
            })

        if shard_idx % 50 == 0 or shard_idx == 1:
            print(f"[INFO] Processed shard {shard_idx}/{len(shard_files)}")

    # overall tables
    methods = ["degraded", "base", "v1", "v2"]
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
            for m in methods[1:]:  # base/v1/v2, degraded optional elsewhere
                vals = mean_dict([r[m] for r in group], metric_keys)
                writer.writerow([sev, m, vals["pesq"], vals["stoi"], vals["estoi"], vals["si_sdr"], vals["lsd"], vals["hf_lsd"]])

    # save sentence-level csv for later boxplots / significance testing
    with open(out_dir / "sentence_level_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["severity", "method", "pesq", "stoi", "estoi", "si_sdr", "lsd", "hf_lsd"])
        for r in rows:
            for m in methods:
                mm = r[m]
                writer.writerow([r["severity"], m, mm["pesq"], mm["stoi"], mm["estoi"], mm["si_sdr"], mm["lsd"], mm["hf_lsd"]])

    # curves by severity
    def plot_severity_metric(metric_key, ylabel, filename, higher_better=True):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(SEVERITIES))
        width = 0.25
        base_vals, v1_vals, v2_vals = [], [], []
        for sev in SEVERITIES:
            group = [r for r in rows if r["severity"] == sev]
            base_vals.append(np.mean([r["base"][metric_key] for r in group]))
            v1_vals.append(np.mean([r["v1"][metric_key] for r in group]))
            v2_vals.append(np.mean([r["v2"][metric_key] for r in group]))
        ax.bar(x - width, base_vals, width, label="Base")
        ax.bar(x, v1_vals, width, label="EDCR-V1")
        ax.bar(x + width, v2_vals, width, label="EDCR-V2")
        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITIES)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Severity")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches='tight')
        plt.close(fig)

    # only draw PESQ curve if PESQ is available
    if HAS_PESQ and all(not math.isnan(overall[m]["pesq"]) for m in methods):
        plot_severity_metric("pesq", "PESQ", "severity_pesq_curve.png")
    plot_severity_metric("si_sdr", "SI-SDR", "severity_si_sdr_curve.png")
    plot_severity_metric("hf_lsd", "HF-LSD", "severity_hf_lsd_curve.png", higher_better=False)

    # complexity table
    # params: base 48245, estimator 1803, v1 2184, v2 2295
    # model sizes from checkpoint files
    base_params = 48245
    est_params = 1803
    v1_params = 2184
    v2_params = 2295
    def file_mb(p):
        return round(Path(p).stat().st_size / (1024*1024), 4) if Path(p).exists() else float('nan')

    with open(out_dir / "table_complexity.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Total Params", "Added Params", "Residual-stage Trainable Params", "Total Learned Params", "Checkpoint Size MB"])
        writer.writerow(["Base Enhancer", base_params, 0, 0, 0, "n/a"])
        writer.writerow(["EDCR-V1", base_params + est_params + v1_params, est_params + v1_params, v1_params, est_params + v1_params, file_mb(args.v1_checkpoint)])
        writer.writerow(["EDCR-V2", base_params + est_params + v2_params, est_params + v2_params, v2_params, est_params + v2_params, file_mb(args.v2_checkpoint)])

    print(f"[DONE] Enhancement evaluation finished. Results saved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
