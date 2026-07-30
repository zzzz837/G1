"""
Condition effectiveness evaluation for EDCR.

Compares four condition modes on the same residual checkpoint:
- zero
- shuffled
- predicted
- oracle

Outputs paper-ready tables and boxplots.
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
from work2.data.label_utils import denormalize_snr, normalize_snr
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2Formal

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


def si_sdr(ref, est, eps=1e-8):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64) - np.mean(ref[:n])
    est = est[:n].astype(np.float64) - np.mean(est[:n])
    ref_energy = np.sum(ref ** 2) + eps
    proj = np.dot(est, ref) / ref_energy
    s_target = proj * ref
    e_noise = est - s_target
    return float(10 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


def lsd(clean_spec, test_spec, hf_only=False):
    clean_mag = np.sqrt(clean_spec[..., 0] ** 2 + clean_spec[..., 1] ** 2 + 1e-12)
    test_mag = np.sqrt(test_spec[..., 0] ** 2 + test_spec[..., 1] ** 2 + 1e-12)
    if hf_only:
        start = HF_START_BIN
        clean_mag = clean_mag[start:]
        test_mag = test_mag[start:]
    val = np.sqrt(np.mean((20 * np.log10(clean_mag + 1e-12) - 20 * np.log10(test_mag + 1e-12)) ** 2, axis=0)).mean()
    return float(val)


def build_oracle_condition(noise_target, snr_target, bandwidth_target, bit_target):
    B = noise_target.shape[0]
    bw_oh = torch.zeros(B, 3, dtype=torch.float32, device=noise_target.device)
    bw_oh.scatter_(1, bandwidth_target.view(-1, 1), 1.0)
    bit_oh = torch.zeros(B, 4, dtype=torch.float32, device=noise_target.device)
    bit_oh.scatter_(1, bit_target.view(-1, 1), 1.0)
    snr_norm = torch.clamp(snr_target / 40.0, 0.0, 1.0)
    return torch.cat([noise_target, snr_norm, bw_oh, bit_oh], dim=1)


def build_zero_condition(batch_size, device):
    return torch.zeros(batch_size, 9, dtype=torch.float32, device=device)


def build_shuffled_condition(oracle_cond, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    idx = torch.randperm(oracle_cond.shape[0], generator=generator)
    return oracle_cond[idx.to(oracle_cond.device)]


def build_predicted_condition(stats, estimator):
    preds = estimator(stats)
    noise_p = torch.sigmoid(preds["noise_logit"])
    bw_p = torch.softmax(preds["bandwidth_logits"], dim=-1)
    bit_p = torch.softmax(preds["bit_logits"], dim=-1)
    return torch.cat([noise_p, preds["snr_pred"], bw_p, bit_p], dim=-1)


def plot_box(data_dict, metric_key, ylabel, savepath):
    labels = list(data_dict.keys())
    values = [data_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#d9e8fb')
        patch.set_edgecolor('#4f81bd')
    for med in bp['medians']:
        med.set_color('#c0504d')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} under different condition sources")
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--gtcrn-checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--estimator-checkpoint", type=str, default="outputs/degradation_estimator_fast/best_model.pt")
    parser.add_argument("--residual-checkpoint", type=str, required=True)
    parser.add_argument("--residual-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--output-dir", type=str, default="outputs/paper_results_condition")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--fast-metrics-only", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.65)
    parser.add_argument("--disable-pesq", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
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

    if args.residual_version == "v1":
        residual_module = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    else:
        residual_module = AdaptiveResidualModuleV2Formal(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    residual_ckpt = torch.load(args.residual_checkpoint, map_location='cpu')
    print(f"[CONFIG] residual_scale={args.residual_scale}")
    print(f"[CONFIG] seed={args.seed}")
    print(f"[CHECKPOINT] residual={Path(args.residual_checkpoint).resolve()}")
    print(f"[EPOCH] residual={residual_ckpt.get('epoch')}")
    residual_module.load_state_dict(residual_ckpt['residual_state_dict'])
    residual_module.eval()

    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.gtcrn_checkpoint, device='cpu')
    estimator_ckpt = torch.load(args.estimator_checkpoint, map_location='cpu')
    estimator = DegradationEstimator(input_dim=estimator_ckpt['input_dim'], hidden_dim=estimator_ckpt.get('hidden_dim', 32)).to(device)
    estimator.load_state_dict(estimator_ckpt['model_state_dict'])
    estimator.eval()

    rows = []
    window = torch.hann_window(512).pow(0.5)

    with torch.inference_mode():
        for shard_idx, shard_file in enumerate(shard_files, start=1):
            shard = torch.load(cache_dir / shard_file, map_location='cpu', weights_only=False)
            offsets = sorted(shard_map[shard_file])
            clean_batch = shard['clean_spec'][offsets].float()
            base_batch = shard['enhanced_base'][offsets].float()
            degraded_batch = shard['degraded_spec'][offsets].float()
            noise_batch = shard['noise_target'][offsets].float()
            snr_batch = shard['snr_target'][offsets].float()
            bw_batch = shard['bandwidth_target'][offsets].long()
            bit_batch = shard['bit_target'][offsets].long()
            severity_batch = [shard['severity'][off] for off in offsets]

            oracle_cond = build_oracle_condition(noise_batch, snr_batch, bw_batch, bit_batch)
            shuffled_cond = build_shuffled_condition(oracle_cond, seed=args.seed + shard_idx)
            zero_cond = build_zero_condition(oracle_cond.shape[0], oracle_cond.device)

            # predicted conditions from cached degraded specs via frozen extractor+estimator
            pred_stats = extractor(degraded_batch.to(device))["stats"]
            predicted_cond = build_predicted_condition(pred_stats, estimator).cpu()

            base_perm = base_batch.permute(0, 3, 2, 1).contiguous().to(device)
            cond_map = {
                'zero': zero_cond.to(device),
                'shuffled': shuffled_cond.to(device),
                'predicted': predicted_cond.to(device),
                'oracle': oracle_cond.to(device),
            }
            enhanced = {}
            for mode, cond in cond_map.items():
                out = residual_module(base_perm, cond)
                scaled_final = base_perm + args.residual_scale * out['residual']
                enhanced[mode] = scaled_final.permute(0, 3, 2, 1).contiguous().cpu()

            for local_idx, off in enumerate(offsets):
                clean_spec = clean_batch[local_idx].numpy()
                base_spec = base_batch[local_idx].numpy()
                degraded_spec = degraded_batch[local_idx].numpy()
                severity = severity_batch[local_idx]
                clean_wav = ri_to_istft(torch.tensor(clean_spec), window=window).numpy()
                base_wav = ri_to_istft(torch.tensor(base_spec), window=window).numpy()
                degraded_wav = ri_to_istft(torch.tensor(degraded_spec), window=window).numpy()

                base_metrics = {
                    'stoi': float('nan'), 'estoi': float('nan'),
                    'si_sdr': si_sdr(clean_wav, base_wav),
                    'lsd': lsd(clean_spec, base_spec, False),
                    'hf_lsd': lsd(clean_spec, base_spec, True),
                }
                for mode in ['zero', 'shuffled', 'predicted', 'oracle']:
                    spec = enhanced[mode][local_idx].numpy()
                    wav = ri_to_istft(torch.tensor(spec), window=window).numpy()
                    met = {
                        'stoi': float('nan'), 'estoi': float('nan'),
                        'si_sdr': si_sdr(clean_wav, wav),
                        'lsd': lsd(clean_spec, spec, False),
                        'hf_lsd': lsd(clean_spec, spec, True),
                    }
                    if not args.fast_metrics_only:
                        met['stoi'] = float(stoi(clean_wav, wav, 16000, extended=False))
                        met['estoi'] = float(stoi(clean_wav, wav, 16000, extended=True))
                    rows.append({
                        'sample_id': f'{shard_file}:{off}',
                        'severity': severity,
                        'method': mode,
                        'base_si_sdr': base_metrics['si_sdr'],
                        'base_lsd': base_metrics['lsd'],
                        'base_hf_lsd': base_metrics['hf_lsd'],
                        'stoi': met['stoi'],
                        'estoi': met['estoi'],
                        'si_sdr': met['si_sdr'],
                        'lsd': met['lsd'],
                        'hf_lsd': met['hf_lsd'],
                        'delta_si_sdr_vs_base': met['si_sdr'] - base_metrics['si_sdr'],
                        'delta_lsd_vs_base': met['lsd'] - base_metrics['lsd'],
                        'delta_hf_lsd_vs_base': met['hf_lsd'] - base_metrics['hf_lsd'],
                    })
            if shard_idx % 20 == 0 or shard_idx == 1:
                print(f'[INFO] Processed shard {shard_idx}/{len(shard_files)}')

    # save sentence-level csv
    sentence_csv = out_dir / 'sentence_level_condition_metrics.csv'
    with open(sentence_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # aggregate table
    modes = ['zero', 'shuffled', 'predicted', 'oracle']
    metric_fields = ['stoi', 'estoi', 'si_sdr', 'lsd', 'hf_lsd']
    with open(out_dir / 'table_condition_effectiveness.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Condition', 'STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD'])
        for mode in modes:
            group = [r for r in rows if r['method'] == mode]
            vals = {k: float(np.mean([g[k] for g in group])) for k in metric_fields}
            writer.writerow([mode, vals['stoi'], vals['estoi'], vals['si_sdr'], vals['lsd'], vals['hf_lsd']])

    # boxplots (skip STOI/ESTOI if fast mode)
    for metric_key, ylabel, fname in [
        ('si_sdr', 'SI-SDR', 'condition_si_sdr_boxplot.png'),
        ('lsd', 'LSD', 'condition_lsd_boxplot.png'),
        ('hf_lsd', 'HF-LSD', 'condition_hf_lsd_boxplot.png'),
    ]:
        data_dict = {m: [r[metric_key] for r in rows if r['method'] == m] for m in modes}
        plot_box(data_dict, metric_key, ylabel, out_dir / fname)

    if not args.fast_metrics_only:
        for metric_key, ylabel, fname in [
            ('stoi', 'STOI', 'condition_stoi_boxplot.png'),
            ('estoi', 'ESTOI', 'condition_estoi_boxplot.png'),
        ]:
            data_dict = {m: [r[metric_key] for r in rows if r['method'] == m] for m in modes}
            plot_box(data_dict, metric_key, ylabel, out_dir / fname)

    print(f'[DONE] Condition effectiveness evaluation finished. Results saved to {out_dir}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[FATAL] {e}', file=sys.stderr)
        raise
