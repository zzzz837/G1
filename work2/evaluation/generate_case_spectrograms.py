"""
Generate representative case spectrogram figures for thesis.

Final thesis version focuses on the deployed main model:
Clean / Degraded / Base / EDCR-V1-Predicted-0.75 plus error maps.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from work2.models.adaptive_residual import AdaptiveResidualModule


def load_index(index_path: Path, split: str):
    shard_to_entries = {}
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get('split') != split:
                continue
            shard_to_entries.setdefault(e['shard_file'], []).append(e['offset'])
    return shard_to_entries


def build_oracle_condition(noise_target, snr_target, bandwidth_target, bit_target):
    B = noise_target.shape[0]
    bw_oh = torch.zeros(B, 3, dtype=torch.float32, device=noise_target.device)
    bw_oh.scatter_(1, bandwidth_target.view(-1, 1), 1.0)
    bit_oh = torch.zeros(B, 4, dtype=torch.float32, device=noise_target.device)
    bit_oh.scatter_(1, bit_target.view(-1, 1), 1.0)
    snr_norm = torch.clamp(snr_target / 40.0, 0.0, 1.0)
    return torch.cat([noise_target, snr_norm, bw_oh, bit_oh], dim=1)


def load_sentence_rows(csv_path: Path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def select_representative_samples(sentence_csv: Path):
    rows = load_sentence_rows(sentence_csv)
    selected = {}
    for sev in ['light', 'medium', 'heavy']:
        group = [r for r in rows if r['method'] == 'v1_s0p75' and r['severity'] == sev]
        if not group:
            raise RuntimeError(f'No rows found for severity={sev} in {sentence_csv}')
        deltas = np.array([float(r['delta_hf_lsd_vs_base']) for r in group], dtype=np.float64)
        median = float(np.median(deltas))
        best = min(group, key=lambda r: abs(float(r['delta_hf_lsd_vs_base']) - median))
        selected[sev] = best['sample_id']
    return selected


def spec_mag_db(spec):
    mag = np.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-12)
    return 20 * np.log10(mag + 1e-12)


def plot_case(savepath, clean, degraded, base, v1, vmax=None, vmin=None):
    titles = ["Clean", "Degraded", "Base", "EDCR-V1-Predicted-0.75", "Base Error", "EDCR Error"]
    specs = [
        spec_mag_db(clean),
        spec_mag_db(degraded),
        spec_mag_db(base),
        spec_mag_db(v1),
        np.abs(spec_mag_db(base) - spec_mag_db(clean)),
        np.abs(spec_mag_db(v1) - spec_mag_db(clean)),
    ]
    if vmax is None:
        vmax = max(float(np.max(s)) for s in specs[:4])
    if vmin is None:
        vmin = min(float(np.min(s)) for s in specs[:4])

    error_specs = specs[4:]
    err_vmax = np.percentile(np.concatenate([e.ravel() for e in error_specs]), 99)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5))
    for ax, title, spec in zip(axes.flat, titles, specs):
        if 'Error' in title:
            im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=err_vmax)
        else:
            im = ax.imshow(spec, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frequency Bin')
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='outputs/adaptive_cache_sharded')
    parser.add_argument('--v1-checkpoint', type=str, default='outputs/adaptive_residual_cached_v1_oracle/best_model.pt')
    parser.add_argument('--sentence-csv', type=str, default='outputs/paper_results_scale_075_predicted_full/sentence_level_metrics.csv')
    parser.add_argument('--output-dir', type=str, default='outputs/paper_results_cases')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--residual-scale', type=float, default=0.75)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    shard_map = load_index(cache_dir / 'index.jsonl', args.split)
    selected = select_representative_samples(Path(args.sentence_csv))

    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1_ckpt = torch.load(args.v1_checkpoint, map_location='cpu')
    print(f"[CONFIG] residual_scale={args.residual_scale}")
    print(f"[CHECKPOINT] V1={Path(args.v1_checkpoint).resolve()}")
    print(f"[EPOCH] V1={v1_ckpt.get('epoch')}")
    v1.load_state_dict(v1_ckpt['residual_state_dict'])
    v1.eval()

    for idx, sev in enumerate(['light', 'medium', 'heavy'], start=1):
        sample_id = selected[sev]
        shard_file, off_str = sample_id.split(':')
        off = int(off_str)
        shard = torch.load(cache_dir / shard_file, map_location='cpu', weights_only=False)
        clean = shard['clean_spec'][off].float()
        degraded = shard['degraded_spec'][off].float()
        base = shard['enhanced_base'][off].float()
        noise_t = shard['noise_target'][off].view(1,1).float()
        snr_t = shard['snr_target'][off].view(1,1).float()
        bw_t = shard['bandwidth_target'][off].view(1).long()
        bit_t = shard['bit_target'][off].view(1).long()
        cond = build_oracle_condition(noise_t, snr_t, bw_t, bit_t).to(device)
        base_perm = base.unsqueeze(0).permute(0,3,2,1).contiguous().to(device)
        with torch.inference_mode():
            v1_out = v1(base_perm, cond)
            v1_spec = (base_perm + args.residual_scale * v1_out['residual']).permute(0,3,2,1)[0].cpu().numpy()
        plot_case(out_dir / f'spectrogram_case_0{idx}.png', clean.numpy(), degraded.numpy(), base.numpy(), v1_spec)

    print(f'[DONE] Generated case spectrograms in {out_dir}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[FATAL] {e}', file=sys.stderr)
        raise
