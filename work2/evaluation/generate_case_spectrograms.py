"""
Generate representative case spectrogram figures for thesis.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from work2.data.stft_utils import ri_to_istft
from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2Formal


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


def spec_mag_db(spec):
    mag = np.sqrt(spec[...,0]**2 + spec[...,1]**2 + 1e-12)
    return 20*np.log10(mag + 1e-12)


def plot_case(savepath, clean, degraded, base, v1, v2, vmax=None, vmin=None):
    titles = ["Clean", "Degraded", "Base", "EDCR-V1", "EDCR-V2", "Base Error", "V1 Error", "V2 Error"]
    specs = [
        spec_mag_db(clean),
        spec_mag_db(degraded),
        spec_mag_db(base),
        spec_mag_db(v1),
        spec_mag_db(v2),
        np.abs(spec_mag_db(base) - spec_mag_db(clean)),
        np.abs(spec_mag_db(v1) - spec_mag_db(clean)),
        np.abs(spec_mag_db(v2) - spec_mag_db(clean)),
    ]
    if vmax is None:
        vmax = max(float(np.max(s)) for s in specs[:5])
    if vmin is None:
        vmin = min(float(np.min(s)) for s in specs[:5])

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    for ax, title, spec in zip(axes.flat, titles, specs):
        if 'Error' in title:
            im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
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
    parser.add_argument('--v2-checkpoint', type=str, default='outputs/adaptive_residual_cached_v2_oracle_fixed/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='outputs/paper_results_cases')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    shard_map = load_index(cache_dir / 'index.jsonl', args.split)
    selected = {'light': None, 'medium': None, 'heavy': None}

    # choose first representative sample per target severity
    for shard_file, offsets in shard_map.items():
        shard = torch.load(cache_dir / shard_file, map_location='cpu', weights_only=False)
        for off in offsets:
            sev = shard['severity'][off]
            if sev in selected and selected[sev] is None:
                selected[sev] = (shard_file, off)
        if all(v is not None for v in selected.values()):
            break

    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v2 = AdaptiveResidualModuleV2Formal(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1.load_state_dict(torch.load(args.v1_checkpoint, map_location='cpu')['residual_state_dict'])
    v2.load_state_dict(torch.load(args.v2_checkpoint, map_location='cpu')['residual_state_dict'])
    v1.eval(); v2.eval()

    for idx, sev in enumerate(['light', 'medium', 'heavy'], start=1):
        shard_file, off = selected[sev]
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
            v1_spec = v1(base_perm, cond)['enhanced_final'].permute(0,3,2,1)[0].cpu().numpy()
            v2_spec = v2(base_perm, cond)['enhanced_final'].permute(0,3,2,1)[0].cpu().numpy()
        plot_case(out_dir / f'spectrogram_case_0{idx}.png', clean.numpy(), degraded.numpy(), base.numpy(), v1_spec, v2_spec)

    print(f'[DONE] Generated case spectrograms in {out_dir}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[FATAL] {e}', file=sys.stderr)
        raise
