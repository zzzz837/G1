import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch

from work2.data.audio_utils import ensure_mono
from work2.data.degradation import apply_composite_degradation, sample_degradation_config
from work2.data.random_utils import seed_everything


SEVERITIES = ["clean", "light", "medium", "heavy"]


def load_test_entries(manifest_path: Path):
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('split') == 'test':
                rows.append(item)
    return rows


def make_noise(length: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(length, generator=g, dtype=torch.float32) * 0.1
    return noise - noise.mean()


def main():
    parser = argparse.ArgumentParser(description='Export a small external-model testset from G1 test split')
    parser.add_argument('--manifest', default='D:\\workshop\\G1\\outputs\\04_评测缓存与清单_复现实验用\\manifest_vctk.jsonl')
    parser.add_argument('--output-root', default='D:\\workshop\\FullSubNet-plus-master\\data\\test_noisy_small')
    parser.add_argument('--samples-per-severity', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    clean_dir = output_root / 'clean'
    noisy_dir = output_root / 'noisy'
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    entries = load_test_entries(manifest_path)
    if not entries:
        raise RuntimeError(f'No test entries found in {manifest_path}')

    selected_entries = entries[:args.samples_per_severity]
    csv_rows = []
    sample_index = 0

    for entry in selected_entries:
        audio_path = Path(entry['path'])
        if not audio_path.exists():
            continue
        wav, fs = sf.read(str(audio_path), dtype='float32')
        if fs != 16000:
            continue
        clean = ensure_mono(torch.from_numpy(wav)).to(torch.float32)

        for sev_idx, severity in enumerate(SEVERITIES):
            local_seed = args.seed * 10000 + sample_index * 100 + sev_idx
            cfg = sample_degradation_config(severity, local_seed)
            noise = make_noise(clean.numel(), local_seed) if cfg.add_noise else None
            result = apply_composite_degradation(clean.clone(), cfg, noise.clone() if noise is not None else None)

            stem = f'test_{sample_index:04d}_{severity}'
            clean_path = clean_dir / f'{stem}.wav'
            noisy_path = noisy_dir / f'{stem}.wav'

            sf.write(str(clean_path), result.clean.numpy(), 16000)
            sf.write(str(noisy_path), result.degraded.numpy(), 16000)

            csv_rows.append({
                'sample_id': stem,
                'source_path': str(audio_path),
                'severity': severity,
                'clean_wav': str(clean_path),
                'noisy_wav': str(noisy_path),
            })

        sample_index += 1

    csv_path = output_root / 'manifest_small_test.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'source_path', 'severity', 'clean_wav', 'noisy_wav'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'[DONE] Exported small testset to {output_root}')
    print(f'[DONE] Wrote manifest to {csv_path}')


if __name__ == '__main__':
    main()
