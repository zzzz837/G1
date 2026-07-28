import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from work2.data.audio_utils import ensure_mono
from work2.data.stft_utils import stft_to_ri
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.models.adaptive_residual import AdaptiveResidualModule


def load_test_wavs(manifest_path: Path, max_files=None):
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('split') == 'test':
                rows.append(item)
    if max_files is not None:
        rows = rows[:max_files]
    return rows


def load_audio(path: Path):
    wav, fs = sf.read(str(path), dtype='float32')
    wav = ensure_mono(torch.from_numpy(wav))
    return wav, fs


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_stage(name, fn, n_runs=5):
    times = []
    tracemalloc.start()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'stage': name,
        'mean_time_s': float(np.mean(times)),
        'std_time_s': float(np.std(times)),
        'peak_memory_mb': float(peak / (1024 * 1024)),
    }


def main():
    parser = argparse.ArgumentParser(description='Lightweight CPU/RTF benchmark for Base / Base+Estimator / EDCR-V1-Predicted-0.75')
    parser.add_argument('--manifest', default='outputs/manifest_vctk.jsonl')
    parser.add_argument('--gtcrn-checkpoint', default='checkpoints/model_trained_on_dns3.tar')
    parser.add_argument('--estimator-checkpoint', default='outputs/degradation_estimator_fast/best_model.pt')
    parser.add_argument('--v1-checkpoint', required=True)
    parser.add_argument('--output-csv', default='outputs/runtime_rtf.csv')
    parser.add_argument('--max-files', type=int, default=8)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--residual-scale', type=float, default=0.75)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    manifest_rows = load_test_wavs(Path(args.manifest), max_files=args.max_files)
    if not manifest_rows:
        raise RuntimeError('No test rows found in manifest')

    sample = manifest_rows[0]
    wav, fs = load_audio(Path(sample['path']))
    if fs != 16000:
        raise RuntimeError(f'Expected 16000 Hz, got {fs}')
    duration_s = len(wav) / fs
    window = torch.hann_window(512).pow(0.5)
    spec = stft_to_ri(wav, n_fft=512, hop_length=256, win_length=512, window=window).unsqueeze(0)

    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.gtcrn_checkpoint, device='cpu')
    est_ckpt = torch.load(args.estimator_checkpoint, map_location='cpu')
    estimator = DegradationEstimator(input_dim=est_ckpt['input_dim'], hidden_dim=est_ckpt.get('hidden_dim', 32)).to(device)
    estimator.load_state_dict(est_ckpt['model_state_dict'])
    estimator.eval()

    v1_ckpt = torch.load(args.v1_checkpoint, map_location='cpu')
    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1.load_state_dict(v1_ckpt['residual_state_dict'])
    v1.eval()

    def run_base():
        _ = extractor(spec)

    def run_base_estimator():
        feats = extractor(spec)
        _ = estimator(feats['stats'])

    def run_full():
        feats = extractor(spec)
        preds = estimator(feats['stats'])
        noise_p = torch.sigmoid(preds['noise_logit'])
        bw_p = torch.softmax(preds['bandwidth_logits'], dim=-1)
        bit_p = torch.softmax(preds['bit_logits'], dim=-1)
        cond = torch.cat([noise_p, preds['snr_pred'], bw_p, bit_p], dim=-1)
        base_perm = feats['enhanced_spec'].permute(0, 3, 2, 1).contiguous()
        out = v1(base_perm, cond)
        _ = base_perm + args.residual_scale * out['residual']

    results = []
    for name, fn, params in [
        ('Base', run_base, 48245),
        ('Base+Estimator', run_base_estimator, 48245 + count_params(estimator)),
        ('Base+Estimator+V1', run_full, 48245 + count_params(estimator) + count_params(v1)),
    ]:
        stats = benchmark_stage(name, fn)
        stats['params'] = params
        stats['audio_duration_s'] = duration_s
        stats['rtf'] = stats['mean_time_s'] / duration_s
        results.append(stats)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stage', 'params', 'audio_duration_s', 'mean_time_s', 'std_time_s', 'rtf', 'peak_memory_mb'])
        writer.writeheader()
        writer.writerows(results)

    print(f'[DONE] Wrote runtime summary to {out_path}')


if __name__ == '__main__':
    main()
