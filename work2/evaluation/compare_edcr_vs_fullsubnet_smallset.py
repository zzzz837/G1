import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pystoi import stoi

from work2.data.stft_utils import stft_to_ri
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.models.adaptive_residual import AdaptiveResidualModule


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
    clean_mag = np.sqrt(clean_spec[..., 0] ** 2 + clean_spec[..., 1] ** 2 + 1e-12)
    test_mag = np.sqrt(test_spec[..., 0] ** 2 + test_spec[..., 1] ** 2 + 1e-12)
    if hf_only:
        start = 128
        clean_mag = clean_mag[start:]
        test_mag = test_mag[start:]
    val = np.sqrt(np.mean((20 * np.log10(clean_mag + 1e-12) - 20 * np.log10(test_mag + 1e-12)) ** 2, axis=0)).mean()
    return float(val)


def mean_dict(list_dicts, keys):
    return {k: float(np.mean([d[k] for d in list_dicts])) for k in keys}


def load_manifest(csv_path: Path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Compare EDCR vs FullSubNet+ on the same small testset')
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--fullsubnet-enhanced-dir', required=True)
    parser.add_argument('--gtcrn-checkpoint', required=True)
    parser.add_argument('--estimator-checkpoint', required=True)
    parser.add_argument('--v1-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--residual-scale', type=float, default=0.75)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    manifest_rows = load_manifest(Path(args.manifest_csv))
    fullsubnet_dir = Path(args.fullsubnet_enhanced_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.gtcrn_checkpoint, device='cpu')
    est_ckpt = torch.load(args.estimator_checkpoint, map_location='cpu')
    estimator = DegradationEstimator(input_dim=est_ckpt['input_dim'], hidden_dim=est_ckpt.get('hidden_dim', 32)).to(device)
    estimator.load_state_dict(est_ckpt['model_state_dict'])
    estimator.eval()

    v1_ckpt = torch.load(args.v1_checkpoint, map_location='cpu')
    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1.load_state_dict(v1_ckpt['residual_state_dict'])
    v1.eval()

    rows = []
    window = torch.hann_window(512).pow(0.5)

    for item in manifest_rows:
        sample_id = item['sample_id']
        severity = item['severity']
        clean_path = Path(item['clean_wav'])
        noisy_path = Path(item['noisy_wav'])
        fullsubnet_path = fullsubnet_dir / f'{sample_id}.wav'
        if not fullsubnet_path.exists():
            continue

        clean_wav, fs1 = sf.read(str(clean_path), dtype='float32')
        noisy_wav, fs2 = sf.read(str(noisy_path), dtype='float32')
        fullsubnet_wav, fs3 = sf.read(str(fullsubnet_path), dtype='float32')
        if fs1 != 16000 or fs2 != 16000 or fs3 != 16000:
            raise RuntimeError('Expected all sample rates to be 16000')

        clean_spec = stft_to_ri(torch.from_numpy(clean_wav), window=window).numpy()
        noisy_spec_t = stft_to_ri(torch.from_numpy(noisy_wav), window=window)
        noisy_spec = noisy_spec_t.numpy()
        fullsubnet_spec = stft_to_ri(torch.from_numpy(fullsubnet_wav), window=window).numpy()

        with torch.inference_mode():
            feats = extractor(noisy_spec_t.unsqueeze(0))
            base_spec_t = feats['enhanced_spec'][0].cpu()
            preds = estimator(feats['stats'])
            noise_p = torch.sigmoid(preds['noise_logit'])
            bw_p = torch.softmax(preds['bandwidth_logits'], dim=-1)
            bit_p = torch.softmax(preds['bit_logits'], dim=-1)
            cond = torch.cat([noise_p, preds['snr_pred'], bw_p, bit_p], dim=-1)
            base_perm = base_spec_t.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
            out = v1(base_perm, cond)
            edcr_spec = (base_perm + args.residual_scale * out['residual']).permute(0, 3, 2, 1)[0].cpu().numpy()

        base_wav = sf.read(str(noisy_path), dtype='float32')[0]  # placeholder, replaced below
        import torch as _torch
        from work2.data.stft_utils import ri_to_istft
        base_wav = ri_to_istft(base_spec_t, window=window, length=len(clean_wav)).numpy()
        edcr_wav = ri_to_istft(_torch.from_numpy(edcr_spec), window=window, length=len(clean_wav)).numpy()

        def metrics(est_wav, est_spec):
            return {
                'stoi': float(stoi(clean_wav, est_wav, 16000, extended=False)),
                'estoi': float(stoi(clean_wav, est_wav, 16000, extended=True)),
                'si_sdr': float(si_sdr(clean_wav, est_wav)),
                'lsd': lsd(clean_spec, est_spec, hf_only=False),
                'hf_lsd': lsd(clean_spec, est_spec, hf_only=True),
            }

        degraded_m = metrics(noisy_wav, noisy_spec)
        base_m = metrics(base_wav, base_spec_t.numpy())
        edcr_m = metrics(edcr_wav, edcr_spec)
        fullsubnet_m = metrics(fullsubnet_wav, fullsubnet_spec)

        rows.append({
            'sample_id': sample_id,
            'severity': severity,
            'degraded': degraded_m,
            'base': base_m,
            'edcr_predicted_0p75': edcr_m,
            'fullsubnet_plus': fullsubnet_m,
        })

    methods = ['degraded', 'base', 'edcr_predicted_0p75', 'fullsubnet_plus']
    metric_keys = ['stoi', 'estoi', 'si_sdr', 'lsd', 'hf_lsd']

    overall = {m: mean_dict([r[m] for r in rows], metric_keys) for m in methods}
    with open(out_dir / 'table_overall.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD'])
        for m in methods:
            vals = overall[m]
            writer.writerow([m, vals['stoi'], vals['estoi'], vals['si_sdr'], vals['lsd'], vals['hf_lsd']])

    severities = ['clean', 'light', 'medium', 'heavy']
    with open(out_dir / 'table_by_severity.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Severity', 'Method', 'STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD'])
        for sev in severities:
            group = [r for r in rows if r['severity'] == sev]
            for m in methods:
                vals = mean_dict([r[m] for r in group], metric_keys)
                writer.writerow([sev, m, vals['stoi'], vals['estoi'], vals['si_sdr'], vals['lsd'], vals['hf_lsd']])

    with open(out_dir / 'sentence_level_metrics.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'severity', 'method', 'stoi', 'estoi', 'si_sdr', 'lsd', 'hf_lsd'])
        for r in rows:
            for m in methods:
                mm = r[m]
                writer.writerow([r['sample_id'], r['severity'], m, mm['stoi'], mm['estoi'], mm['si_sdr'], mm['lsd'], mm['hf_lsd']])

    print(f'[DONE] EDCR vs FullSubNet+ comparison saved to {out_dir}')


if __name__ == '__main__':
    main()
