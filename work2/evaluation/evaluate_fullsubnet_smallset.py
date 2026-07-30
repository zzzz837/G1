import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi


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


def stft_ri(wav, n_fft=512, hop_length=256, win_length=512):
    import torch
    window = torch.hann_window(win_length).pow(0.5)
    spec = torch.stft(torch.from_numpy(wav), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    return torch.view_as_real(spec).numpy()


def lsd(clean_spec, test_spec, hf_only=False):
    clean_mag = np.sqrt(clean_spec[..., 0] ** 2 + clean_spec[..., 1] ** 2 + 1e-12)
    test_mag = np.sqrt(test_spec[..., 0] ** 2 + test_spec[..., 1] ** 2 + 1e-12)
    if hf_only:
        start = 128
        clean_mag = clean_mag[start:]
        test_mag = test_mag[start:]
    val = np.sqrt(np.mean((20 * np.log10(clean_mag + 1e-12) - 20 * np.log10(test_mag + 1e-12)) ** 2, axis=0)).mean()
    return float(val)


def load_manifest(csv_path: Path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def mean_dict(list_dicts, keys):
    return {k: float(np.mean([d[k] for d in list_dicts])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description='Evaluate FullSubNet+ on exported small testset')
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--enhanced-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    manifest_rows = load_manifest(Path(args.manifest_csv))
    enhanced_dir = Path(args.enhanced_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in manifest_rows:
        sample_id = item['sample_id']
        severity = item['severity']
        clean_path = Path(item['clean_wav'])
        noisy_path = Path(item['noisy_wav'])
        enh_path = enhanced_dir / f'{sample_id}.wav'
        if not enh_path.exists():
            continue

        clean_wav, fs1 = sf.read(str(clean_path), dtype='float32')
        noisy_wav, fs2 = sf.read(str(noisy_path), dtype='float32')
        enh_wav, fs3 = sf.read(str(enh_path), dtype='float32')
        if fs1 != 16000 or fs2 != 16000 or fs3 != 16000:
            raise RuntimeError('Expected all sample rates to be 16000')

        clean_spec = stft_ri(clean_wav)
        noisy_spec = stft_ri(noisy_wav)
        enh_spec = stft_ri(enh_wav)

        def metrics(est_wav, est_spec):
            return {
                'stoi': float(stoi(clean_wav, est_wav, 16000, extended=False)),
                'estoi': float(stoi(clean_wav, est_wav, 16000, extended=True)),
                'si_sdr': float(si_sdr(clean_wav, est_wav)),
                'lsd': lsd(clean_spec, est_spec, hf_only=False),
                'hf_lsd': lsd(clean_spec, est_spec, hf_only=True),
            }

        degraded_m = metrics(noisy_wav, noisy_spec)
        fullsubnet_m = metrics(enh_wav, enh_spec)

        rows.append({
            'sample_id': sample_id,
            'severity': severity,
            'degraded': degraded_m,
            'fullsubnet_plus': fullsubnet_m,
        })

    methods = ['degraded', 'fullsubnet_plus']
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

    print(f'[DONE] FullSubNet+ smallset evaluation saved to {out_dir}')


if __name__ == '__main__':
    main()
