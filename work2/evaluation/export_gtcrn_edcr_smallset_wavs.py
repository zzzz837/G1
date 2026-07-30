import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch

from work2.data.stft_utils import stft_to_ri, ri_to_istft
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.models.adaptive_residual import AdaptiveResidualModule


def load_manifest(csv_path: Path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Export GTCRN and EDCR enhanced wavs on the same small external-model testset')
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--gtcrn-checkpoint', required=True)
    parser.add_argument('--estimator-checkpoint', required=True)
    parser.add_argument('--v1-checkpoint', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--residual-scale', type=float, default=0.75)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    manifest_rows = load_manifest(Path(args.manifest_csv))
    output_root = Path(args.output_root)
    gtcrn_dir = output_root / 'gtcrn'
    edcr_dir = output_root / 'edcr'
    gtcrn_dir.mkdir(parents=True, exist_ok=True)
    edcr_dir.mkdir(parents=True, exist_ok=True)

    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.gtcrn_checkpoint, device='cpu')
    est_ckpt = torch.load(args.estimator_checkpoint, map_location='cpu')
    estimator = DegradationEstimator(input_dim=est_ckpt['input_dim'], hidden_dim=est_ckpt.get('hidden_dim', 32)).to(device)
    estimator.load_state_dict(est_ckpt['model_state_dict'])
    estimator.eval()

    v1_ckpt = torch.load(args.v1_checkpoint, map_location='cpu')
    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    v1.load_state_dict(v1_ckpt['residual_state_dict'])
    v1.eval()

    window = torch.hann_window(512).pow(0.5)

    for item in manifest_rows:
        sample_id = item['sample_id']
        noisy_path = Path(item['noisy_wav'])
        if not noisy_path.exists():
            continue

        noisy_wav, fs = sf.read(str(noisy_path), dtype='float32')
        if fs != 16000:
            raise RuntimeError(f'Expected 16000 Hz, got {fs} for {noisy_path}')

        noisy_spec = stft_to_ri(torch.from_numpy(noisy_wav), window=window)
        with torch.inference_mode():
            feats = extractor(noisy_spec.unsqueeze(0))
            base_spec_t = feats['enhanced_spec'][0].cpu()
            preds = estimator(feats['stats'])
            noise_p = torch.sigmoid(preds['noise_logit'])
            bw_p = torch.softmax(preds['bandwidth_logits'], dim=-1)
            bit_p = torch.softmax(preds['bit_logits'], dim=-1)
            cond = torch.cat([noise_p, preds['snr_pred'], bw_p, bit_p], dim=-1)
            base_perm = base_spec_t.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
            out = v1(base_perm, cond)
            edcr_spec_t = (base_perm + args.residual_scale * out['residual']).permute(0, 3, 2, 1)[0].cpu()

        gtcrn_wav = ri_to_istft(base_spec_t, window=window, length=len(noisy_wav)).numpy()
        edcr_wav = ri_to_istft(edcr_spec_t, window=window, length=len(noisy_wav)).numpy()

        sf.write(str(gtcrn_dir / f'{sample_id}.wav'), gtcrn_wav, 16000)
        sf.write(str(edcr_dir / f'{sample_id}.wav'), edcr_wav, 16000)

    print(f'[DONE] Exported GTCRN wavs to {gtcrn_dir}')
    print(f'[DONE] Exported EDCR wavs to {edcr_dir}')


if __name__ == '__main__':
    main()
