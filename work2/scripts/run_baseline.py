"""
Baseline measurement script for GTCRN (Phase 2 compatibility layer).
- Forces CPU execution
- Loads official GTCRN with DNS3 checkpoint
- Measures parameter count, inference time, RTF
- Saves enhanced audio and metrics JSON
- Uses stft_utils for PyTorch 2.x compatibility (no global monkey-patch)
"""
import os
import sys
import time
import json
import hashlib
import platform
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from work2.data.stft_utils import stft_to_ri, ri_to_istft

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_git_commit():
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    ckpt_path = REPO_ROOT / "checkpoints" / "model_trained_on_dns3.tar"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    audio_path = REPO_ROOT / "test_wavs" / "mix.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    mix, fs = sf.read(str(audio_path), dtype="float32")
    if fs != 16000:
        raise ValueError(f"Sample rate must be 16000 Hz, got {fs} Hz")
    if mix.ndim > 1:
        raise ValueError(f"Audio must be mono, got shape {mix.shape}")

    duration = len(mix) / fs
    print(f"[INFO] Input: {len(mix)} samples, {fs} Hz, {duration:.3f} s")

    from gtcrn import GTCRN
    model = GTCRN().to(device).eval()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])

    total_params, trainable_params = count_params(model)
    print(f"[INFO] Parameters: {total_params} total, {trainable_params} trainable")

    window = torch.hann_window(512).pow(0.5)
    spec = stft_to_ri(
        torch.from_numpy(mix),
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=window,
        center=True,
    )

    for _ in range(3):
        with torch.no_grad():
            _ = model(spec.unsqueeze(0))

    n_runs = 10
    times = []
    for _ in range(n_runs):
        torch.manual_seed(SEED)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(spec.unsqueeze(0))[0]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = float(np.mean(times))
    rtf = mean_time / duration
    print(f"[INFO] Mean inference time: {mean_time:.4f} s (n={n_runs})")
    print(f"[INFO] RTF: {rtf:.4f}")

    enh_wav = ri_to_istft(
        output,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=window,
        center=True,
    )
    enh_np = enh_wav.detach().cpu().numpy()

    if np.any(np.isnan(enh_np)):
        raise RuntimeError("Output contains NaN values")

    out_dir = REPO_ROOT / "outputs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "enh_dns3.wav"
    sf.write(str(out_path), enh_np, fs)
    print(f"[INFO] Saved: {out_path}")

    metrics = {
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "sample_rate": int(fs),
        "input_samples": int(len(mix)),
        "input_duration_seconds": round(duration, 4),
        "output_samples": int(len(enh_np)),
        "parameter_count": int(total_params),
        "trainable_parameter_count": int(trainable_params),
        "mean_inference_seconds": round(mean_time, 6),
        "rtf": round(rtf, 6),
        "checkpoint": str(ckpt_path.name),
        "checkpoint_sha256": compute_sha256(ckpt_path),
        "n_runs": n_runs,
        "seed": SEED,
    }

    json_path = out_dir / "baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved: {json_path}")

    print("[DONE] Baseline measurement complete.")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
