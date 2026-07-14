"""
Generate degradation demonstration: clean, light, medium, heavy waveforms,
spectrograms, waveform plots, and metadata JSON.
"""
import argparse
import json
import sys
import os
from pathlib import Path

import torch
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import (
    ensure_mono,
    peak_normalize,
    match_length,
    rms,
    check_waveform,
)
from work2.data.degradation import (
    apply_composite_degradation,
    sample_degradation_config,
)
import soundfile as sf


OUTPUT_DIR = Path("outputs/degradation_demo")


def make_test_signal(duration: float = 3.0, sample_rate: int = 16000, seed: int = 2026) -> torch.Tensor:
    """Synthesize a multi-tone test signal with smooth envelope."""
    seed_everything(seed)
    freqs = [200, 500, 1000, 3000, 6000]
    t = torch.arange(0, int(sample_rate * duration), dtype=torch.float32) / sample_rate
    signal = torch.zeros_like(t)

    for freq in freqs:
        signal += torch.sin(2.0 * np.pi * freq * t)

    signal = signal / len(freqs)

    fade_len = int(0.05 * sample_rate)
    env = torch.ones_like(signal)
    env[:fade_len] = torch.linspace(0, 1, fade_len)
    env[-fade_len:] = torch.linspace(1, 0, fade_len)
    signal = signal * env

    return peak_normalize(signal, peak=0.9)


def make_fixed_noise(length: int, seed: int = 42) -> torch.Tensor:
    """Generate reproducible noise."""
    seed_everything(seed)
    noise = torch.randn(length, dtype=torch.float32) * 0.1
    noise = noise - noise.mean()
    return noise


def plot_spectrogram(ax, waveform: torch.Tensor, title: str, sample_rate: int = 16000):
    """Plot a spectrogram on the given axis."""
    n_fft = 512
    hop_length = 128

    x = waveform.detach().cpu().numpy()
    f, t_seg, Sxx = scipy.signal.spectrogram(
        x, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length,
        window="hann", mode="magnitude"
    )

    db = 20.0 * np.log10(Sxx + 1e-12)
    vmin = -80
    vmax = np.percentile(db.flatten(), 99)

    ax.pcolormesh(t_seg, f, db, shading="gouraud", vmin=vmin, vmax=vmax, cmap="inferno")
    ax.set_ylim(0, 8000)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")


def plot_waveform(ax, waveform: torch.Tensor, title: str, sample_rate: int = 16000):
    """Plot waveform on the given axis."""
    x = waveform.detach().cpu().numpy()
    t = np.arange(len(x)) / sample_rate
    ax.plot(t, x, linewidth=0.5, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description="Generate degradation demo")
    parser.add_argument("--input", type=str, default=None, help="Path to clean wav file")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    # Load or synthesize clean signal
    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input not found: {args.input}")
        clean, in_fs = sf.read(args.input, dtype="float32")
        if in_fs != 16000:
            raise ValueError(f"Sample rate must be 16000 Hz, got {in_fs}")
        clean = ensure_mono(torch.from_numpy(clean))
    else:
        clean = make_test_signal(duration=3.0, sample_rate=16000, seed=args.seed)

    check_waveform(clean)
    print(f"[INFO] Clean signal: {clean.numel()} samples, {clean.numel()/16000:.2f} s")

    # Generate noise
    noise = make_fixed_noise(clean.numel(), seed=args.seed)

    # Process each severity
    severities = ["clean", "light", "medium", "heavy"]
    results = {}
    metadata = []

    for sev in severities:
        cfg = sample_degradation_config(sev, seed=args.seed)
        result = apply_composite_degradation(clean.clone(), cfg, noise.clone())
        results[sev] = result

        wav_path = OUTPUT_DIR / f"{sev}.wav"
        sf.write(str(wav_path), result.degraded.detach().cpu().numpy(), 16000)
        print(f"[INFO] Saved: {wav_path}")

        meta = result.to_dict()
        try:
            meta["file"] = str(wav_path.relative_to(Path.cwd()))
        except ValueError:
            meta["file"] = str(wav_path)
        meta["duration"] = round(result.degraded.numel() / 16000, 4)
        metadata.append(meta)

    # Save metadata JSON
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved: {meta_path}")

    # Spectrogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, sev in zip(axes.flatten(), severities):
        plot_spectrogram(ax, results[sev].degraded, sev.capitalize())
    fig.suptitle("Spectrogram Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    spec_path = OUTPUT_DIR / "spectrogram_comparison.png"
    fig.savefig(spec_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved: {spec_path}")

    # Waveform comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, sev in zip(axes.flatten(), severities):
        plot_waveform(ax, results[sev].degraded, sev.capitalize())
    fig.suptitle("Waveform Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    wavfig_path = OUTPUT_DIR / "waveform_comparison.png"
    fig.savefig(wavfig_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved: {wavfig_path}")

    print("[DONE] Degradation demo generation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
