"""
Cache GTCRN degradation features from clean audio manifest.

Extracts bottleneck statistics via FrozenGTCRNFeatureExtractor for each
degraded sample and saves compact label+stats files.
"""
import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import ensure_mono, peak_normalize, match_length, check_waveform
from work2.data.stft_utils import stft_to_ri
from work2.data.degradation import (
    apply_composite_degradation,
    sample_degradation_config,
    CompositeDegradationConfig,
)
from work2.data.label_utils import make_degradation_labels
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor


def collate_batch(batch_stats: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch_stats, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Cache GTCRN degradation features")
    parser.add_argument("--clean-manifest", type=str, default="outputs/manifest.jsonl")
    parser.add_argument("--noise-manifest", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--output-dir", type=str, default="outputs/estimator_cache")
    parser.add_argument("--samples-per-file", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading GTCRN feature extractor from {args.checkpoint}")
    t_load = time.perf_counter()
    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.checkpoint, device="cpu")
    stats_dim = extractor.get_stats_dim()
    t_load = time.perf_counter() - t_load
    print(f"[INFO] Feature extractor loaded in {t_load:.1f}s. Stats dim = {stats_dim}")

    # Determine mode: real manifest, single wav, or synthetic
    manifest_path = Path(args.clean_manifest)
    use_synthetic = not manifest_path.exists()

    if use_synthetic:
        print("[WARN] No clean manifest found. Generating synthetic pseudo-speech.")
        clean_sources = _generate_synthetic_sources(
            num_sources=32, duration=args.segment_seconds, sample_rate=16000, seed=args.seed
        )
    else:
        clean_sources = _load_manifest_sources(manifest_path, args.segment_seconds, 16000, args.seed)

    noise_sources = _load_or_generate_noise(args.noise_manifest, 16000, args.seed, clean_sources)

    severities = ["clean", "light", "medium", "heavy"]

    total_samples = 0
    index_entries = []
    extraction_times = []

    for src_idx, (src_path, clean_wav) in enumerate(clean_sources):
        split = _assign_split(src_idx, len(clean_sources), args.seed)

        for sev_idx, sev in enumerate(severities):
            local_seed = args.seed * 10000 + src_idx * 100 + sev_idx
            seed_everything(local_seed)

            cfg = sample_degradation_config(sev, local_seed)
            noise_wav = noise_sources[src_idx % len(noise_sources)] if cfg.add_noise else None

            result = apply_composite_degradation(clean_wav.clone(), cfg, noise_wav.clone() if noise_wav is not None else None)

            # STFT
            window = torch.hann_window(512).pow(0.5)
            spec = stft_to_ri(result.degraded, n_fft=512, hop_length=256, win_length=512, window=window)

            # GTCRN extraction
            t_start = time.perf_counter()
            feats = extractor(spec.unsqueeze(0))
            t_elapsed = time.perf_counter() - t_start
            extraction_times.append(t_elapsed)

            labels = make_degradation_labels(result)

            cache_entry = {
                "stats": feats["stats"].squeeze(0).cpu(),
                "noise_target": torch.tensor([labels["noise_present"]]),
                "snr_target": torch.tensor([labels["snr_target"]]),
                "snr_valid": torch.tensor([labels["snr_valid"]]),
                "bandwidth_target": torch.tensor([labels["bandwidth_class"]]),
                "bit_target": torch.tensor([labels["bit_class"]]),
                "severity": sev,
            }

            cache_filename = f"sample_{total_samples:06d}.pt"
            cache_filepath = out_dir / cache_filename
            torch.save(cache_entry, cache_filepath)

            index_entries.append({
                "cache_file": cache_filename,
                "split": split,
                "source_path": str(src_path),
                "severity": sev,
                "seed": local_seed,
                "stats_dim": stats_dim,
            })

            total_samples += 1

    # Save index
    index_path = out_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    splits = {"train": 0, "valid": 0, "test": 0}
    sev_counts = {s: 0 for s in severities}
    for e in index_entries:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
        sev_counts[e["severity"]] = sev_counts.get(e["severity"], 0) + 1

    cache_size_mb = sum(
        f.stat().st_size for f in out_dir.glob("*.pt")
    ) / (1024 * 1024)

    summary = {
        "total_samples": total_samples,
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["test"],
        "severity_counts": sev_counts,
        "stats_dim": stats_dim,
        "cache_size_mb": round(cache_size_mb, 2),
        "mean_extraction_time_s": round(float(np.mean(extraction_times)), 4),
        "seed": args.seed,
        "synthetic": use_synthetic,
    }

    summary_path = out_dir / "cache_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Cached {total_samples} samples")
    print(f"[INFO] Split: train={splits['train']}, valid={splits['valid']}, test={splits['test']}")
    print(f"[INFO] Stats dim: {stats_dim}")
    print(f"[INFO] Cache size: {cache_size_mb:.1f} MB")
    print(f"[INFO] Mean extraction time: {summary['mean_extraction_time_s']:.4f}s")
    print(f"[INFO] Summary saved: {summary_path}")
    print("[DONE]")


def _load_manifest_sources(manifest_path: Path, seg_dur: float, fs: int, seed: int):
    sources = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            audio_path = Path(e.get("path", e.get("relative_path", "")))
            if not audio_path.exists():
                continue
            wav, in_fs = sf.read(str(audio_path), dtype="float32")
            wav = ensure_mono(torch.from_numpy(wav))
            if in_fs != fs:
                raise ValueError(f"Sample rate mismatch: {in_fs} != {fs}")
            wav = match_length(wav, int(seg_dur * fs) * 4)
            sources.append((str(audio_path), wav))
    return sources


def _generate_synthetic_sources(num_sources: int, duration: float, sample_rate: int, seed: int):
    import random
    rng = random.Random(seed)
    sources = []
    for i in range(num_sources):
        n_samples = int(duration * sample_rate)
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate

        base_freq = rng.uniform(80, 400)
        n_harmonics = rng.randint(3, 8)
        sig = torch.zeros(n_samples)
        for h in range(1, n_harmonics + 1):
            amp = rng.uniform(0.3, 1.0) / h
            phase = rng.uniform(0, 2 * np.pi)
            freq_mod = 1.0 + 0.01 * torch.sin(2 * np.pi * rng.uniform(2, 8) * t)
            sig += amp * torch.sin(2 * np.pi * base_freq * h * t * freq_mod + phase)

        n_seg = rng.randint(2, 5)
        seg_len = n_samples // n_seg
        env = torch.ones(n_samples)
        for s in range(n_seg):
            start = s * seg_len
            end = min((s + 1) * seg_len, n_samples)
            if rng.random() < 0.3:
                env[start:end] = rng.uniform(0.0, 0.1)
            else:
                fade_in = min(400, (end - start) // 2)
                env[start:start + fade_in] = torch.linspace(0, 1, fade_in)
                env[end - fade_in:end] = torch.linspace(1, 0, fade_in)

        sig = sig * env
        sig = peak_normalize(sig, peak=0.9)
        sources.append((f"synth_{i:04d}", sig))

    return sources


def _load_or_generate_noise(noise_manifest: Optional[str], fs: int, seed: int, clean_sources):
    if noise_manifest and Path(noise_manifest).exists():
        noise_list = []
        with open(noise_manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                p = Path(e.get("path", ""))
                if p.exists():
                    n, _ = sf.read(str(p), dtype="float32")
                    noise_list.append(torch.from_numpy(n))
        return noise_list

    seed_everything(seed)
    noises = []
    for src in clean_sources:
        n = torch.randn(src[1].numel(), dtype=torch.float32) * 0.1
        n = n - n.mean()
        noises.append(n)
    return noises


def _assign_split(src_idx: int, total_sources: int, seed: int) -> str:
    import hashlib
    h = hashlib.sha256(f"split_{seed}_{src_idx}".encode()).hexdigest()
    bucket = int(h, 16) % 10
    if bucket < 8:
        return "train"
    elif bucket < 9:
        return "valid"
    else:
        return "test"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
