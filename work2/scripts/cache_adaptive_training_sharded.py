"""
Shard-based cache for adaptive residual training.

Unlike estimator cache, this stores the tensors needed for residual training:
- clean_spec
- degraded_spec
- enhanced_base (from frozen GTCRN)
- degradation condition labels

This avoids re-running GTCRN during adaptive residual training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import ensure_mono, match_length
from work2.data.stft_utils import stft_to_ri
from work2.data.degradation import apply_composite_degradation, sample_degradation_config
from work2.data.label_utils import make_degradation_labels
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor

SEVERITIES = ["clean", "light", "medium", "heavy"]


def _get_streaming_noise(n_samples: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(n_samples, generator=g, dtype=torch.float32) * 0.1
    return noise - noise.mean()


def _assign_split(src_idx: int, seed: int) -> str:
    import hashlib
    h = hashlib.sha256(f"split_{seed}_{src_idx}".encode()).hexdigest()
    bucket = int(h, 16) % 10
    if bucket < 8:
        return "train"
    elif bucket < 9:
        return "valid"
    return "test"


def _load_existing_index(index_path: Path):
    existing = set()
    index_entries = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                existing.add((e["source_path"], e["severity"], e["seed"]))
                index_entries.append(e)
    return existing, index_entries


def _flush_shard(shard_buffer: list[dict], out_dir: Path, shard_id: int) -> tuple[str, int]:
    if not shard_buffer:
        return "", 0
    shard_file = f"adaptive_shard_{shard_id:05d}.pt"
    shard_path = out_dir / shard_file

    payload = {
        "clean_spec": torch.stack([x["clean_spec"] for x in shard_buffer], dim=0),
        "degraded_spec": torch.stack([x["degraded_spec"] for x in shard_buffer], dim=0),
        "enhanced_base": torch.stack([x["enhanced_base"] for x in shard_buffer], dim=0),
        "noise_target": torch.tensor([x["noise_target"] for x in shard_buffer], dtype=torch.float32).view(-1, 1),
        "snr_target": torch.tensor([x["snr_target"] for x in shard_buffer], dtype=torch.float32).view(-1, 1),
        "snr_valid": torch.tensor([x["snr_valid"] for x in shard_buffer], dtype=torch.bool).view(-1, 1),
        "bandwidth_target": torch.tensor([x["bandwidth_target"] for x in shard_buffer], dtype=torch.long),
        "bit_target": torch.tensor([x["bit_target"] for x in shard_buffer], dtype=torch.long),
        "severity": [x["severity"] for x in shard_buffer],
        "source_path": [x["source_path"] for x in shard_buffer],
        "seed": [x["seed"] for x in shard_buffer],
        "split": [x["split"] for x in shard_buffer],
        "bandwidth_limited": torch.tensor([x["bandwidth_limited"] for x in shard_buffer], dtype=torch.bool),
    }
    torch.save(payload, shard_path)
    return shard_file, len(shard_buffer)


def main():
    parser = argparse.ArgumentParser(description="Cache adaptive residual training tensors into shard files")
    parser.add_argument("--clean-manifest", type=str, default="outputs/manifest_vctk.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--output-dir", type=str, default="outputs/adaptive_cache_sharded")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only-split", type=str, default=None, choices=["train", "valid", "test"],
                        help="Only cache samples whose manifest split matches this value")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    extractor = FrozenGTCRNFeatureExtractor(checkpoint_path=args.checkpoint, device="cpu")

    manifest_path = Path(args.clean_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    existing, index_entries = _load_existing_index(index_path)
    print(f"[INFO] Resuming: {len(existing)} existing cached samples found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_lines = [json.loads(line) for line in f if line.strip()]

    shard_buffer = []
    shard_id = len(list(out_dir.glob("adaptive_shard_*.pt")))
    total_new = 0
    forward_times = []

    for src_idx, entry in enumerate(manifest_lines):
        audio_path = Path(entry.get("path", entry.get("relative_path", "")))
        if not audio_path.exists():
            continue
        try:
            wav, in_fs = sf.read(str(audio_path), dtype="float32")
        except Exception as e:
            print(f"[WARN] Skipping {audio_path}: {e}")
            continue
        if in_fs != 16000:
            continue

        clean_wav = ensure_mono(torch.from_numpy(wav))
        target_len = int(args.segment_seconds * in_fs)
        clean_wav = match_length(clean_wav, target_len)
        split = _assign_split(src_idx, args.seed)
        if args.only_split is not None and split != args.only_split:
            continue
 
        for sev_idx, sev in enumerate(SEVERITIES):
            local_seed = args.seed * 10000 + src_idx * 100 + sev_idx
            key = (str(audio_path), sev, local_seed)
            if key in existing:
                continue

            cfg = sample_degradation_config(sev, local_seed)
            noise_wav = _get_streaming_noise(clean_wav.numel(), local_seed) if cfg.add_noise else None
            result = apply_composite_degradation(clean_wav.clone(), cfg, noise_wav.clone() if noise_wav is not None else None)
            labels = make_degradation_labels(result)

            window = torch.hann_window(512).pow(0.5)
            clean_spec = stft_to_ri(result.clean, n_fft=512, hop_length=256, win_length=512, window=window)
            degraded_spec = stft_to_ri(result.degraded, n_fft=512, hop_length=256, win_length=512, window=window)

            t0 = time.perf_counter()
            feat_out = extractor(degraded_spec.unsqueeze(0))
            forward_times.append(time.perf_counter() - t0)

            shard_buffer.append({
                "clean_spec": clean_spec.cpu(),
                "degraded_spec": degraded_spec.cpu(),
                "enhanced_base": feat_out["enhanced_spec"].squeeze(0).cpu(),
                "noise_target": labels["noise_present"],
                "snr_target": labels["snr_target"],
                "snr_valid": labels["snr_valid"],
                "bandwidth_target": labels["bandwidth_class"],
                "bit_target": labels["bit_class"],
                "severity": sev,
                "source_path": str(audio_path),
                "seed": local_seed,
                "split": split,
                "bandwidth_limited": (cfg.intermediate_rate is not None or cfg.cutoff_hz is not None),
            })
            existing.add(key)
            total_new += 1

            if len(shard_buffer) >= args.shard_size:
                shard_file, shard_count = _flush_shard(shard_buffer, out_dir, shard_id)
                for i in range(shard_count):
                    item = shard_buffer[i]
                    index_entries.append({
                        "shard_file": shard_file,
                        "offset": i,
                        "split": item["split"],
                        "source_path": item["source_path"],
                        "severity": item["severity"],
                        "seed": item["seed"],
                    })
                shard_buffer = []
                shard_id += 1
                with open(index_path, "w", encoding="utf-8") as f:
                    for e in index_entries:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"[INFO] Flushed adaptive shard {shard_id-1:05d}, total cached={len(index_entries)}")

        if src_idx % 500 == 0:
            print(f"[INFO] Progress: {src_idx}/{len(manifest_lines)} sources, {len(existing)} cached samples")

    if shard_buffer:
        shard_file, shard_count = _flush_shard(shard_buffer, out_dir, shard_id)
        for i in range(shard_count):
            item = shard_buffer[i]
            index_entries.append({
                "shard_file": shard_file,
                "offset": i,
                "split": item["split"],
                "source_path": item["source_path"],
                "severity": item["severity"],
                "seed": item["seed"],
            })
        with open(index_path, "w", encoding="utf-8") as f:
            for e in index_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[INFO] Flushed final adaptive shard {shard_id:05d}, total cached={len(index_entries)}")

    splits = {"train": 0, "valid": 0, "test": 0}
    sev_counts = {s: 0 for s in SEVERITIES}
    for e in index_entries:
        splits[e["split"]] += 1
        sev_counts[e["severity"]] += 1

    cache_size_mb = sum(f.stat().st_size for f in out_dir.glob("adaptive_shard_*.pt")) / (1024 * 1024)
    summary = {
        "total_samples": len(index_entries),
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["test"],
        "severity_counts": sev_counts,
        "cache_size_mb": round(cache_size_mb, 2),
        "mean_gtcrn_forward_time_s": round(float(np.mean(forward_times)), 4) if forward_times else 0.0,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "n_shards": len(list(out_dir.glob('adaptive_shard_*.pt'))),
    }
    with open(out_dir / "cache_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Cached {len(index_entries)} adaptive samples")
    print(f"[INFO] Split: train={splits['train']}, valid={splits['valid']}, test={splits['test']}")
    print(f"[INFO] Cache size: {cache_size_mb:.1f} MB")
    print(f"[INFO] Summary saved: {out_dir / 'cache_summary.json'}")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
