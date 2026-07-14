"""
Manifest builder for GTCRN work2 data pipeline.

Recursively finds wav files, records metadata, and splits into train/valid/test.
Outputs JSONL manifest and summary JSON.
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf
import numpy as np

from work2.data.random_utils import seed_everything


def scan_wavs(directory: Path) -> list[dict]:
    """Recursively find wav files and extract metadata."""
    entries = []
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for wav_path in sorted(directory.rglob("*.wav")):
        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            print(f"[WARN] Skipping {wav_path}: {e}", file=sys.stderr)
            continue

        try:
            rel = str(wav_path.relative_to(Path.cwd()))
        except ValueError:
            rel = str(wav_path)
        entries.append({
            "path": str(wav_path),
            "relative_path": rel,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": round(info.duration, 4),
        })

    return entries


def split_entries(entries: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries into 80% train, 10% valid, 10% test by file hash."""
    seed_everything(seed)

    hashes = []
    for entry in entries:
        h = hashlib.sha256(entry["path"].encode()).hexdigest()
        hashes.append(int(h, 16))

    n = len(entries)
    indices = list(range(n))
    indices.sort(key=lambda i: hashes[i])

    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    train = [entries[i] for i in train_indices]
    valid = [entries[i] for i in valid_indices]
    test = [entries[i] for i in test_indices]

    return train, valid, test


def write_jsonl(entries: list[dict], split: str, output_path: Path):
    """Write entries as JSONL with split tag."""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            entry["split"] = split
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build data manifest")
    parser.add_argument("--clean-dir", type=str, default=None, help="Path to clean speech directory")
    parser.add_argument("--noise-dir", type=str, default=None, help="Path to noise directory")
    parser.add_argument("--output", type=str, default="outputs/manifest.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    all_entries = []

    # Scan clean directory
    if args.clean_dir:
        clean_path = Path(args.clean_dir)
        print(f"[INFO] Scanning clean directory: {clean_path}")
        clean_entries = scan_wavs(clean_path)
        for e in clean_entries:
            e["type"] = "clean"
        all_entries.extend(clean_entries)
        print(f"[INFO] Found {len(clean_entries)} clean wav files")

    # Scan noise directory
    if args.noise_dir:
        noise_path = Path(args.noise_dir)
        print(f"[INFO] Scanning noise directory: {noise_path}")
        noise_entries = scan_wavs(noise_path)
        for e in noise_entries:
            e["type"] = "noise"
        all_entries.extend(noise_entries)
        print(f"[INFO] Found {len(noise_entries)} noise wav files")

    if not all_entries:
        print("[WARN] No wav files found. Creating empty manifest.")

    # Split
    train, valid, test = split_entries(all_entries, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in train:
            entry["split"] = "train"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for entry in valid:
            entry["split"] = "valid"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for entry in test:
            entry["split"] = "test"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "total_files": len(all_entries),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "seed": args.seed,
        "clean_dir": args.clean_dir,
        "noise_dir": args.noise_dir,
    }

    if all_entries:
        durations = [e["duration"] for e in all_entries]
        summary["total_duration_seconds"] = round(float(np.sum(durations)), 2)
        summary["mean_duration_seconds"] = round(float(np.mean(durations)), 4)

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Manifest saved: {output_path}")
    print(f"[INFO] Summary saved: {summary_path}")
    print(f"[INFO] Split: train={len(train)}, valid={len(valid)}, test={len(test)}")

    if not all_entries:
        print("[WARN] Empty manifest — valid for testing tool without real dataset.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
