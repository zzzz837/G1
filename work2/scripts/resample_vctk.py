"""
Resample VCTK audio from 48kHz (or other rates) to 16kHz for GTCRN.
Uses scipy.signal.resample_poly for high-quality downsampling.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal


def resample_wav(input_path: Path, output_path: Path, target_rate: int = 16000):
    data, orig_rate = sf.read(str(input_path), dtype="float32")
    print(f"  {input_path.name:20s}  {orig_rate}Hz → {target_rate}Hz  ({len(data)} → ", end="")

    if orig_rate == target_rate:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), data, target_rate)
        print(f"{len(data)} samples, passthrough)")
        return

    gcd_val = int(np.gcd(int(orig_rate), target_rate))
    up = target_rate // gcd_val
    down = int(orig_rate) // gcd_val

    resampled = scipy.signal.resample_poly(data.astype("float64"), up, down).astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), resampled, target_rate)
    print(f"{len(resampled)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Resample VCTK to 16kHz")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory (recursive)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target-rate", type=int, default=16000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    wav_files = list(input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    print(f"[INFO] Found {len(wav_files)} wav files")
    print(f"[INFO] Resampling to {args.target_rate} Hz...")

    for wav_path in sorted(wav_files):
        rel = wav_path.relative_to(input_dir)
        out_path = output_dir / rel
        resample_wav(wav_path, out_path, args.target_rate)

    print(f"[DONE] Resampled {len(wav_files)} files → {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
