"""
Verify that the official infer.py output matches the new baseline script output.
"""
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TOLERANCE = 1e-4


def main():
    official_path = REPO_ROOT / "test_wavs" / "enh.wav"
    baseline_path = REPO_ROOT / "outputs" / "baseline" / "enh_dns3.wav"

    for p in [official_path, baseline_path]:
        if not p.exists():
            print(f"[FAIL] File not found: {p}")
            sys.exit(1)

    official, fs1 = sf.read(str(official_path), dtype="float32")
    baseline, fs2 = sf.read(str(baseline_path), dtype="float32")

    if fs1 != fs2:
        print(f"[FAIL] Sample rate mismatch: {fs1} vs {fs2}")
        sys.exit(1)

    if official.shape != baseline.shape:
        print(f"[FAIL] Length mismatch: {official.shape} vs {baseline.shape}")
        sys.exit(1)

    diff = np.abs(official - baseline)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    print(f"[INFO] Sample rate: {fs1} Hz")
    print(f"[INFO] Length: {len(official)} samples")
    print(f"[INFO] Max  absolute error: {max_err:.6e}")
    print(f"[INFO] Mean absolute error: {mean_err:.6e}")

    if max_err > TOLERANCE:
        print(f"[FAIL] Max error {max_err:.6e} exceeds tolerance {TOLERANCE}")
        sys.exit(1)

    print("[PASS] Official and baseline outputs match within tolerance.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
