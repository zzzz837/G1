"""
PyTorch Dataset for GTCRN degradation features (pre-cached).
"""
import json
import torch
from pathlib import Path
from typing import Optional


class DegradationFeatureDataset(torch.utils.data.Dataset):
    """
    Loads pre-cached degradation features and labels from index file.

    Each cache entry is a .pt dict with keys:
        stats: Tensor(D,)
        noise_target: Tensor(1,)
        snr_target: Tensor(1,)
        snr_valid: Tensor(1,)
        bandwidth_target: Tensor(,)
        bit_target: Tensor(,)
        severity: str
    """

    def __init__(
        self,
        index_path: str,
        cache_dir: str,
        split: str = "train",
    ) -> None:
        super().__init__()

        self.cache_dir = Path(cache_dir)
        self.split = split

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.entries = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("split") == split:
                    self.entries.append(entry)

        if len(self.entries) == 0:
            raise ValueError(f"No entries found for split='{split}' in {index_path}")

        self._check_first_entry()

    def _check_first_entry(self):
        """Verify the first entry loads correctly and check dimensions."""
        entry = self.entries[0]
        filepath = self.cache_dir / entry["cache_file"]
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")

        sample = torch.load(filepath, map_location="cpu", weights_only=False)

        required_keys = ["stats", "noise_target", "snr_target", "snr_valid",
                         "bandwidth_target", "bit_target"]
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"Missing key '{key}' in cached sample {filepath}")

        if sample["stats"].isnan().any() or sample["stats"].isinf().any():
            raise ValueError(f"NaN/Inf in stats of {filepath}")

        self._stats_dim = sample["stats"].shape[0]

    @property
    def stats_dim(self) -> int:
        return self._stats_dim

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        filepath = self.cache_dir / entry["cache_file"]

        sample = torch.load(filepath, map_location="cpu", weights_only=False)

        return {
            "stats": sample["stats"].to(torch.float32),
            "noise_target": sample["noise_target"].to(torch.float32).squeeze(),
            "snr_target": sample["snr_target"].to(torch.float32).squeeze(),
            "snr_valid": sample["snr_valid"].to(torch.bool).squeeze(),
            "bandwidth_target": sample["bandwidth_target"].to(torch.long).squeeze(),
            "bit_target": sample["bit_target"].to(torch.long).squeeze(),
            "severity": sample.get("severity", ""),
        }
