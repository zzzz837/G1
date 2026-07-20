"""
Dataset for shard-based GTCRN degradation feature cache.
"""
import json
from pathlib import Path
import torch


class DegradationFeatureShardDataset(torch.utils.data.Dataset):
    def __init__(self, index_path: str, cache_dir: str, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.index_path = Path(index_path)
        self.split = split

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache dir not found: {self.cache_dir}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.entries = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("split") == split:
                    self.entries.append(e)

        if not self.entries:
            raise ValueError(f"No entries for split={split}")

        self._shard_cache = {}
        self._stats_dim = self._load_sample(0)["stats"].shape[0]

    def _load_shard(self, shard_file: str):
        if shard_file not in self._shard_cache:
            self._shard_cache[shard_file] = torch.load(self.cache_dir / shard_file, map_location="cpu", weights_only=False)
        return self._shard_cache[shard_file]

    def _load_sample(self, idx: int):
        e = self.entries[idx]
        shard = self._load_shard(e["shard_file"])
        off = e["offset"]
        return {
            "stats": shard["stats"][off],
            "noise_target": shard["noise_target"][off],
            "snr_target": shard["snr_target"][off],
            "snr_valid": shard["snr_valid"][off],
            "bandwidth_target": shard["bandwidth_target"][off],
            "bit_target": shard["bit_target"][off],
            "severity": shard["severity"][off],
        }

    @property
    def stats_dim(self):
        return self._stats_dim

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        s = self._load_sample(idx)
        return {
            "stats": s["stats"].to(torch.float32),
            "noise_target": s["noise_target"].to(torch.float32).view(-1),
            "snr_target": s["snr_target"].to(torch.float32).view(-1),
            "snr_valid": s["snr_valid"].to(torch.bool).view(-1),
            "bandwidth_target": s["bandwidth_target"].to(torch.long).view(()),
            "bit_target": s["bit_target"].to(torch.long).view(()),
            "severity": s["severity"],
        }
