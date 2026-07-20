"""
Shard dataset for adaptive residual training.
"""
import json
from pathlib import Path
import torch


class AdaptiveResidualShardDataset(torch.utils.data.Dataset):
    def __init__(self, index_path: str, cache_dir: str, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.index_path = Path(index_path)
        self.split = split
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
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
        self._cache = {}

    def _load_shard(self, shard_file: str):
        if shard_file not in self._cache:
            self._cache[shard_file] = torch.load(self.cache_dir / shard_file, map_location="cpu", weights_only=False)
        return self._cache[shard_file]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        shard = self._load_shard(e["shard_file"])
        off = e["offset"]
        return {
            "clean_spec": shard["clean_spec"][off].to(torch.float32),
            "degraded_spec": shard["degraded_spec"][off].to(torch.float32),
            "enhanced_base": shard["enhanced_base"][off].to(torch.float32),
            "noise_target": shard["noise_target"][off].to(torch.float32).view(-1),
            "snr_target": shard["snr_target"][off].to(torch.float32).view(-1),
            "snr_valid": shard["snr_valid"][off].to(torch.bool).view(-1),
            "bandwidth_target": shard["bandwidth_target"][off].to(torch.long).view(()),
            "bit_target": shard["bit_target"][off].to(torch.long).view(()),
            "bandwidth_limited": shard["bandwidth_limited"][off].to(torch.bool),
            "severity": shard["severity"][off],
        }
