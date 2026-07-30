"""Bounded-LRU shard dataset for adaptive residual training."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class AdaptiveResidualShardDataset(torch.utils.data.Dataset):
    def __init__(self, index_path: str, cache_dir: str, split: str = "train", max_cached_shards: int = 1):
        self.cache_dir = Path(cache_dir)
        self.index_path = Path(index_path)
        self.split = split
        if max_cached_shards < 0:
            raise ValueError(f"max_cached_shards must be >= 0, got {max_cached_shards}")
        self.max_cached_shards = int(max_cached_shards)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        self.entries: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.index_path}:{line_no}") from exc
                if entry.get("split") == split:
                    if "shard_file" not in entry or "offset" not in entry:
                        raise KeyError(f"Missing shard_file/offset at {self.index_path}:{line_no}")
                    self.entries.append(entry)
        if not self.entries:
            raise ValueError(f"No entries for split={split}")

        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _load_shard(self, shard_file: str) -> dict[str, Any]:
        if shard_file in self._cache:
            shard = self._cache.pop(shard_file)
            self._cache[shard_file] = shard
            return shard

        shard_path = self.cache_dir / shard_file
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(shard, dict):
            raise TypeError(f"Shard must be a dict, got {type(shard).__name__}: {shard_path}")

        if self.max_cached_shards > 0:
            self._cache[shard_file] = shard
            while len(self._cache) > self.max_cached_shards:
                self._cache.popitem(last=False)
        return shard

    @staticmethod
    def _sample_copy(shard: dict[str, Any], key: str, offset: int, dtype: torch.dtype) -> torch.Tensor:
        if key not in shard:
            raise KeyError(f"Missing key in shard: {key}")
        value = shard[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Shard field {key!r} must be a tensor, got {type(value).__name__}")
        if offset < 0 or offset >= value.shape[0]:
            raise IndexError(f"Offset {offset} out of range for {key}, first dimension={value.shape[0]}")
        return value[offset].to(dtype=dtype).clone()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        shard = self._load_shard(str(entry["shard_file"]))
        off = int(entry["offset"])
        return {
            "clean_spec": self._sample_copy(shard, "clean_spec", off, torch.float32),
            "degraded_spec": self._sample_copy(shard, "degraded_spec", off, torch.float32),
            "enhanced_base": self._sample_copy(shard, "enhanced_base", off, torch.float32),
            "noise_target": self._sample_copy(shard, "noise_target", off, torch.float32).view(-1),
            "snr_target": self._sample_copy(shard, "snr_target", off, torch.float32).view(-1),
            "snr_valid": self._sample_copy(shard, "snr_valid", off, torch.bool).view(-1),
            "bandwidth_target": self._sample_copy(shard, "bandwidth_target", off, torch.long).view(()),
            "bit_target": self._sample_copy(shard, "bit_target", off, torch.long).view(()),
            "bandwidth_limited": self._sample_copy(shard, "bandwidth_limited", off, torch.bool),
            "severity": shard["severity"][off],
        }
