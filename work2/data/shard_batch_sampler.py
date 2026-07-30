"""
Shard-aware batch sampler.

Sampling order:
1. Shuffle shard order.
2. Shuffle sample indices inside each shard.
3. Yield batches containing samples from only one shard.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class ShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        drop_last: bool = False,
        seed: int = 2026,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if not hasattr(dataset, "entries"):
            raise TypeError("dataset must expose an 'entries' attribute")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            if "shard_file" not in entry:
                raise KeyError(f"Dataset entry {index} has no shard_file")
            shard_file = str(entry["shard_file"])
            grouped_indices[shard_file].append(index)

        if not grouped_indices:
            raise ValueError("No shard entries found in dataset")

        self.shard_to_indices = dict(grouped_indices)
        self.shard_names = list(self.shard_to_indices.keys())
        self._num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self) -> int:
        total = 0
        for indices in self.shard_to_indices.values():
            size = len(indices)
            if self.drop_last:
                total += size // self.batch_size
            else:
                total += math.ceil(size / self.batch_size)
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        shard_names = self.shard_names.copy()

        if self.shuffle_shards:
            rng.shuffle(shard_names)

        for shard_name in shard_names:
            indices = self.shard_to_indices[shard_name].copy()
            if self.shuffle_within_shard:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start: start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        return self._num_batches
