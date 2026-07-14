"""
Reproducible random control for GTCRN work2 pipeline.
"""
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python random, NumPy, and PyTorch (CPU) for full reproducibility.

    Args:
        seed: integer seed for all random number generators

    Note:
        Also sets PyTorch deterministic flags for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(False)
