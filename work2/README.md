# Work2: Adaptive Residual GTCRN

This directory contains the second research phase of the master thesis: improving GTCRN with adaptive residual learning.

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `models/` | Improved GTCRN models and new modules |
| `data/` | Composite degradation generation and datasets |
| `losses/` | New loss functions: degradation estimation, high-frequency and residual constraints |
| `train/` | Feature caching and lightweight head training |
| `evaluation/` | PESQ, STOI, SI-SDR, RTF metrics and plotting |
| `tests/` | Model equivalence and module unit tests |
| `scripts/` | Utility and automation scripts |
| `baseline/` | Official GTCRN baseline checksums and reference data |
| `reports/` | Phase reports and documentation |

## Phase 1: Baseline Reproduction (Current)

**Status**: Completed

**Constraint**: Do NOT modify official `gtcrn.py`, `loss.py`, or `infer.py`.

## Future Work

Phase 2+ will implement novel modules in `models/` and training pipelines in `train/`.
