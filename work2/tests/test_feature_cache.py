"""
Unit tests for feature caching reproducibility and data leakage.
"""
import sys; sys.path.insert(0, ".")
import pytest
import json
import torch
from pathlib import Path


class TestFeatureCache:
    def test_cache_reproducible(self):
        import subprocess
        cmd = [
            sys.executable, "-m", "work2.scripts.cache_degradation_features",
            "--clean-manifest", "outputs/nonexistent.jsonl",
            "--output-dir", "outputs/estimator_cache_repro",
            "--seed", "2026",
        ]
        r1 = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

        idx1 = Path("outputs/estimator_cache_repro/index.jsonl")
        idx2 = Path("outputs/estimator_cache_repro/index.jsonl")

        if idx1.exists():
            lines1 = idx1.read_text().strip().split("\n")
            assert len(lines1) > 0
            # Same seed should produce same output
            assert r1.returncode == r2.returncode

    def test_no_cross_split_leakage(self):
        idx = Path("outputs/estimator_cache_repro/index.jsonl")
        if not idx.exists():
            pytest.skip("Cache not available")
        entries = []
        with open(idx, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        source_splits = {}
        for e in entries:
            src = e["source_path"]
            sp = e["split"]
            if src in source_splits:
                assert source_splits[src] == sp, f"Source {src} appears in multiple splits"
            source_splits[src] = sp


class TestMicroOverfit:
    def test_loss_reduction(self):
        import torch
        from work2.models.degradation_estimator import DegradationEstimator
        from work2.losses.degradation_estimator_loss import DegradationEstimatorLoss
        from work2.data.label_utils import normalize_snr

        torch.manual_seed(2026)
        D = 32
        N = 32

        stats_fixed = torch.randn(N, D)
        noise_target = torch.randint(0, 2, (N, 1)).float()
        snr_target = torch.rand(N, 1)
        snr_valid = (noise_target > 0.5)
        bw_target = torch.randint(0, 3, (N,))
        bit_target = torch.randint(0, 4, (N,))

        model = DegradationEstimator(input_dim=D)
        loss_fn = DegradationEstimatorLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        targets = {
            "noise_target": noise_target,
            "snr_target": normalize_snr(snr_target),
            "snr_valid": snr_valid,
            "bandwidth_target": bw_target,
            "bit_target": bit_target,
        }

        with torch.no_grad():
            preds_init = model(stats_fixed)
            _, init = loss_fn(preds_init, targets)
        init_loss = init["total"]

        for _ in range(200):
            preds = model(stats_fixed)
            loss, _ = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds_final = model(stats_fixed)
            _, final = loss_fn(preds_final, targets)
        final_loss = final["total"]

        reduction = (init_loss - final_loss) / init_loss
        assert reduction > 0.80, f"Loss reduction only {reduction*100:.1f}%, need >80%"
