"""
Unit tests for DegradationEstimatorLoss (masked SNR loss).
"""
import sys; sys.path.insert(0, ".")
import pytest
import torch
from work2.losses.degradation_estimator_loss import DegradationEstimatorLoss


class TestEstimatorLoss:
    @pytest.fixture
    def loss_fn(self):
        return DegradationEstimatorLoss()

    def test_normal_batch(self, loss_fn):
        preds = {
            "noise_logit": torch.randn(4, 1),
            "snr_pred": torch.rand(4, 1),
            "bandwidth_logits": torch.randn(4, 3),
            "bit_logits": torch.randn(4, 4),
        }
        targets = {
            "noise_target": torch.tensor([[1.], [0.], [1.], [0.]]),
            "snr_target": torch.tensor([[0.5], [0.0], [0.3], [0.9]]),
            "snr_valid": torch.tensor([[True], [False], [True], [False]]),
            "bandwidth_target": torch.tensor([1, 0, 2, 0]),
            "bit_target": torch.tensor([2, 0, 3, 1]),
        }
        total, comps = loss_fn(preds, targets)
        assert not torch.isnan(total)
        for k, v in comps.items():
            assert not torch.isnan(torch.tensor(v)), f"NaN in {k}"
            assert v >= 0, f"Negative loss in {k}"

    def test_no_valid_snr(self, loss_fn):
        preds = {
            "noise_logit": torch.randn(4, 1),
            "snr_pred": torch.rand(4, 1),
            "bandwidth_logits": torch.randn(4, 3),
            "bit_logits": torch.randn(4, 4),
        }
        targets = {
            "noise_target": torch.tensor([[0.], [0.], [0.], [0.]]),
            "snr_target": torch.tensor([[0.5], [0.0], [0.3], [0.9]]),
            "snr_valid": torch.tensor([[False], [False], [False], [False]]),
            "bandwidth_target": torch.tensor([0, 0, 0, 0]),
            "bit_target": torch.tensor([0, 0, 0, 0]),
        }
        total, comps = loss_fn(preds, targets)
        assert not torch.isnan(total)
        assert comps["snr"] == 0.0
