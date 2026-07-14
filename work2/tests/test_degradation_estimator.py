"""
Unit tests for DegradationEstimator.
"""
import sys; sys.path.insert(0, ".")
import pytest
import torch
from work2.models.degradation_estimator import DegradationEstimator


class TestDegradationEstimator:
    @pytest.fixture
    def model(self):
        return DegradationEstimator(input_dim=32)

    def test_output_shapes(self, model):
        x = torch.randn(4, 32)
        out = model(x)
        assert out["noise_logit"].shape == (4, 1)
        assert out["snr_pred"].shape == (4, 1)
        assert out["bandwidth_logits"].shape == (4, 3)
        assert out["bit_logits"].shape == (4, 4)

    def test_parameter_count_under_limit(self, model):
        n = model.count_trainable_params()
        assert n < 10000, f"Too many params: {n}"

    def test_backward_produces_gradients(self, model):
        x = torch.randn(8, 32)
        out = model(x)
        loss = out["noise_logit"].mean()
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found in estimator parameters"

    def test_snr_pred_in_range(self, model):
        x = torch.randn(16, 32)
        out = model(x)
        snr = out["snr_pred"]
        assert snr.min() >= 0.0
        assert snr.max() <= 1.0
