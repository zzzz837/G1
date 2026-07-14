"""
Unit tests for adaptive residual module and GTCRN adaptive pipeline.
"""
import sys; sys.path.insert(0, ".")
import pytest
import torch
import numpy as np

from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2
from work2.models.gtcrn_adaptive import GTCRNAdaptive, build_degradation_condition_vector
from work2.losses.residual_loss import AdaptiveResidualLoss


class TestAdaptiveResidualModule:
    @pytest.fixture
    def module(self):
        return AdaptiveResidualModule(n_freqs=257)

    def test_output_shape(self, module):
        enhanced = torch.randn(2, 2, 10, 257)
        cond = torch.randn(2, 9)
        out = module(enhanced, cond)
        assert out["enhanced_final"].shape == enhanced.shape
        assert out["residual"].shape == enhanced.shape
        assert out["alpha"].shape[0] == 2

    def test_no_nan_inf(self, module):
        enhanced = torch.randn(2, 2, 10, 257)
        cond = torch.randn(2, 9)
        out = module(enhanced, cond)
        for k, v in out.items():
            assert not v.isnan().any(), f"NaN in {k}"
            assert not v.isinf().any(), f"Inf in {k}"

    def test_param_count(self, module):
        n = module.count_trainable_params()
        assert n < 50000, f"Too many params: {n}"

    def test_residual_applied(self, module):
        torch.manual_seed(42)
        enhanced = torch.randn(2, 2, 10, 257)
        cond = torch.randn(2, 9)
        out = module(enhanced, cond)
        assert not torch.equal(out["enhanced_final"], enhanced)

    def test_invalid_channels_raises(self, module):
        enhanced = torch.randn(2, 3, 10, 257)
        cond = torch.randn(2, 9)
        with pytest.raises(ValueError):
            module(enhanced, cond)


class TestAdaptiveResidualV2:
    @pytest.fixture
    def module_v2(self):
        return AdaptiveResidualModuleV2(n_freqs=257)

    def test_output_shape(self, module_v2):
        enhanced = torch.randn(2, 2, 10, 257)
        cond = torch.randn(2, 9)
        out = module_v2(enhanced, cond)
        assert out["enhanced_final"].shape == enhanced.shape

    def test_bands_sum_to_freqs(self, module_v2):
        assert module_v2.band_boundaries[0] == 0
        assert module_v2.band_boundaries[-1] == 257


class TestGTCRNAdaptive:
    @pytest.fixture(scope="module")
    def model(self):
        return GTCRNAdaptive("checkpoints/model_trained_on_dns3.tar", "cpu")

    def test_gtcrn_frozen(self, model):
        for p in model.extractor.parameters():
            assert not p.requires_grad

    def test_full_forward(self, model):
        spec = torch.randn(2, 257, 10, 2)
        out = model(spec)
        assert "enhanced_final" in out
        assert "enhanced_base" in out
        assert "residual" in out
        assert "degradation_conds" in out
        assert "degradation_preds" in out
        assert out["enhanced_final"].shape == (2, 257, 10, 2)
        assert out["degradation_conds"].shape == (2, 9)

    def test_no_nan_inf(self, model):
        spec = torch.randn(1, 257, 10, 2)
        out = model(spec)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not v.isnan().any(), f"NaN in {k}"
                assert not v.isinf().any(), f"Inf in {k}"

    def test_param_counts(self, model):
        counts = model.count_params()
        assert counts["extractor_trainable"] == 0
        assert counts["estimator_total"] < 5000
        assert counts["residual_total"] < 20000

    def test_train_modes(self, model):
        model.train_estimator_and_residual()
        assert any(p.requires_grad for p in model.estimator.parameters())
        assert any(p.requires_grad for p in model.residual_module.parameters())

        model.train_residual_only()
        assert not any(p.requires_grad for p in model.estimator.parameters())
        assert any(p.requires_grad for p in model.residual_module.parameters())


class TestAdaptiveResidualLoss:
    @pytest.fixture
    def loss_fn(self):
        return AdaptiveResidualLoss()

    def test_normal_batch(self, loss_fn):
        enhanced = torch.randn(4, 2, 10, 257)
        clean = torch.randn(4, 2, 10, 257)
        residual = torch.randn(4, 2, 10, 257) * 0.1
        bw = torch.tensor([True, False, True, False])
        total, comps = loss_fn(enhanced, clean, residual, bw)
        assert not torch.isnan(total)
        assert all(v >= 0 for v in comps.values())

    def test_no_bw_limited(self, loss_fn):
        enhanced = torch.randn(4, 2, 10, 257)
        clean = torch.randn(4, 2, 10, 257)
        residual = torch.randn(4, 2, 10, 257) * 0.1
        bw = torch.tensor([False, False, False, False])
        total, comps = loss_fn(enhanced, clean, residual, bw)
        assert comps["hf"] == 0.0

    def test_residual_penalty(self, loss_fn):
        enhanced = torch.randn(1, 2, 10, 257)
        clean = torch.randn(1, 2, 10, 257)
        small_res = torch.zeros(1, 2, 10, 257)
        large_res = torch.ones(1, 2, 10, 257) * 0.1
        bw = torch.tensor([False])

        _, c_small = loss_fn(enhanced, clean, small_res, bw)
        _, c_large = loss_fn(enhanced, clean, large_res, bw)
        assert c_small["res"] < c_large["res"]


class TestConditionVector:
    def test_shape(self):
        B = 3
        noise_logit = torch.randn(B, 1)
        snr_pred = torch.rand(B, 1)
        bw_logits = torch.randn(B, 3)
        bit_logits = torch.randn(B, 4)

        cond = build_degradation_condition_vector(noise_logit, snr_pred, bw_logits, bit_logits)
        assert cond.shape == (B, 9)

    def test_values_in_range(self):
        B = 5
        noise_logit = torch.randn(B, 1)
        snr_pred = torch.rand(B, 1)
        bw_logits = torch.randn(B, 3)
        bit_logits = torch.randn(B, 4)

        cond = build_degradation_condition_vector(noise_logit, snr_pred, bw_logits, bit_logits)
        assert (cond >= 0).all()
        assert (cond <= 1).all()
