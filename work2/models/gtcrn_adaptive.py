"""
End-to-end adaptive GTCRN inference pipeline.

Combines:
    1. Frozen GTCRN (Phase 1) → base enhanced spectrogram + bottleneck stats
    2. Degradation Estimator (Phase 3) → degradation predictions
    3. Adaptive Residual Module → condition-modulated spectral correction

Training: only the residual module and estimator are trained.
GTCRN backbone is always frozen.
"""
import torch
import torch.nn as nn
from typing import Optional

from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor
from work2.models.degradation_estimator import DegradationEstimator
from work2.models.adaptive_residual import AdaptiveResidualModule, AdaptiveResidualModuleV2Formal


def build_degradation_condition_vector(
    noise_logit: torch.Tensor,
    snr_pred: torch.Tensor,
    bandwidth_logits: torch.Tensor,
    bit_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate degradation predictions into a condition vector.

    Inputs:
        noise_logit: (B, 1) raw logit
        snr_pred: (B, 1) normalized SNR [0, 1]
        bandwidth_logits: (B, 3) class logits
        bit_logits: (B, 4) class logits

    Returns:
        cond: (B, 9) concatenated condition vector
    """
    noise_p = torch.sigmoid(noise_logit)
    bw_p = torch.softmax(bandwidth_logits, dim=-1)
    bit_p = torch.softmax(bit_logits, dim=-1)

    return torch.cat([noise_p, snr_pred, bw_p, bit_p], dim=-1)


class GTCRNAdaptive(nn.Module):
    """
    Full adaptive GTCRN pipeline.

    Usage:
        model = GTCRNAdaptive(
            checkpoint_path="checkpoints/model_trained_on_dns3.tar",
            device="cpu",
        )

        # Training mode: train estimator and residual
        model.train_estimator_and_residual()

        # Inference
        out = model(spec_ri)  # returns enhanced_final, metrics, etc.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        freeze_gtcrn: bool = True,
        freeze_estimator: bool = False,
        residual_version: str = "v1",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.freeze_gtcrn = freeze_gtcrn
        self.freeze_estimator = freeze_estimator
        self.residual_version = residual_version

        self.extractor = FrozenGTCRNFeatureExtractor(
            checkpoint_path=checkpoint_path, device=str(self.device)
        )

        self.stats_dim = self.extractor.get_stats_dim()
        self.estimator = DegradationEstimator(input_dim=self.stats_dim).to(self.device)

        if self.residual_version == "v1":
            self.residual_module = AdaptiveResidualModule(
                n_freqs=257, cond_dim=9, hidden_dim=32
            ).to(self.device)
        elif self.residual_version == "v2":
            self.residual_module = AdaptiveResidualModuleV2Formal(
                n_freqs=257, cond_dim=9, hidden_dim=32
            ).to(self.device)
        else:
            raise ValueError(f"Unknown residual_version: {self.residual_version}")

        self._apply_freezing()

    def _apply_freezing(self):
        if self.freeze_gtcrn:
            for param in self.extractor.parameters():
                param.requires_grad_(False)

        if self.freeze_estimator:
            for param in self.estimator.parameters():
                param.requires_grad_(False)

    def train_estimator_and_residual(self):
        self.freeze_estimator = False
        self._apply_freezing()
        self.estimator.train()
        self.residual_module.train()

    def train_residual_only(self):
        self.freeze_estimator = True
        self._apply_freezing()
        self.residual_module.train()

    @torch.no_grad()
    def extract_features(self, spec_ri: torch.Tensor) -> dict:
        return self.extractor(spec_ri)

    def forward(
        self,
        spec_ri: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            spec_ri: (B, F, T, 2) complex spectrogram

        Returns:
            dict with:
                enhanced_final: (B, 2, T, F) final enhanced
                enhanced_base: (B, 2, T, F) GTCRN base
                residual: (B, 2, T, F) applied correction
                degradation_conds: (B, 9) condition vector
                degradation_preds: estimator output dict
                bottleneck: (B, C, T, F) GTCRN bottleneck
                stats: (B, 2C) bottleneck statistics
        """
        batch_size = spec_ri.shape[0]
        original_shape = spec_ri.shape

        spec_ri = spec_ri.to(self.device)
        feats = self.extractor(spec_ri)

        enhanced_base = feats["enhanced_spec"]  # (B, F, T, 2)
        enhanced_base_permuted = enhanced_base.permute(0, 3, 2, 1)  # (B, 2, T, F)
        stats = feats["stats"]

        degradation_preds = self.estimator(stats)

        cond = build_degradation_condition_vector(
            degradation_preds["noise_logit"],
            degradation_preds["snr_pred"],
            degradation_preds["bandwidth_logits"],
            degradation_preds["bit_logits"],
        )

        residual_out = self.residual_module(enhanced_base_permuted, cond)

        enhanced_final_permuted = residual_out["enhanced_final"]
        enhanced_final = enhanced_final_permuted.permute(0, 3, 2, 1)
        residual_permuted = residual_out["residual"]
        residual = residual_permuted.permute(0, 3, 2, 1)

        return {
            "enhanced_final": enhanced_final,
            "enhanced_base": enhanced_base,
            "residual": residual,
            "degradation_conds": cond,
            "degradation_preds": degradation_preds,
            "bottleneck": feats["bottleneck"],
            "stats": stats,
        }

    def count_params(self) -> dict[str, int]:
        return {
            "extractor_total": sum(p.numel() for p in self.extractor.parameters()),
            "extractor_trainable": sum(p.numel() for p in self.extractor.parameters() if p.requires_grad),
            "estimator_total": sum(p.numel() for p in self.estimator.parameters()),
            "estimator_trainable": sum(p.numel() for p in self.estimator.parameters() if p.requires_grad),
            "residual_total": sum(p.numel() for p in self.residual_module.parameters()),
            "residual_trainable": sum(p.numel() for p in self.residual_module.parameters() if p.requires_grad),
        }
