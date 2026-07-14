"""
Lightweight multi-task degradation estimator.

Input: GTCRN bottleneck statistics (B, D)
Output: noise presence logit, normalized SNR, bandwidth logits, bit-depth logits

Total trainable params: < 10,000
"""
import torch
import torch.nn as nn


class DegradationEstimator(nn.Module):
    """
    Small multi-task head predicting degradation labels from frozen GTCRN stats.

    Architecture:
        stats (B, D)
          -> Linear(D, 32) -> LayerNorm -> PReLU
          -> Linear(32, 16) -> PReLU
          -> multi-task heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act1 = nn.PReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.PReLU()

        self.noise_head = nn.Linear(hidden_dim // 2, 1)
        self.snr_head = nn.Linear(hidden_dim // 2, 1)
        self.bw_head = nn.Linear(hidden_dim // 2, 3)
        self.bit_head = nn.Linear(hidden_dim // 2, 4)

    def forward(
        self,
        stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            stats: (B, D) GTCRN bottleneck statistics

        Returns:
            dict with keys:
                noise_logit: (B, 1) raw logit for BCE
                snr_pred: (B, 1) normalized SNR in [0, 1]
                bandwidth_logits: (B, 3) class logits
                bit_logits: (B, 4) class logits
        """
        h = self.fc1(stats)
        h = self.ln(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)

        noise_logit = self.noise_head(h)
        snr_pred = torch.sigmoid(self.snr_head(h))
        bandwidth_logits = self.bw_head(h)
        bit_logits = self.bit_head(h)

        return {
            "noise_logit": noise_logit,
            "snr_pred": snr_pred,
            "bandwidth_logits": bandwidth_logits,
            "bit_logits": bit_logits,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
