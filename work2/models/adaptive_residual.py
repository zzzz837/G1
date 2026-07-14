"""
Adaptive Residual Compensation Module.

Takes GTCRN-enhanced spectrogram and degradation predictions,
generates a frequency-aware, condition-modulated residual correction.

Architecture:
  enhanced_spec (B,2,T,F) + degradation_conds (B,9)
    → Condition Embedding (FC→LN→PReLU)
    → Frequency-band processing (Conv1d on F-dim)
    → Condition-modulated gating
    → Residual added to base enhanced
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidualModule(nn.Module):
    """
    Lightweight adaptive residual compensator.

    Condition vector (B, 9):
        [noise_logit(1), snr_pred(1), bw_logits(3), bit_logits(4)]

    Operates on magnitude spectrogram, then reconstructs real/imag.
    """

    def __init__(
        self,
        n_freqs: int = 257,
        cond_dim: int = 9,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        self.n_freqs = n_freqs
        self.cond_dim = cond_dim

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        self.freq_conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2, groups=2)
        self.freq_prelu = nn.PReLU()
        self.freq_conv2 = nn.Conv1d(16, 2, kernel_size=5, padding=2, groups=2)

        self.cond_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        enhanced_spec: torch.Tensor,
        degradation_conds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            enhanced_spec: (B, 2, T, F) GTCRN base enhanced spectrogram
            degradation_conds: (B, 9) concatenated degradation predictions
                [noise_logit(1), snr_pred(1), bw_logits(3), bit_logits(4)]

        Returns:
            dict with:
                enhanced_final: (B, 2, T, F) final enhanced spectrogram
                residual: (B, 2, T, F) the applied residual correction
                alpha: (B, 2, 1) per-channel gate values
        """
        B, C, T, F_in = enhanced_spec.shape

        if C != 2:
            raise ValueError(f"Expected 2 channels (real, imag), got {C}")

        cond = self.cond_embed(degradation_conds)

        x = enhanced_spec.reshape(B * T, C, F_in)
        x = self.freq_prelu(self.freq_conv1(x))
        x = self.freq_conv2(x)
        residual = x.reshape(B, T, C, F_in).permute(0, 2, 1, 3)

        alpha = self.cond_gate(cond)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)

        residual_gated = alpha * residual
        enhanced_final = enhanced_spec + residual_gated

        return {
            "enhanced_final": enhanced_final,
            "residual": residual_gated,
            "alpha": alpha,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveResidualModuleV2(nn.Module):
    """
    Enhanced version with frequency-band attention.

    Splits spectrum into low/mid/high bands and applies per-band
    condition-modulated gating with learnable residual kernels.
    """

    def __init__(
        self,
        n_freqs: int = 257,
        cond_dim: int = 9,
        hidden_dim: int = 32,
        n_bands: int = 3,
    ) -> None:
        super().__init__()

        self.n_freqs = n_freqs
        self.n_bands = n_bands

        self.band_boundaries = [0, int(n_freqs * 0.3), int(n_freqs * 0.7), n_freqs]

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, 8, kernel_size=5, padding=2, groups=2),
                nn.PReLU(),
                nn.Conv1d(8, 2, kernel_size=5, padding=2, groups=2),
            )
            for _ in range(n_bands)
        ])

        self.band_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, n_bands * 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        enhanced_spec: torch.Tensor,
        degradation_conds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, C, T, F_in = enhanced_spec.shape
        cond = self.cond_embed(degradation_conds)

        alpha_per_band = self.band_gate(cond)
        alpha_per_band = alpha_per_band.reshape(B, self.n_bands, C).transpose(1, 2)
        alpha_per_band = alpha_per_band.unsqueeze(-1)

        residual = torch.zeros_like(enhanced_spec)

        for b in range(self.n_bands):
            f_start = self.band_boundaries[b]
            f_end = self.band_boundaries[b + 1]

            band_spec = enhanced_spec[:, :, :, f_start:f_end]
            b_t, b_f = band_spec.shape[2], band_spec.shape[3]
            band_flat = band_spec.reshape(B * b_t, C, b_f)
            band_residual = self.band_convs[b](band_flat)
            band_residual = band_residual.reshape(B, b_t, C, b_f).permute(0, 2, 1, 3)

            alpha_b = alpha_per_band[:, :, b:b+1]
            residual[:, :, :, f_start:f_end] = alpha_b * band_residual

        enhanced_final = enhanced_spec + residual

        return {
            "enhanced_final": enhanced_final,
            "residual": residual,
            "alpha": alpha_per_band,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
