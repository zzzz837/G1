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

        x = (
            enhanced_spec
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B * T, C, F_in)
        )
        x = self.freq_prelu(self.freq_conv1(x))
        x = self.freq_conv2(x)
        residual = (
            x.reshape(B, T, C, F_in)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

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


class AdaptiveResidualModuleV2Formal(nn.Module):
    """
    Formal lightweight multi-band expert residual module.

    Three residual experts are defined over low / mid / high bands.
    Each band receives its own independent sigmoid gate scaled by alpha_max:
        alpha = alpha_max * sigmoid(G(e_d))
    and the final residual is:
        R = alpha_L * R_L + alpha_M * R_M + alpha_H * R_H

    Compared with the earlier V2 draft:
    - uses correct (B,T,C,Fb) -> (B*T,C,Fb) reshaping
    - uses independent sigmoid gates instead of softmax competition
    - avoids cross-band competition for non-overlapping bands
    """

    def __init__(self, n_freqs: int = 257, cond_dim: int = 9, hidden_dim: int = 32, alpha_max: float = 0.5):
        super().__init__()
        self.n_freqs = n_freqs
        self.band_boundaries = [0, int(n_freqs * 0.30), int(n_freqs * 0.70), n_freqs]
        self.n_bands = 3
        self.alpha_max = alpha_max

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        self.expert_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, self.n_bands),
        )

        self.low_expert = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=5, padding=2, groups=2),
            nn.PReLU(),
            nn.Conv1d(8, 2, kernel_size=5, padding=2, groups=2),
        )
        self.mid_expert = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=5, padding=2, groups=2),
            nn.PReLU(),
            nn.Conv1d(8, 2, kernel_size=5, padding=2, groups=2),
        )
        self.high_expert = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=5, padding=2, groups=2),
            nn.PReLU(),
            nn.Conv1d(8, 2, kernel_size=5, padding=2, groups=2),
        )

    def forward(self, enhanced_spec: torch.Tensor, degradation_conds: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, T, F_in = enhanced_spec.shape
        if C != 2:
            raise ValueError(f"Expected 2 channels (real, imag), got {C}")
        if F_in != self.n_freqs:
            raise ValueError(f"Expected {self.n_freqs} frequency bins, got {F_in}")

        cond = self.cond_embed(degradation_conds)
        alpha = self.alpha_max * torch.sigmoid(self.expert_gate(cond))  # (B, 3)

        experts = (self.low_expert, self.mid_expert, self.high_expert)
        residual_bands = []

        for band_idx, expert in enumerate(experts):
            f_start = self.band_boundaries[band_idx]
            f_end = self.band_boundaries[band_idx + 1]
            band_width = f_end - f_start

            band_spec = enhanced_spec[..., f_start:f_end]  # (B,2,T,Fb)
            band_flat = (
                band_spec
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B * T, C, band_width)
            )

            band_residual = expert(band_flat)
            band_residual = (
                band_residual
                .reshape(B, T, C, band_width)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            alpha_b = alpha[:, band_idx].view(B, 1, 1, 1)
            residual_bands.append(alpha_b * band_residual)

        residual = torch.cat(residual_bands, dim=-1)
        enhanced_final = enhanced_spec + residual

        return {
            "enhanced_final": enhanced_final,
            "residual": residual,
            "alpha": alpha,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
