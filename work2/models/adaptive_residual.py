"""
Adaptive residual modules for EDCR-Net.

V1 currently supports an optional sample-wise dynamic residual scale head.
V2-Refined-B changes relative to V2-Refined-A:
- keeps alpha_max=0.2 and the global condition gate
- initializes the global gate conservatively (default sigmoid(-2)=0.119)
- zero-initializes expert output layers, so training starts exactly at Base
- bounds each expert candidate relative to the RMS of its Base band
  so expert weights cannot bypass alpha/global gates by arbitrary rescaling
- returns effective/raw gates and per-sample residual ratios
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class AdaptiveResidualModule(nn.Module):
    """Original V1 residual module retained for comparison."""

    def __init__(
        self,
        n_freqs: int = 257,
        cond_dim: int = 9,
        hidden_dim: int = 32,
        min_scale: float = 0.15,
        max_scale: float = 0.85,
    ) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.cond_dim = int(cond_dim)
        if not 0.0 <= min_scale < max_scale <= 1.0:
            raise ValueError('Invalid dynamic scale range.')
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
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
        self.scale_head = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._initialize_scale_head(initial_scale=0.65)

    def _initialize_scale_head(self, initial_scale: float) -> None:
        initial_scale = min(max(initial_scale, self.min_scale + 1e-4), self.max_scale - 1e-4)
        normalized = (initial_scale - self.min_scale) / (self.max_scale - self.min_scale)
        initial_logit = math.log(normalized / (1.0 - normalized))
        last_layer = self.scale_head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.constant_(last_layer.bias, initial_logit)


    def forward_v1_impl(
        self,
        enhanced_spec: torch.Tensor,
        degradation_conds: torch.Tensor,
        force_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if enhanced_spec.ndim != 4:
            raise ValueError("enhanced_spec must be shaped (B,2,T,F).")
        bsz, channels, frames, n_freqs = enhanced_spec.shape
        if channels != 2 or n_freqs != self.n_freqs:
            raise ValueError(f"Unexpected enhanced_spec shape: {tuple(enhanced_spec.shape)}")
        if degradation_conds.shape != (bsz, self.cond_dim):
            raise ValueError(
                f"degradation_conds must be ({bsz},{self.cond_dim}), "
                f"got {tuple(degradation_conds.shape)}"
            )

        cond = self.cond_embed(degradation_conds)
        x = enhanced_spec.permute(0, 2, 1, 3).contiguous().reshape(
            bsz * frames, channels, n_freqs
        )
        x = self.freq_prelu(self.freq_conv1(x))
        x = self.freq_conv2(x)
        candidate = x.reshape(bsz, frames, channels, n_freqs).permute(0, 2, 1, 3).contiguous()
        alpha = self.cond_gate(cond).unsqueeze(-1).unsqueeze(-1)
        raw_residual = alpha * candidate

        if force_scale is None:
            scale_logit = self.scale_head(degradation_conds)
            dynamic_scale = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(scale_logit)
        else:
            if not self.min_scale <= force_scale <= self.max_scale:
                raise ValueError(
                    f"force_scale must be in [{self.min_scale}, {self.max_scale}], got {force_scale}"
                )
            dynamic_scale = enhanced_spec.new_full((bsz, 1), float(force_scale))

        scale_4d = dynamic_scale[:, :, None, None]
        residual = scale_4d * raw_residual
        enhanced_final = enhanced_spec + residual
        residual_ratio = torch.linalg.vector_norm(residual.flatten(1), dim=1) / (
            torch.linalg.vector_norm(enhanced_spec.flatten(1), dim=1) + 1e-8
        )
        return {
            "enhanced_final": enhanced_final,
            "raw_residual": raw_residual,
            "residual": residual,
            "dynamic_scale": dynamic_scale,
            "alpha": alpha.squeeze(-1).squeeze(-1),
            "raw_alpha": alpha.squeeze(-1).squeeze(-1),
            "global_alpha": torch.ones(bsz, 1, device=enhanced_spec.device),
            "residual_ratio": residual_ratio,
        }

    def forward(self, enhanced_spec: torch.Tensor, degradation_conds: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_v1_impl(enhanced_spec, degradation_conds)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveResidualModuleV2Formal(nn.Module):
    """Gate-controlled, amplitude-bounded three-band residual module."""

    def __init__(
        self,
        n_freqs: int = 257,
        cond_dim: int = 9,
        hidden_dim: int = 32,
        alpha_max: float = 0.2,
        candidate_scale: float = 1.0,
        global_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if n_freqs < 3:
            raise ValueError("n_freqs must be at least 3.")
        if not (0.0 < alpha_max <= 1.0):
            raise ValueError("alpha_max must be in (0,1].")
        if candidate_scale <= 0:
            raise ValueError("candidate_scale must be positive.")

        self.n_freqs = int(n_freqs)
        self.cond_dim = int(cond_dim)
        self.n_bands = 3
        self.alpha_max = float(alpha_max)
        self.candidate_scale = float(candidate_scale)
        self.global_gate_init = float(global_gate_init)
        self.band_boundaries = [0, int(n_freqs * 0.30), int(n_freqs * 0.70), n_freqs]

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
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.low_expert = self._make_expert()
        self.mid_expert = self._make_expert()
        self.high_expert = self._make_expert()
        self.reset_parameters()

    @staticmethod
    def _make_expert() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=5, padding=2, groups=2),
            nn.PReLU(),
            nn.Conv1d(8, 2, kernel_size=5, padding=2, groups=2),
        )

    def reset_parameters(self) -> None:
        # Exact Base initialization.
        for expert in (self.low_expert, self.mid_expert, self.high_expert):
            last_conv = expert[-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

        # Conservative global-gate initialization.
        final_linear = self.global_gate[-2]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("global_gate must end with Linear -> Sigmoid.")
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, self.global_gate_init)

        # Raw band gates start at alpha_max * 0.5.
        band_linear = self.expert_gate[-1]
        nn.init.zeros_(band_linear.bias)

    def _bound_candidate(self, candidate: torch.Tensor, band_spec: torch.Tensor) -> torch.Tensor:
        """
        Bound candidate amplitude by the Base-band RMS.

        tanh prevents arbitrary expert-weight growth from cancelling small gates.
        With effective gate <= alpha_max, the residual ratio is therefore controlled
        by construction rather than only by a soft penalty.
        """
        band_rms = band_spec.detach().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
        band_rms = band_rms.clamp_min(1e-5)
        return torch.tanh(candidate) * band_rms * self.candidate_scale

    def forward(self, enhanced_spec: torch.Tensor, degradation_conds: torch.Tensor) -> dict[str, torch.Tensor]:
        if enhanced_spec.ndim != 4:
            raise ValueError("enhanced_spec must be shaped (B,2,T,F).")
        bsz, channels, frames, n_freqs = enhanced_spec.shape
        if channels != 2 or n_freqs != self.n_freqs:
            raise ValueError(f"Unexpected enhanced_spec shape: {tuple(enhanced_spec.shape)}")
        if degradation_conds.shape != (bsz, self.cond_dim):
            raise ValueError(
                f"degradation_conds must be ({bsz},{self.cond_dim}), "
                f"got {tuple(degradation_conds.shape)}"
            )

        cond = self.cond_embed(degradation_conds)
        raw_alpha = self.alpha_max * torch.sigmoid(self.expert_gate(cond))
        global_alpha = self.global_gate(cond)
        effective_alpha = raw_alpha * global_alpha

        experts = (self.low_expert, self.mid_expert, self.high_expert)
        residual_bands: list[torch.Tensor] = []
        candidate_ratios: list[torch.Tensor] = []

        for band_idx, expert in enumerate(experts):
            f_start = self.band_boundaries[band_idx]
            f_end = self.band_boundaries[band_idx + 1]
            band_width = f_end - f_start
            band_spec = enhanced_spec[..., f_start:f_end]
            band_flat = band_spec.permute(0, 2, 1, 3).contiguous().reshape(
                bsz * frames, channels, band_width
            )
            candidate = expert(band_flat)
            candidate = candidate.reshape(bsz, frames, channels, band_width).permute(
                0, 2, 1, 3
            ).contiguous()
            candidate = self._bound_candidate(candidate, band_spec)

            gate = effective_alpha[:, band_idx].view(bsz, 1, 1, 1)
            residual_bands.append(gate * candidate)

            candidate_ratio = torch.linalg.vector_norm(candidate.flatten(1), dim=1) / (
                torch.linalg.vector_norm(band_spec.flatten(1), dim=1) + 1e-8
            )
            candidate_ratios.append(candidate_ratio)

        residual = torch.cat(residual_bands, dim=-1)
        enhanced_final = enhanced_spec + residual
        residual_ratio = torch.linalg.vector_norm(residual.flatten(1), dim=1) / (
            torch.linalg.vector_norm(enhanced_spec.flatten(1), dim=1) + 1e-8
        )

        return {
            "enhanced_final": enhanced_final,
            "residual": residual,
            "alpha": effective_alpha,
            "raw_alpha": raw_alpha,
            "global_alpha": global_alpha,
            "residual_ratio": residual_ratio,
            "candidate_ratio": torch.stack(candidate_ratios, dim=1),
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
