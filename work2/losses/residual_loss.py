"""Losses for V2-Refined-B gate-controlled residual training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidualLoss(nn.Module):
    def __init__(
        self,
        lambda_mag: float = 1.0,
        lambda_hf: float = 0.5,
        lambda_res: float = 0.05,
        lambda_complex: float = 0.1,
        lambda_protect: float = 0.1,
        lambda_gate: float = 0.02,
        lambda_ratio: float = 0.2,
        hf_cutoff_ratio: float = 0.55,
        clean_protect_weight: float = 1.0,
        light_protect_weight: float = 0.5,
        medium_protect_weight: float = 0.1,
        heavy_protect_weight: float = 0.0,
        gate_targets: tuple[float, float, float, float] = (0.05, 0.25, 0.55, 0.85),
        ratio_caps: tuple[float, float, float, float] = (0.03, 0.08, 0.15, 0.20),
    ) -> None:
        super().__init__()
        if not (0.0 < hf_cutoff_ratio < 1.0):
            raise ValueError("hf_cutoff_ratio must be in (0,1).")
        if any(not (0.0 <= x <= 1.0) for x in gate_targets):
            raise ValueError("gate_targets must be in [0,1].")
        if any(x < 0.0 for x in ratio_caps):
            raise ValueError("ratio_caps must be non-negative.")

        self.lambda_mag = float(lambda_mag)
        self.lambda_hf = float(lambda_hf)
        self.lambda_res = float(lambda_res)
        self.lambda_complex = float(lambda_complex)
        self.lambda_protect = float(lambda_protect)
        self.lambda_gate = float(lambda_gate)
        self.lambda_ratio = float(lambda_ratio)
        self.hf_cutoff_ratio = float(hf_cutoff_ratio)

        self.register_buffer(
            "severity_weights",
            torch.tensor(
                [clean_protect_weight, light_protect_weight, medium_protect_weight, heavy_protect_weight],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer("gate_targets", torch.tensor(gate_targets, dtype=torch.float32), persistent=False)
        self.register_buffer("ratio_caps", torch.tensor(ratio_caps, dtype=torch.float32), persistent=False)

    @staticmethod
    def _severity_indices(severity, device: torch.device) -> torch.Tensor:
        if isinstance(severity, torch.Tensor):
            result = severity.to(device=device, dtype=torch.long).view(-1)
        else:
            mapping = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}
            result = torch.tensor([mapping[str(x)] for x in severity], device=device, dtype=torch.long)
        if result.numel() and (result.min().item() < 0 or result.max().item() > 3):
            raise ValueError("Severity indices must be in {0,1,2,3}.")
        return result

    def forward(
        self,
        enhanced_final: torch.Tensor,
        clean_spec: torch.Tensor,
        residual: torch.Tensor,
        bandwidth_limited: torch.Tensor,
        enhanced_base: torch.Tensor | None = None,
        severity=None,
        global_alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if enhanced_final.shape != clean_spec.shape or residual.shape != enhanced_final.shape:
            raise ValueError("enhanced_final, clean_spec and residual must share shape.")
        if enhanced_base is None or enhanced_base.shape != enhanced_final.shape:
            raise ValueError("enhanced_base with matching shape is required.")
        if severity is None:
            raise ValueError("severity is required.")

        severity_idx = self._severity_indices(severity, enhanced_final.device)
        if severity_idx.numel() != enhanced_final.shape[0]:
            raise ValueError("severity length must equal batch size.")

        enhanced_mag = torch.sqrt(enhanced_final[:, 0].pow(2) + enhanced_final[:, 1].pow(2) + 1e-12)
        clean_mag = torch.sqrt(clean_spec[:, 0].pow(2) + clean_spec[:, 1].pow(2) + 1e-12)
        loss_mag = F.l1_loss(enhanced_mag, clean_mag) + F.mse_loss(enhanced_mag, clean_mag)

        loss_hf = enhanced_final.new_zeros(())
        bandwidth_limited = bandwidth_limited.to(device=enhanced_final.device, dtype=torch.bool).view(-1)
        idx = bandwidth_limited.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            hf_start = int(enhanced_final.shape[-1] * self.hf_cutoff_ratio)
            en_hf = enhanced_mag[idx, :, hf_start:]
            cl_hf = clean_mag[idx, :, hf_start:]
            loss_hf = F.l1_loss(en_hf, cl_hf) + F.mse_loss(en_hf, cl_hf)

        loss_complex = F.l1_loss(enhanced_final, clean_spec)
        loss_res = residual.pow(2).mean()

        protect_weights = self.severity_weights.to(enhanced_final.device)[severity_idx]
        per_sample_delta = (enhanced_final - enhanced_base).abs().flatten(1).mean(dim=1)
        loss_protect = (per_sample_delta * protect_weights).mean()

        # Condition-aware global gate prior. It is intentionally weak: the main
        # enhancement objective remains spectral reconstruction.
        loss_gate = enhanced_final.new_zeros(())
        if self.lambda_gate > 0:
            if global_alpha is None:
                raise ValueError("global_alpha is required when lambda_gate > 0.")
            gate_target = self.gate_targets.to(enhanced_final.device)[severity_idx]
            loss_gate = F.mse_loss(global_alpha.view(-1), gate_target)

        # Hinge penalty only when residual ratio exceeds the severity-specific cap.
        residual_ratio = torch.linalg.vector_norm(residual.flatten(1), dim=1) / (
            torch.linalg.vector_norm(enhanced_base.flatten(1), dim=1) + 1e-8
        )
        ratio_cap = self.ratio_caps.to(enhanced_final.device)[severity_idx]
        loss_ratio = torch.relu(residual_ratio - ratio_cap).pow(2).mean()

        total = (
            self.lambda_mag * loss_mag
            + self.lambda_hf * loss_hf
            + self.lambda_complex * loss_complex
            + self.lambda_protect * loss_protect
            + self.lambda_res * loss_res
            + self.lambda_gate * loss_gate
            + self.lambda_ratio * loss_ratio
        )

        return total, {
            "total": float(total.detach().item()),
            "mag": float(loss_mag.detach().item()),
            "hf": float(loss_hf.detach().item()),
            "complex": float(loss_complex.detach().item()),
            "protect": float(loss_protect.detach().item()),
            "res": float(loss_res.detach().item()),
            "gate": float(loss_gate.detach().item()),
            "ratio": float(loss_ratio.detach().item()),
        }
