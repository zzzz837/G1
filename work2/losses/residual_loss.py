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
        lambda_protect: float = 0.05,
        lambda_gate: float = 0.0,
        lambda_ratio: float = 0.10,
        lambda_si_sdr: float = 0.05,
        lambda_delta: float = 0.05,
        lambda_scale: float = 0.02,
        lambda_scale_smooth: float = 0.005,
        lambda_scale_order: float = 0.01,
        hf_start_bin: int = 128,
        clean_protect_weight: float = 1.0,
        light_protect_weight: float = 0.5,
        medium_protect_weight: float = 0.1,
        heavy_protect_weight: float = 0.0,
        gate_targets: tuple[float, float, float, float] = (0.05, 0.20, 0.40, 0.65),
        ratio_caps: tuple[float, float, float, float] = (0.03, 0.08, 0.15, 0.20),
        scale_targets: tuple[float, float, float, float] = (0.25, 0.55, 0.70, 0.85),
    ) -> None:
        super().__init__()
        if hf_start_bin < 0:
            raise ValueError('hf_start_bin must be non-negative.')
        if any(not (0.0 <= x <= 1.0) for x in gate_targets):
            raise ValueError('gate_targets must be in [0,1].')
        if any(x < 0.0 for x in ratio_caps):
            raise ValueError('ratio_caps must be non-negative.')

        self.lambda_mag = float(lambda_mag)
        self.lambda_hf = float(lambda_hf)
        self.lambda_res = float(lambda_res)
        self.lambda_complex = float(lambda_complex)
        self.lambda_protect = float(lambda_protect)
        self.lambda_gate = float(lambda_gate)
        self.lambda_ratio = float(lambda_ratio)
        self.lambda_si_sdr = float(lambda_si_sdr)
        self.lambda_delta = float(lambda_delta)
        self.lambda_scale = float(lambda_scale)
        self.lambda_scale_smooth = float(lambda_scale_smooth)
        self.lambda_scale_order = float(lambda_scale_order)
        self.hf_start_bin = int(hf_start_bin)

        self.register_buffer(
            'severity_weights',
            torch.tensor(
                [clean_protect_weight, light_protect_weight, medium_protect_weight, heavy_protect_weight],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer('gate_targets', torch.tensor(gate_targets, dtype=torch.float32), persistent=False)
        self.register_buffer('ratio_caps', torch.tensor(ratio_caps, dtype=torch.float32), persistent=False)
        self.register_buffer('scale_targets', torch.tensor(scale_targets, dtype=torch.float32), persistent=False)

    @staticmethod
    def _severity_indices(severity, device: torch.device) -> torch.Tensor:
        if isinstance(severity, torch.Tensor):
            result = severity.to(device=device, dtype=torch.long).view(-1)
        else:
            mapping = {'clean': 0, 'light': 1, 'medium': 2, 'heavy': 3}
            result = torch.tensor([mapping[str(x)] for x in severity], device=device, dtype=torch.long)
        if result.numel() and (result.min().item() < 0 or result.max().item() > 3):
            raise ValueError('Severity indices must be in {0,1,2,3}.')
        return result

    @staticmethod
    def _si_sdr_loss(estimate_wav: torch.Tensor, target_wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        estimate_wav = estimate_wav - estimate_wav.mean(dim=-1, keepdim=True)
        target_wav = target_wav - target_wav.mean(dim=-1, keepdim=True)
        scale = (estimate_wav * target_wav).sum(dim=-1, keepdim=True) / (target_wav.square().sum(dim=-1, keepdim=True) + eps)
        projected = scale * target_wav
        noise = estimate_wav - projected
        si_sdr = 10.0 * torch.log10((projected.square().sum(dim=-1) + eps) / (noise.square().sum(dim=-1) + eps))
        return -si_sdr.mean()

    @staticmethod
    def _temporal_delta_loss(estimate_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        if estimate_mag.ndim != 3:
            raise ValueError(f'Expected [B,T,F], got {tuple(estimate_mag.shape)}')
        est_delta = estimate_mag[:, 1:, :] - estimate_mag[:, :-1, :]
        tgt_delta = target_mag[:, 1:, :] - target_mag[:, :-1, :]
        return F.l1_loss(est_delta, tgt_delta)

    @staticmethod
    def _scale_order_loss(scale_pred: torch.Tensor, severity_idx: torch.Tensor) -> torch.Tensor:
        loss = scale_pred.new_zeros(())
        clean = scale_pred[severity_idx == 0]
        light = scale_pred[severity_idx == 1]
        medium = scale_pred[severity_idx == 2]
        heavy = scale_pred[severity_idx == 3]

        if clean.numel() > 0 and light.numel() > 0:
            loss = loss + F.relu(0.1 - light.mean() + clean.mean())
        if light.numel() > 0 and medium.numel() > 0:
            loss = loss + F.relu(0.1 - medium.mean() + light.mean())
        if medium.numel() > 0 and heavy.numel() > 0:
            loss = loss + F.relu(0.1 - heavy.mean() + medium.mean())
        return loss

    def forward(
        self,
        enhanced_final: torch.Tensor,
        clean_spec: torch.Tensor,
        residual: torch.Tensor,
        bandwidth_limited: torch.Tensor,
        enhanced_base: torch.Tensor | None = None,
        severity=None,
        global_alpha: torch.Tensor | None = None,
        dynamic_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if enhanced_final.shape != clean_spec.shape or residual.shape != enhanced_final.shape:
            raise ValueError('enhanced_final, clean_spec and residual must share shape.')
        if enhanced_base is None or enhanced_base.shape != enhanced_final.shape:
            raise ValueError('enhanced_base with matching shape is required.')
        if severity is None:
            raise ValueError('severity is required.')

        severity_idx = self._severity_indices(severity, enhanced_final.device)
        if severity_idx.numel() != enhanced_final.shape[0]:
            raise ValueError('severity length must equal batch size.')

        enhanced_mag = torch.sqrt(enhanced_final[:, 0].pow(2) + enhanced_final[:, 1].pow(2) + 1e-12)
        clean_mag = torch.sqrt(clean_spec[:, 0].pow(2) + clean_spec[:, 1].pow(2) + 1e-12)
        loss_mag = F.l1_loss(enhanced_mag, clean_mag) + F.mse_loss(enhanced_mag, clean_mag)

        loss_hf = enhanced_final.new_zeros(())
        bandwidth_limited = bandwidth_limited.to(device=enhanced_final.device, dtype=torch.bool).view(-1)
        idx = bandwidth_limited.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            hf_start = min(self.hf_start_bin, enhanced_final.shape[-1] - 1)
            en_hf = enhanced_mag[idx, :, hf_start:]
            cl_hf = clean_mag[idx, :, hf_start:]
            loss_hf = F.l1_loss(en_hf, cl_hf) + F.mse_loss(en_hf, cl_hf)

        loss_complex = F.l1_loss(enhanced_final, clean_spec)
        loss_res = residual.pow(2).mean()

        estimate_wav = torch.istft(
            torch.view_as_complex(enhanced_final.permute(0, 3, 2, 1).contiguous()),
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512, device=enhanced_final.device).pow(0.5),
            length=(enhanced_final.shape[2] - 1) * 256,
        )
        target_wav = torch.istft(
            torch.view_as_complex(clean_spec.permute(0, 3, 2, 1).contiguous()),
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512, device=clean_spec.device).pow(0.5),
            length=(clean_spec.shape[2] - 1) * 256,
        )
        loss_si_sdr_raw = self._si_sdr_loss(estimate_wav, target_wav)
        loss_si_sdr = loss_si_sdr_raw / 20.0
        loss_delta = self._temporal_delta_loss(enhanced_mag, clean_mag)

        protect_weights = self.severity_weights.to(enhanced_final.device)[severity_idx]
        per_sample_delta = (enhanced_final - enhanced_base).abs().flatten(1).mean(dim=1)
        loss_protect = (per_sample_delta * protect_weights).mean()

        loss_gate = enhanced_final.new_zeros(())
        if self.lambda_gate > 0:
            if global_alpha is None:
                raise ValueError('global_alpha is required when lambda_gate > 0.')
            if global_alpha.shape[0] != enhanced_final.shape[0]:
                raise ValueError(f'global_alpha batch mismatch: {tuple(global_alpha.shape)}')
            alpha_flat = global_alpha.reshape(enhanced_final.shape[0], -1)
            gate_target = self.gate_targets.to(enhanced_final.device)[severity_idx]
            target_expanded = gate_target[:, None].expand_as(alpha_flat)
            loss_gate = F.mse_loss(alpha_flat, target_expanded)

        loss_scale = enhanced_final.new_zeros(())
        loss_scale_smooth = enhanced_final.new_zeros(())
        loss_scale_order = enhanced_final.new_zeros(())
        scale_mean = enhanced_final.new_zeros(())
        scale_min = enhanced_final.new_zeros(())
        scale_max = enhanced_final.new_zeros(())
        if dynamic_scale is None:
            raise ValueError('dynamic_scale required for dynamic-scale training.')

        scale_pred = dynamic_scale.reshape(-1)
        scale_mean = scale_pred.mean()
        scale_min = scale_pred.min()
        scale_max = scale_pred.max()
        if self.lambda_scale > 0:
            if scale_pred.shape[0] != enhanced_final.shape[0]:
                raise ValueError('dynamic_scale batch size mismatch.')
            target_scale = self.scale_targets.to(enhanced_final.device)[severity_idx]
            loss_scale = F.smooth_l1_loss(scale_pred, target_scale)
        if scale_pred.numel() > 1:
            loss_scale_smooth = torch.relu(scale_pred.std(unbiased=False) - 0.25).pow(2)
        if self.lambda_scale_order > 0:
            loss_scale_order = self._scale_order_loss(scale_pred, severity_idx)

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
            + self.lambda_si_sdr * loss_si_sdr
            + self.lambda_delta * loss_delta
            + self.lambda_scale * loss_scale
            + self.lambda_scale_smooth * loss_scale_smooth
            + self.lambda_scale_order * loss_scale_order
        )

        return total, {
            'total': float(total.detach().item()),
            'mag': float(loss_mag.detach().item()),
            'hf': float(loss_hf.detach().item()),
            'complex': float(loss_complex.detach().item()),
            'protect': float(loss_protect.detach().item()),
            'res': float(loss_res.detach().item()),
            'gate': float(loss_gate.detach().item()),
            'ratio': float(loss_ratio.detach().item()),
            'si_sdr_raw': float(loss_si_sdr_raw.detach().item()),
            'si_sdr': float(loss_si_sdr.detach().item()),
            'delta': float(loss_delta.detach().item()),
            'scale': float(loss_scale.detach().item()),
            'scale_smooth': float(loss_scale_smooth.detach().item()),
            'scale_order': float(loss_scale_order.detach().item()),
            'scale_mean': float(scale_mean.detach().item()),
            'scale_min': float(scale_min.detach().item()),
            'scale_max': float(scale_max.detach().item()),
        }
