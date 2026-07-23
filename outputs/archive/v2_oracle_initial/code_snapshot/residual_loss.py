"""
Loss functions for adaptive residual training.

Components:
  1. Spectral magnitude loss (L1 + MSE on magnitude)
  2. High-frequency emphasis (extra weight for bandwidth-limited samples)
  3. Residual regularization (encourage small residuals)
  4. Composite loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidualLoss(nn.Module):
    """
    Joint loss for training the adaptive residual module.

    L = lambda_mag * L_mag
      + lambda_hf  * L_hf   (high-frequency emphasis)
      + lambda_res  * L_res  (residual L2 penalty)

    L_mag: masked L1 + MSE on magnitude spectrogram
    L_hf:  extra magnitude loss on frequencies above a cutoff,
           activated only for bandwidth-limited samples
    L_res: L2 norm of the residual, scaled by gate alpha
    """

    def __init__(
        self,
        lambda_mag: float = 1.0,
        lambda_hf: float = 1.0,
        lambda_res: float = 0.1,
        hf_cutoff_ratio: float = 0.55,
    ) -> None:
        super().__init__()

        self.lambda_mag = lambda_mag
        self.lambda_hf = lambda_hf
        self.lambda_res = lambda_res
        self.hf_cutoff_ratio = hf_cutoff_ratio

    def forward(
        self,
        enhanced_final: torch.Tensor,
        clean_spec: torch.Tensor,
        residual: torch.Tensor,
        bandwidth_limited: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            enhanced_final: (B, 2, T, F) final enhanced complex spectrogram
            clean_spec: (B, 2, T, F) clean complex spectrogram
            residual: (B, 2, T, F) applied residual correction
            bandwidth_limited: (B,) bool tensor, True if original was BW-limited

        Returns:
            total_loss, loss_dict
        """
        enhanced_mag = torch.sqrt(
            enhanced_final[:, 0] ** 2 + enhanced_final[:, 1] ** 2 + 1e-12
        )
        clean_mag = torch.sqrt(
            clean_spec[:, 0] ** 2 + clean_spec[:, 1] ** 2 + 1e-12
        )

        mag_l1 = F.l1_loss(enhanced_mag, clean_mag)
        mag_mse = F.mse_loss(enhanced_mag, clean_mag)
        L_mag = mag_l1 + mag_mse

        L_hf = torch.tensor(0.0, device=enhanced_final.device)
        n_bw_limited = int(bandwidth_limited.sum().item())
        if n_bw_limited > 0:
            F_total = enhanced_final.shape[-1]
            hf_start = int(F_total * self.hf_cutoff_ratio)
            idx = bandwidth_limited.nonzero(as_tuple=False).squeeze(-1)

            en_hf = enhanced_mag[idx, :, hf_start:]
            cl_hf = clean_mag[idx, :, hf_start:]

            L_hf = F.l1_loss(en_hf, cl_hf) + F.mse_loss(en_hf, cl_hf)

        alpha_like_power = (residual ** 2).mean()
        L_res = alpha_like_power

        total = (
            self.lambda_mag * L_mag
            + self.lambda_hf * L_hf
            + self.lambda_res * L_res
        )

        loss_dict = {
            "total": total.item(),
            "mag": L_mag.item(),
            "hf": L_hf.item(),
            "res": L_res.item(),
        }

        return total, loss_dict
