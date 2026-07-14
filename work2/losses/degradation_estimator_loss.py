"""
Multi-task loss for degradation estimator.

Components:
  - noise: BCEWithLogitsLoss
  - SNR: SmoothL1Loss (masked by snr_valid)
  - bandwidth: CrossEntropyLoss
  - bit-depth: CrossEntropyLoss
"""
import torch
import torch.nn as nn


class DegradationEstimatorLoss(nn.Module):
    """
    Weighted sum of four task losses.

    L = lambda_noise * BCE(noise_logit, noise_target)
      + lambda_snr * SmoothL1(snr_pred, snr_target) * mask
      + lambda_bw * CE(bw_logits, bw_target)
      + lambda_bit * CE(bit_logits, bit_target)

    SNR loss is masked: only samples with snr_valid=True contribute.
    If no valid SNR samples in batch, L_snr = 0 (no NaN).
    """

    def __init__(
        self,
        lambda_noise: float = 1.0,
        lambda_snr: float = 1.0,
        lambda_bw: float = 1.0,
        lambda_bit: float = 1.0,
    ) -> None:
        super().__init__()

        self.lambda_noise = lambda_noise
        self.lambda_snr = lambda_snr
        self.lambda_bw = lambda_bw
        self.lambda_bit = lambda_bit

        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute total and per-task losses.

        Args:
            predictions: from DegradationEstimator.forward()
            targets: dict with noise_target, snr_target, snr_valid,
                     bandwidth_target, bit_target

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of float values for each component
        """
        noise_pred = predictions["noise_logit"]
        noise_target = targets["noise_target"]
        noise_loss = self.bce(noise_pred, noise_target)

        snr_pred = predictions["snr_pred"]
        snr_target = targets["snr_target"]
        snr_valid = targets["snr_valid"]

        n_valid = int(snr_valid.sum().item())
        if n_valid > 0:
            snr_loss_per_sample = self.smooth_l1(snr_pred, snr_target)
            snr_loss = (snr_loss_per_sample * snr_valid).sum() / n_valid
        else:
            snr_loss = torch.tensor(0.0, device=noise_pred.device)

        bw_pred = predictions["bandwidth_logits"]
        bw_target = targets["bandwidth_target"]
        bw_loss = self.ce(bw_pred, bw_target)

        bit_pred = predictions["bit_logits"]
        bit_target = targets["bit_target"]
        bit_loss = self.ce(bit_pred, bit_target)

        total = (
            self.lambda_noise * noise_loss
            + self.lambda_snr * snr_loss
            + self.lambda_bw * bw_loss
            + self.lambda_bit * bit_loss
        )

        loss_dict = {
            "total": total.item(),
            "noise": noise_loss.item(),
            "snr": snr_loss.item(),
            "bandwidth": bw_loss.item(),
            "bit": bit_loss.item(),
        }

        return total, loss_dict
