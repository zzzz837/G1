"""
Label utility functions for degradation estimation.

Defines SNR normalization range and conversion helpers.
"""
import torch

SNR_MIN_DB = 0.0
SNR_MAX_DB = 40.0


def normalize_snr(snr_db: torch.Tensor) -> torch.Tensor:
    """
    Normalize SNR in dB to [0, 1].

    snr_norm = clip((snr_db - SNR_MIN_DB) / (SNR_MAX_DB - SNR_MIN_DB), 0, 1)

    Args:
        snr_db: SNR values in dB, shape (B,) or (B,1)

    Returns:
        normalized SNR, same shape as input
    """
    snr_norm = (snr_db - SNR_MIN_DB) / (SNR_MAX_DB - SNR_MIN_DB)
    snr_norm = torch.clamp(snr_norm, 0.0, 1.0)
    return snr_norm


def denormalize_snr(snr_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized SNR back to dB.

    snr_db = snr_norm * (SNR_MAX_DB - SNR_MIN_DB) + SNR_MIN_DB

    Args:
        snr_norm: normalized SNR in [0, 1]

    Returns:
        SNR in dB
    """
    return snr_norm * (SNR_MAX_DB - SNR_MIN_DB) + SNR_MIN_DB


def make_degradation_labels(result) -> dict:
    """
    Convert a DegradationResult to a label dict for training.

    Args:
        result: DegradationResult from degradation.py

    Returns:
        dict with keys:
            noise_present, snr_target, snr_valid,
            bandwidth_class, bit_class, severity
    """
    noise_present = 1.0 if result.noise_present else 0.0
    snr_valid = result.noise_present and result.snr_target_db is not None
    snr_target = result.snr_target_db if snr_valid else 0.0

    return {
        "noise_present": float(noise_present),
        "snr_target": float(snr_target),
        "snr_valid": bool(snr_valid),
        "bandwidth_class": int(result.bandwidth_class),
        "bit_class": int(result.bit_class),
        "severity": str(result.severity),
    }
