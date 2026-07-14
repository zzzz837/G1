"""
Audio utility functions for GTCRN work2 degradation pipeline.
All functions operate on CPU float32 tensors.
"""
import torch
import numpy as np


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert multi-channel audio to mono by averaging channels.

    Args:
        waveform: shape (samples,) or (channels, samples)

    Returns:
        torch.Tensor: 1D float32 tensor of shape (samples,)

    Raises:
        ValueError: if waveform is empty or has unsupported dimensionality
    """
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")

    if waveform.numel() == 0:
        raise ValueError("Waveform is empty")

    if waveform.dim() == 1:
        return waveform.to(torch.float32)

    if waveform.dim() == 2:
        return waveform.mean(dim=0).to(torch.float32)

    raise ValueError(f"Unsupported tensor shape: {waveform.shape}, expected 1D or 2D")


def peak_normalize(waveform: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    """
    Normalize waveform so that max absolute value equals 'peak'.

    Args:
        waveform: 1D float32 tensor
        peak: target peak amplitude (default 0.95)

    Returns:
        torch.Tensor: normalized waveform

    Raises:
        ValueError: if waveform contains NaN or Inf
    """
    if waveform.numel() == 0:
        raise ValueError("Waveform is empty")

    max_val = waveform.abs().max()
    if max_val == 0.0:
        return waveform.clone()

    if torch.isinf(max_val) or torch.isnan(max_val):
        raise ValueError("Waveform contains Inf or NaN")

    return waveform * (peak / max_val)


def match_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Match waveform to target length by cropping or cycle-repeating.

    Args:
        waveform: 1D float32 tensor
        target_length: desired number of samples

    Returns:
        torch.Tensor of length target_length

    Raises:
        ValueError: if waveform is empty or target_length <= 0
    """
    if waveform.numel() == 0:
        raise ValueError("Waveform is empty")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    n = waveform.numel()

    if n >= target_length:
        return waveform[:target_length].clone()

    repeats = (target_length + n - 1) // n
    repeated = waveform.repeat(repeats)
    return repeated[:target_length].clone()


def rms(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute root mean square of a waveform.

    Args:
        waveform: 1D float32 tensor
        eps: epsilon to avoid division by zero

    Returns:
        scalar tensor
    """
    return torch.sqrt(torch.mean(waveform.pow(2)) + eps)


def calculate_snr_db(clean: torch.Tensor, degraded: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate SNR in dB between clean and degraded signals.

    SNR = 10 * log10( power(clean) / power(noise) )
    noise = degraded - clean

    Args:
        clean: clean signal, 1D tensor
        degraded: degraded signal, 1D tensor
        eps: epsilon for numerical stability

    Returns:
        float: SNR in dB

    Raises:
        ValueError: if tensors have different lengths
    """
    if clean.shape != degraded.shape:
        raise ValueError(f"Shape mismatch: {clean.shape} vs {degraded.shape}")

    noise = degraded - clean
    signal_power = torch.mean(clean.pow(2))
    noise_power = torch.mean(noise.pow(2))

    if signal_power < eps:
        raise ValueError(f"Clean signal power is too low: {signal_power.item():.2e}")

    snr = 10.0 * torch.log10(signal_power / (noise_power + eps))
    return float(snr.item())


def check_waveform(waveform: torch.Tensor) -> None:
    """
    Validate waveform integrity.

    Checks:
    - 1D shape
    - No NaN or Inf
    - Not empty
    - Values approximately within [-1.0, 1.0] (allows small overshoot)

    Args:
        waveform: tensor to check

    Raises:
        TypeError, ValueError: on validation failure
    """
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")

    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {waveform.shape}")

    if waveform.numel() == 0:
        raise ValueError("Waveform is empty")

    if torch.isnan(waveform).any():
        raise ValueError("Waveform contains NaN")

    if torch.isinf(waveform).any():
        raise ValueError("Waveform contains Inf")

    abs_max = waveform.abs().max().item()
    if abs_max > 1.0001:
        raise ValueError(f"Waveform values exceed [-1, 1]: max abs = {abs_max:.6f}")
