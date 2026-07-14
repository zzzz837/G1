"""
Composite degradation generation pipeline for GTCRN work2.

Provides:
- Atomic degradation functions: noise addition, bandwidth limiting, resampling,
  low-pass filtering, quantization
- Composite degradation with fixed processing order
- Severity presets: clean, light, medium, heavy
- DegradationResult with label mask

Processing order (fixed):
    clean -> resample_roundtrip -> lowpass_filter -> quantization -> noise addition -> anti-clip normalize
"""
import warnings
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
import scipy.signal

from work2.data.audio_utils import (
    ensure_mono,
    peak_normalize,
    match_length,
    rms,
    calculate_snr_db,
    check_waveform,
)
from work2.data.random_utils import seed_everything


# ---------------------------------------------------------------------------
# Atomic degradation functions
# ---------------------------------------------------------------------------

def quantize_waveform(
    waveform: torch.Tensor,
    bit_depth: int,
) -> torch.Tensor:
    """
    Apply symmetric uniform quantization to a waveform.

    Maps [-1, 1] to 2^bit_depth levels and back to float32.
    16-bit also undergoes true quantization (not a passthrough).

    Args:
        waveform: 1D float32 tensor
        bit_depth: quantization bit depth (8, 10, 12, or 16)

    Returns:
        quantized waveform, same length and dtype

    Raises:
        ValueError: if bit_depth is not supported
    """
    valid_depths = {8, 10, 12, 16}
    if bit_depth not in valid_depths:
        raise ValueError(f"bit_depth must be one of {valid_depths}, got {bit_depth}")

    x = torch.clamp(waveform, -1.0, 1.0)

    max_val = 2 ** (bit_depth - 1) - 1
    x_int = torch.round(x * max_val)
    x_int = torch.clamp(x_int, -max_val, max_val)
    x_out = x_int / max_val

    return x_out.to(torch.float32)


def add_noise_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    avoid_clipping: bool = True,
) -> tuple[torch.Tensor, float]:
    """
    Add noise to a clean signal at a specified SNR.

    Noise is DC-removed, matched in length (repeat-cycling if needed),
    scaled to achieve the target SNR, then added.

    Args:
        clean: clean waveform, 1D float32
        noise: noise waveform, 1D float32
        snr_db: target signal-to-noise ratio in dB
        avoid_clipping: if True, peak-normalize the result to 0.95

    Returns:
        tuple[torch.Tensor, float]: (degraded_waveform, achieved_snr_db)

    Raises:
        ValueError: if clean or noise power is too low
    """
    clean = ensure_mono(clean).to(torch.float32)
    noise = ensure_mono(noise).to(torch.float32)

    clean_rms_val = torch.sqrt(torch.mean(clean.pow(2)))
    if clean_rms_val.item() < 1e-10:
        raise ValueError(f"Clean signal power too low: RMS = {clean_rms_val.item():.2e}")

    noise = noise - noise.mean()
    noise = match_length(noise, clean.numel())

    noise_rms_val = torch.sqrt(torch.mean(noise.pow(2)))
    if noise_rms_val.item() < 1e-12:
        raise ValueError(f"Noise power too low after DC removal: RMS = {noise_rms_val.item():.2e}")

    snr_linear = 10.0 ** (-snr_db / 20.0)
    target_noise_rms = clean_rms_val * snr_linear
    noise_scaled = noise * (target_noise_rms / (noise_rms_val + 1e-12))

    degraded = clean + noise_scaled
    achieved_snr = calculate_snr_db(clean, degraded)

    if avoid_clipping:
        degraded = peak_normalize(degraded, peak=0.95)

    return degraded.to(torch.float32), achieved_snr


def lowpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    order: int = 8,
) -> torch.Tensor:
    """
    Apply a Butterworth low-pass filter using scipy SOS.

    Uses zero-phase filtfilt when input is long enough; falls back to
    causal sosfilt otherwise with a warning.

    Args:
        waveform: 1D float32 tensor
        sample_rate: sample rate in Hz
        cutoff_hz: cutoff frequency in Hz
        order: Butterworth filter order (default 8)

    Returns:
        filtered waveform, same length as input

    Raises:
        ValueError: if cutoff >= Nyquist
    """
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz ({cutoff_hz}) must be below Nyquist ({nyquist})")

    x = waveform.detach().cpu().numpy().astype(np.float64)

    sos = scipy.signal.butter(order, cutoff_hz, btype="low", output="sos", fs=sample_rate)

    pad_len = 3 * max(len(sos), order)
    if len(x) >= pad_len:
        x_filtered = scipy.signal.sosfiltfilt(sos, x)
    else:
        warnings.warn(
            f"Input too short ({len(x)} samples) for zero-phase filtfilt; using causal sosfilt"
        )
        x_filtered = scipy.signal.sosfilt(sos, x)

    result = torch.from_numpy(x_filtered.astype(np.float32))

    return result[:waveform.numel()]


def resample_roundtrip(
    waveform: torch.Tensor,
    source_rate: int,
    intermediate_rate: int,
) -> torch.Tensor:
    """
    Perform downsampling then upsampling back to source rate.

    source_rate -> intermediate_rate -> source_rate

    Uses scipy.signal.resample_poly.

    Args:
        waveform: 1D float32 tensor
        source_rate: original sample rate (Hz)
        intermediate_rate: intermediate sample rate (Hz)

    Returns:
        waveform at source_rate, exactly same length as input

    Raises:
        ValueError: if rates are invalid
    """
    if intermediate_rate >= source_rate:
        raise ValueError(
            f"intermediate_rate ({intermediate_rate}) must be less than source_rate ({source_rate})"
        )

    x = waveform.detach().cpu().numpy().astype(np.float64)
    orig_len = len(x)

    # Downsample
    gcd_down = np.gcd(source_rate, intermediate_rate)
    x_down = scipy.signal.resample_poly(
        x, intermediate_rate // gcd_down, source_rate // gcd_down
    )

    # Upsample back
    gcd_up = np.gcd(intermediate_rate, source_rate)
    x_up = scipy.signal.resample_poly(
        x_down, source_rate // gcd_up, intermediate_rate // gcd_up
    )

    if len(x_up) > orig_len:
        x_up = x_up[:orig_len]
    elif len(x_up) < orig_len:
        x_up = np.pad(x_up, (0, orig_len - len(x_up)), mode="edge")

    return torch.from_numpy(x_up.astype(np.float32))


def apply_bandwidth_degradation(
    waveform: torch.Tensor,
    sample_rate: int,
    intermediate_rate: Optional[int] = None,
    cutoff_hz: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply bandwidth degradation by combining resampling and low-pass filtering.

    Processing order:
    1. If intermediate_rate set: resample down then up
    2. If cutoff_hz set: apply low-pass filter

    Args:
        waveform: 1D float32 tensor
        sample_rate: current sample rate
        intermediate_rate: optional intermediate rate for round-trip resampling
        cutoff_hz: optional low-pass cutoff frequency

    Returns:
        degraded waveform, same length and sample rate
    """
    result = waveform.clone().to(torch.float32)

    if intermediate_rate is not None:
        result = resample_roundtrip(result, sample_rate, intermediate_rate)

    if cutoff_hz is not None:
        result = lowpass_filter(result, sample_rate, cutoff_hz)

    return result


# ---------------------------------------------------------------------------
# Composite degradation
# ---------------------------------------------------------------------------

@dataclass
class CompositeDegradationConfig:
    """Configuration for composite degradation pipeline."""
    severity: str = "clean"
    sample_rate: int = 16000
    intermediate_rate: Optional[int] = None
    cutoff_hz: Optional[float] = None
    bit_depth: int = 16
    snr_db: Optional[float] = None
    add_noise: bool = False
    seed: int = 2026

    def __post_init__(self):
        if self.severity not in ("clean", "light", "medium", "heavy"):
            raise ValueError(f"Unknown severity: {self.severity}")


@dataclass
class DegradationResult:
    """Result of applying composite degradation."""
    degraded: torch.Tensor
    clean: torch.Tensor
    severity: str
    snr_target_db: Optional[float]
    snr_achieved_db: Optional[float]
    intermediate_rate: Optional[int]
    cutoff_hz: Optional[float]
    bit_depth: int
    bandwidth_class: int
    bit_class: int
    noise_present: bool
    degradation_mask: int
    seed: int
    sample_rate: int = 16000

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "duration": len(self.degraded) / self.sample_rate,
            "target_snr": self.snr_target_db,
            "achieved_snr": self.snr_achieved_db,
            "intermediate_rate": self.intermediate_rate,
            "cutoff_hz": self.cutoff_hz,
            "bit_depth": self.bit_depth,
            "bandwidth_class": self.bandwidth_class,
            "bit_class": self.bit_class,
            "degradation_mask": self.degradation_mask,
            "peak": float(self.degraded.abs().max().item()),
            "rms": float(rms(self.degraded).item()),
            "noise_present": self.noise_present,
        }


def _bit_depth_to_class(bit_depth: int) -> int:
    """Map bit depth to class index."""
    mapping = {16: 0, 12: 1, 10: 2, 8: 3}
    return mapping[bit_depth]


def _bandwidth_to_class(cutoff_hz: Optional[float]) -> int:
    """Map bandwidth cutoff to class index."""
    if cutoff_hz is None:
        return 0  # full band
    if cutoff_hz >= 6000:
        return 1  # 6 kHz
    if cutoff_hz >= 4000:
        return 2  # 4 kHz
    return 2


def _compute_degradation_mask(
    noise_present: bool,
    bandwidth_limited: bool,
    quantization_applied: bool,
) -> int:
    """
    Compute degradation bitmask.

    bit 0 (LSB): noise present (1 = present)
    bit 1: bandwidth limited (1 = limited)
    bit 2: quantization applied (1 = applied, i.e. bit_depth < 16 or truly quantized)

    Example:
        0b001 (1): noise only
        0b010 (2): bandwidth only
        0b100 (4): quantization only
        0b111 (7): all three
    """
    mask = 0
    if noise_present:
        mask |= (1 << 0)
    if bandwidth_limited:
        mask |= (1 << 1)
    if quantization_applied:
        mask |= (1 << 2)
    return mask


def apply_composite_degradation(
    clean: torch.Tensor,
    config: CompositeDegradationConfig,
    noise: Optional[torch.Tensor] = None,
) -> DegradationResult:
    """
    Apply a full composite degradation pipeline.

    Processing order (FIXED):
        clean -> bandwidth degradation -> quantization -> noise addition -> anti-clip normalize

    Args:
        clean: clean waveform, 1D float32
        config: degradation configuration
        noise: optional noise waveform (required if config.add_noise=True)

    Returns:
        DegradationResult containing degraded waveform and all labels

    Raises:
        ValueError: if noise is required but not provided
    """
    seed_everything(config.seed)

    clean = ensure_mono(clean).to(torch.float32)
    check_waveform(clean)

    result = clean.clone()

    # Step 1: Bandwidth degradation
    bw_limited = (config.intermediate_rate is not None) or (config.cutoff_hz is not None)
    if bw_limited:
        result = apply_bandwidth_degradation(
            result,
            config.sample_rate,
            intermediate_rate=config.intermediate_rate,
            cutoff_hz=config.cutoff_hz,
        )

    # Step 2: Quantization
    quant_applied = (config.bit_depth < 16)
    result = quantize_waveform(result, config.bit_depth)

    # Step 3: Noise addition
    snr_achieved = None
    if config.add_noise:
        if noise is None:
            raise ValueError("config.add_noise=True but no noise waveform provided")
        result, snr_achieved = add_noise_at_snr(
            result,
            noise,
            config.snr_db,
            avoid_clipping=True,
        )

    # Step 4: Final anti-clip normalization
    result = peak_normalize(result, peak=0.95)

    bandwidth_class = _bandwidth_to_class(config.cutoff_hz)
    bit_class = _bit_depth_to_class(config.bit_depth)
    degradation_mask = _compute_degradation_mask(
        noise_present=config.add_noise,
        bandwidth_limited=bw_limited,
        quantization_applied=quant_applied,
    )

    return DegradationResult(
        degraded=result.to(torch.float32),
        clean=clean,
        severity=config.severity,
        snr_target_db=config.snr_db,
        snr_achieved_db=snr_achieved,
        intermediate_rate=config.intermediate_rate,
        cutoff_hz=config.cutoff_hz,
        bit_depth=config.bit_depth,
        bandwidth_class=bandwidth_class,
        bit_class=bit_class,
        noise_present=config.add_noise,
        degradation_mask=degradation_mask,
        seed=config.seed,
        sample_rate=config.sample_rate,
    )


# ---------------------------------------------------------------------------
# Severity presets
# ---------------------------------------------------------------------------

def sample_degradation_config(
    severity: str,
    seed: int,
) -> CompositeDegradationConfig:
    """
    Return a CompositeDegradationConfig for a given severity level.

    Supported severities:
        clean:   no degradation (16-bit passthrough)
        light:   12 kHz resample, 6 kHz low-pass, 10-12 bit
        medium:  8 kHz resample, 4 kHz low-pass, 10-12 bit, 15-25 dB noise
        heavy:   8 kHz resample, 4 kHz low-pass, 8-bit, 5-10 dB noise

    Args:
        severity: one of "clean", "light", "medium", "heavy"
        seed: random seed for reproducible random parameter selection

    Returns:
        CompositeDegradationConfig
    """
    seed_everything(seed)
    rng = random.Random(seed)

    if severity == "clean":
        return CompositeDegradationConfig(
            severity="clean",
            sample_rate=16000,
            intermediate_rate=None,
            cutoff_hz=None,
            bit_depth=16,
            snr_db=None,
            add_noise=False,
            seed=seed,
        )

    if severity == "light":
        bit_depth = rng.choice([10, 12])
        return CompositeDegradationConfig(
            severity="light",
            sample_rate=16000,
            intermediate_rate=12000,
            cutoff_hz=6000.0,
            bit_depth=bit_depth,
            snr_db=None,
            add_noise=False,
            seed=seed,
        )

    if severity == "medium":
        bit_depth = rng.choice([10, 12])
        snr_db = rng.uniform(15.0, 25.0)
        return CompositeDegradationConfig(
            severity="medium",
            sample_rate=16000,
            intermediate_rate=8000,
            cutoff_hz=4000.0,
            bit_depth=bit_depth,
            snr_db=snr_db,
            add_noise=True,
            seed=seed,
        )

    if severity == "heavy":
        snr_db = rng.uniform(5.0, 10.0)
        return CompositeDegradationConfig(
            severity="heavy",
            sample_rate=16000,
            intermediate_rate=8000,
            cutoff_hz=4000.0,
            bit_depth=8,
            snr_db=snr_db,
            add_noise=True,
            seed=seed,
        )

    raise ValueError(f"Unknown severity: {severity}")
