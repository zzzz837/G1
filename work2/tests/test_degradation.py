"""
Unit tests for the composite degradation pipeline.
"""
import sys
sys.path.insert(0, ".")

import pytest
import torch
import numpy as np

from work2.data.random_utils import seed_everything
from work2.data.audio_utils import rms, check_waveform
from work2.data.degradation import (
    quantize_waveform,
    add_noise_at_snr,
    resample_roundtrip,
    lowpass_filter,
    apply_bandwidth_degradation,
    CompositeDegradationConfig,
    DegradationResult,
    apply_composite_degradation,
    sample_degradation_config,
    _compute_degradation_mask,
    _bit_depth_to_class,
    _bandwidth_to_class,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_sine(freq_hz: float, duration: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    t = torch.arange(0, int(sample_rate * duration), dtype=torch.float32) / sample_rate
    return torch.sin(2.0 * np.pi * freq_hz * t)


def make_noise(length: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(length, dtype=torch.float32) * 0.1


# ── Test 1: Quantization ──────────────────────────────────────────────────

class TestQuantization:
    @pytest.mark.parametrize("bit_depth", [8, 10, 12, 16])
    def test_shape_preserved(self, bit_depth):
        w = make_sine(1000, duration=0.5)
        q = quantize_waveform(w, bit_depth)
        assert q.shape == w.shape

    @pytest.mark.parametrize("bit_depth", [8, 10, 12, 16])
    def test_no_nan_inf(self, bit_depth):
        w = make_sine(1000, duration=0.5)
        q = quantize_waveform(w, bit_depth)
        assert not torch.isnan(q).any()
        assert not torch.isinf(q).any()

    def test_unique_levels_under_limit(self):
        w = make_sine(1000, duration=0.5)
        for bd in [8, 10, 12]:
            q = quantize_waveform(w, bd)
            levels = len(torch.unique(q))
            max_levels = 2 ** bd
            assert levels <= max_levels, f"bit_depth={bd}: {levels} > {max_levels}"

    def test_lower_bit_has_higher_error(self):
        w = make_sine(1000, duration=0.5)
        err8 = (w - quantize_waveform(w, 8)).abs().mean()
        err12 = (w - quantize_waveform(w, 12)).abs().mean()
        err16 = (w - quantize_waveform(w, 16)).abs().mean()
        assert err8 > err12, f"err8={err8} <= err12={err12}"
        assert err12 >= err16, f"err12={err12} < err16={err16}"

    def test_16bit_really_quantizes(self):
        w = torch.linspace(-1, 1, 100000, dtype=torch.float32)
        q = quantize_waveform(w, 16)
        assert not torch.equal(w, q)

    def test_invalid_bit_depth(self):
        w = make_sine(1000, duration=0.1)
        with pytest.raises(ValueError):
            quantize_waveform(w, 7)


# ── Test 2: SNR achievement ───────────────────────────────────────────────

class TestSNR:
    @pytest.mark.parametrize("target_snr", [5, 10, 20, 30])
    def test_snr_accuracy(self, target_snr):
        seed_everything(2026)
        clean = make_sine(1000, duration=2.0)
        noise = make_noise(len(clean))
        _, achieved = add_noise_at_snr(clean, noise, target_snr)
        assert abs(achieved - target_snr) < 0.5, f"target={target_snr}, achieved={achieved:.2f}"

    def test_zero_noise_raises(self):
        clean = make_sine(1000, duration=0.5)
        noise = torch.zeros(16000)
        with pytest.raises(ValueError):
            add_noise_at_snr(clean, noise, 10.0)

    def test_zero_clean_raises(self):
        clean = torch.zeros(16000)
        noise = make_noise(16000)
        with pytest.raises(ValueError):
            add_noise_at_snr(clean, noise, 10.0)


# ── Test 3: Resample length preservation ──────────────────────────────────

class TestResample:
    @pytest.mark.parametrize("src, inter", [(16000, 12000), (16000, 8000)])
    def test_length_preserved(self, src, inter):
        w = make_sine(1000, duration=1.0, sample_rate=src)
        out = resample_roundtrip(w, src, inter)
        assert len(out) == len(w), f"in={len(w)}, out={len(out)}"

    def test_invalid_rate(self):
        w = make_sine(1000, duration=0.5)
        with pytest.raises(ValueError):
            resample_roundtrip(w, 16000, 24000)


# ── Test 4: Low-pass filter effect ────────────────────────────────────────

class TestLowpass:
    def test_attenuation(self):
        seed_everything(2026)
        w1k = make_sine(1000, duration=2.0)
        w6k = make_sine(6000, duration=2.0)
        mix = w1k + w6k
        filtered = lowpass_filter(mix, 16000, 4000.0)

        rms_before_6k = rms(w6k).item()
        rms_after_6k = rms(
            lowpass_filter(w6k, 16000, 4000.0)
        ).item()

        attenuation_db = 20.0 * np.log10(rms_after_6k / (rms_before_6k + 1e-12))
        assert attenuation_db < -20.0, f"6 kHz attenuation only {attenuation_db:.1f} dB"

    def test_1k_preserved(self):
        w1k = make_sine(1000, duration=2.0)
        filtered = lowpass_filter(w1k, 16000, 4000.0)
        rms_before = rms(w1k).item()
        rms_after = rms(filtered).item()
        ratio = rms_after / rms_before
        assert ratio > 0.5, f"1 kHz RMS ratio = {ratio:.3f}"

    def test_cutoff_above_nyquist_raises(self):
        w = make_sine(1000, duration=0.5)
        with pytest.raises(ValueError):
            lowpass_filter(w, 16000, 9000.0)


# ── Test 5: Reproducibility ───────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_output(self):
        seed_everything(2026)
        w = make_sine(1000, duration=1.0)
        noise = make_noise(len(w))
        cfg = CompositeDegradationConfig(
            severity="medium",
            intermediate_rate=8000,
            cutoff_hz=4000.0,
            bit_depth=10,
            snr_db=20.0,
            add_noise=True,
            seed=2026,
        )

        r1 = apply_composite_degradation(w.clone(), cfg, noise.clone())
        seed_everything(2026)
        r2 = apply_composite_degradation(w.clone(), cfg, noise.clone())

        assert torch.equal(r1.degraded, r2.degraded)
        assert r1.to_dict() == r2.to_dict()

    def test_different_seed_different_output(self):
        seed_everything(2026)
        w = make_sine(1000, duration=1.0)
        noise1 = torch.randn(len(w), dtype=torch.float32) * 0.1
        seed_everything(42)
        noise2 = torch.randn(len(w), dtype=torch.float32) * 0.1

        assert not torch.equal(noise1, noise2)


# ── Test 6: Composite degradation labels ──────────────────────────────────

class TestDegradationLabels:
    def test_clean_labels(self):
        w = make_sine(1000, duration=0.5)
        cfg = sample_degradation_config("clean", 2026)
        result = apply_composite_degradation(w, cfg)
        assert result.severity == "clean"
        assert not result.noise_present
        assert result.bandwidth_class == 0
        assert result.bit_class == 0
        assert result.degradation_mask == 0

    def test_light_labels(self):
        w = make_sine(1000, duration=0.5)
        cfg = sample_degradation_config("light", 2026)
        result = apply_composite_degradation(w, cfg)
        assert result.severity == "light"
        assert not result.noise_present
        assert result.cutoff_hz == 6000.0
        assert result.intermediate_rate == 12000
        assert result.bandwidth_class == 1
        assert result.bit_class in (1, 2)  # 12-bit=1 or 10-bit=2

    def test_medium_labels(self):
        w = make_sine(1000, duration=0.5)
        noise = make_noise(len(w))
        cfg = sample_degradation_config("medium", 2026)
        result = apply_composite_degradation(w, cfg, noise)
        assert result.severity == "medium"
        assert result.noise_present
        assert result.cutoff_hz == 4000.0
        assert result.intermediate_rate == 8000
        assert result.bandwidth_class == 2
        assert 15.0 <= result.snr_target_db <= 25.0
        assert result.snr_achieved_db is not None

    def test_heavy_labels(self):
        w = make_sine(1000, duration=0.5)
        noise = make_noise(len(w))
        cfg = sample_degradation_config("heavy", 2026)
        result = apply_composite_degradation(w, cfg, noise)
        assert result.severity == "heavy"
        assert result.noise_present
        assert result.bit_depth == 8
        assert result.bit_class == 3

    def test_degradation_mask_noise_only(self):
        mask = _compute_degradation_mask(True, False, False)
        assert mask == 1  # 0b001

    def test_degradation_mask_bandwidth_only(self):
        mask = _compute_degradation_mask(False, True, False)
        assert mask == 2  # 0b010

    def test_degradation_mask_quant_only(self):
        mask = _compute_degradation_mask(False, False, True)
        assert mask == 4  # 0b100

    def test_degradation_mask_all(self):
        mask = _compute_degradation_mask(True, True, True)
        assert mask == 7  # 0b111


# ── Test 7: Invalid inputs ───────────────────────────────────────────────

class TestInvalidInputs:
    def test_empty_waveform(self):
        w = torch.tensor([], dtype=torch.float32)
        with pytest.raises(ValueError):
            check_waveform(w)

    def test_cutoff_above_nyquist_raises(self):
        w = make_sine(1000, duration=0.5)
        with pytest.raises(ValueError):
            lowpass_filter(w, 16000, 9000.0)

    def test_invalid_bit_depth(self):
        w = make_sine(1000, duration=0.1)
        with pytest.raises(ValueError):
            quantize_waveform(w, 7)

    def test_noise_required_but_missing(self):
        w = make_sine(1000, duration=0.5)
        cfg = CompositeDegradationConfig(
            severity="medium", add_noise=True, snr_db=10.0, seed=2026
        )
        with pytest.raises(ValueError):
            apply_composite_degradation(w, cfg, noise=None)

    def test_invalid_severity(self):
        with pytest.raises(ValueError):
            sample_degradation_config("invalid", 2026)
