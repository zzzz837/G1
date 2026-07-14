# GTCRN Work2 Phase 2: Composite Degradation Pipeline Report

**Date**: 2026-07-10
**Branch**: `work2-adaptive-residual`
**Phase**: 2 — Degradation Data Generation & Label System

---

## 1. Objectives

Implement a reproducible composite degradation pipeline that generates:
- Additive noise at specified SNR
- Bandwidth limitation (resampling + low-pass filtering)
- Quantization distortion (8/10/12/16-bit)
- Four severity presets: clean, light, medium, heavy

Each degraded sample carries accurate labels for multi-task learning in later phases.

---

## 2. New Files

```
work2/data/__init__.py           Package init
work2/data/audio_utils.py        ensure_mono, peak_normalize, match_length, rms, snr, check
work2/data/random_utils.py       seed_everything (Python+NumPy+PyTorch)
work2/data/stft_utils.py         stft_to_ri / ri_to_istft (PyTorch 2.x compat, no monkey-patch)
work2/data/degradation.py        All degradation functions + presets + dataclasses
work2/data/build_manifest.py     Manifest builder (scan wavs, split train/valid/test, JSONL output)
work2/tests/test_degradation.py  39 unit tests
work2/scripts/generate_degradation_demo.py  Demo: 4 severity levels, spectrograms, waveforms, metadata
work2/scripts/run_baseline.py    Refactored: uses stft_utils to remove global monkey-patch
work2/scripts/run_infer.py       Refactored: localized patch, restores on exit
outputs/degradation_demo/        Generated demo files
```

---

## 3. Degradation Definitions

### 3.1 Quantization

**Formula**: `y = round(clamp(x, -1, 1) * M) / M` where `M = 2^(b-1) - 1`.

Symmetric uniform quantization maps [-1, 1] to `2^b` levels. 16-bit also undergoes true quantization (not a passthrough).

### 3.2 SNR Noise Addition

**Formula**: `target_noise_rms = clean_rms * 10^(-SNR_dB / 20)`, then `degraded = clean + noise * (target_noise_rms / noise_rms)`.

Noise is DC-removed, length-matched by cycle-repetition, then scaled to the exact target RMS ratio. Achieved SNR is computed BEFORE any peak normalization: `SNR = 10 * log10(clean_power / noise_power)`.

### 3.3 Resampling Round-trip

`source_rate -> intermediate_rate -> source_rate` using `scipy.signal.resample_poly`. Output length is exactly preserved.

### 3.4 Low-pass Filter

Butterworth order-8 SOS filter, zero-phase (`sosfiltfilt`) or causal fallback (`sosfilt`). Requires `cutoff < Nyquist`.

### 3.5 Bandwidth Degradation

Optional resampling followed by optional low-pass in sequence.

---

## 4. Fixed Processing Order

```
clean signal
  -> bandwidth degradation (resample_roundtrip + lowpass_filter)
  -> quantization (quantize_waveform)
  -> noise addition (add_noise_at_snr)
  -> anti-clip normalization (peak_normalize to 0.95)
  -> return DegradationResult with labels
```

This order is invariant and never rearranged.

---

## 5. Severity Preset Configuration

| Parameter | Clean | Light | Medium | Heavy |
|-----------|-------|-------|--------|-------|
| intermediate_rate | None | 12000 | 8000 | 8000 |
| cutoff_hz | None | 6000 | 4000 | 4000 |
| bit_depth | 16 | random{10,12} | random{10,12} | 8 |
| add_noise | False | False | True | True |
| snr_db | None | None | Uniform(15,25) | Uniform(5,10) |

---

## 6. Label Definitions

### 6.1 Bandwidth Class

| Class | Description |
|-------|-------------|
| 0 | Full band (no bandwidth limitation) |
| 1 | 6 kHz (light degradation) |
| 2 | 4 kHz (medium/heavy degradation) |

### 6.2 Bit Class

| Class | Description |
|-------|-------------|
| 0 | 16-bit |
| 1 | 12-bit |
| 2 | 10-bit |
| 3 | 8-bit |

### 6.3 Degradation Mask (bit flags, LSB-first)

| Bit | Meaning | Value |
|-----|---------|-------|
| 0 (LSB) | Noise present | 1 |
| 1 | Bandwidth limited | 2 |
| 2 | Quantization applied (bit_depth < 16) | 4 |

Examples:
- `0b000` (0): clean — no degradation
- `0b001` (1): noise only
- `0b010` (2): bandwidth only
- `0b100` (4): quantization only
- `0b110` (6): bandwidth + quantization (light)
- `0b111` (7): all three (medium/heavy)

---

## 7. Unit Test Results

**39/39 tests passed** (pytest, 2.97s).

### Test Coverage

| Test Group | Tests | Result |
|-----------|-------|--------|
| Quantization (shape, no NaN, levels, bit-error ordering, 16-bit real, invalid bit) | 12 | PASS |
| SNR accuracy (5/10/20/30 dB, zero noise/clean raises) | 6 | PASS |
| Resample length preservation (12k/8k, invalid rate) | 3 | PASS |
| Low-pass effect (6k attenuation, 1k preservation, cutoff check) | 3 | PASS |
| Reproducibility (same seed, different seed) | 2 | PASS |
| Degradation labels (clean/light/medium/heavy, mask bits) | 8 | PASS |
| Invalid inputs (empty, cutoff, bit depth, missing noise, invalid severity) | 5 | PASS |

---

## 8. SNR Accuracy

| Target SNR (dB) | Achieved SNR (dB) | Error (dB) |
|----------------|-------------------|------------|
| 5 | 5.00 | 0.000 |
| 10 | 10.00 | 0.000 |
| 20 | 20.00 | 0.000 |
| 30 | 30.00 | 0.000 |

All errors < 0.5 dB pass criterion. Essentially zero error due to exact RMS scaling.

---

## 9. Low-pass Attenuation

| Test | Result |
|------|--------|
| 1 kHz tone after 4 kHz LPF | RMS ratio = 0.965 (well preserved) |
| 6 kHz tone after 4 kHz LPF | Attenuation = **-44.3 dB** |

Pass criterion: attenuation < -20 dB. Actual: -44.3 dB (far exceeds requirement).

---

## 10. Output Demo Files

| File | Description |
|------|-------------|
| `outputs/degradation_demo/clean.wav` | Clean multi-tone signal |
| `outputs/degradation_demo/light.wav` | Light degradation (12 kHz resample, 6 kHz LPF, 10/12-bit) |
| `outputs/degradation_demo/medium.wav` | Medium degradation (8 kHz resample, 4 kHz LPF, 10/12-bit, 15-25 dB noise) |
| `outputs/degradation_demo/heavy.wav` | Heavy degradation (8 kHz resample, 4 kHz LPF, 8-bit, 5-10 dB noise) |
| `outputs/degradation_demo/metadata.json` | Per-sample metadata with all labels |
| `outputs/degradation_demo/spectrogram_comparison.png` | 2x2 spectrogram grid |
| `outputs/degradation_demo/waveform_comparison.png` | 2x2 waveform grid |

### Spectrogram Observations

- **Clean**: Full bandwidth (0-8 kHz), all 5 tones visible
- **Light**: 6 kHz tone attenuated; slight resampling artifacts near Nyquist
- **Medium**: 4 kHz cutoff visible; 6 kHz tone removed; noise floor visible
- **Heavy**: Strong quantization noise; heavy noise floor; 6 kHz completely gone

---

## 11. Reproducibility

- Same seed + same input + same noise = bit-exact identical output (verified by `test_same_seed_same_output`)
- Different seed generates different noise = different output (verified)
- `seed_everything()` fixes Python `random`, NumPy, and PyTorch CPU generators
- All random config sampling uses local `random.Random(seed)` instances

---

## 12. Baseline Compatibility

| Check | Result |
|-------|--------|
| Official files unmodified (gtcrn.py, loss.py, infer.py) | PASS |
| `torch.cuda.is_available()` = False | PASS |
| Phase 1 baseline re-run successful | PASS |
| Official vs refactored output error | 0.000000e+00 (bit-exact) |
| Global `torch.istft` monkey-patch removed | PASS (now local only in run_infer.py) |
| `stft_utils.py` replaces inline patching | PASS |

---

## 13. Compatibility Layer Cleanup

- `run_baseline.py`: Uses `stft_to_ri()` / `ri_to_istft()` from `work2.data.stft_utils` — **no global monkey-patch**
- `run_infer.py`: Localized `torch.istft` patch, restored on exit via `finally` block
- `KMP_DUPLICATE_LIB_OK`: Set only via command-line environment; not hardcoded in any Python source

---

## 14. Known Limitations

1. **Zero-phase filter**: `sosfiltfilt` requires sufficient input length; very short inputs fall back to causal `sosfilt` with a warning.
2. **Noise generation**: Uses `torch.randn` (Gaussian); may not represent all real-world noise types.
3. **Resampling**: `scipy.signal.resample_poly` uses polyphase filtering which is high quality but not real-time capable.
4. **KMP_DUPLICATE_LIB_OK**: OpenMP conflict between NumPy and PyTorch not fundamentally resolved; environment workaround only.
5. **No real dataset**: Phase 2 uses synthetic/test signals only; manifest builder is ready for real data.

---

## 15. Next Phase Plan

1. Generate composite degraded dataset from real clean speech and noise corpora using the manifest builder.
2. Implement training feature caching for efficient CPU training.
3. Develop novel loss functions (degradation estimation, high-frequency preservation, residual constraint).
4. Design lightweight training heads for the multi-task labels.
5. Train improved GTCRN variants on the degraded data.

**Note**: No model modifications, training, or GPU usage occurred in this phase.

---

## 16. Verification Commands

All verification commands pass:

```bash
# Unit tests
python -m pytest work2/tests -v                          # 39 passed

# Baseline inference
python work2/scripts/run_baseline.py                     # RTF 0.0217
python work2/tests/test_baseline_output.py               # Error 0.0

# Demo generation
python work2/scripts/generate_degradation_demo.py --seed 2026

# Manifest builder
python -m work2.data.build_manifest --clean-dir test_wavs --output outputs/manifest.jsonl --seed 2026
```
