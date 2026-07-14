# GTCRN Baseline Reproduction Report

## Phase 1: Work2 Adaptive Residual — Baseline Establishment

**Date**: 2026-07-10

---

## 1. Repository & Environment

| Item | Value |
|------|-------|
| Repository | `gtcrn-main` (official GTCRN ICASSP 2024) |
| Branch | `work2-adaptive-residual` |
| Git commit | `a87981c68f3c92eae023f00c1b07956e6b44bb1e` |
| Operating System | Windows 11 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12450HX |
| Python | 3.13.9 |
| PyTorch | 2.13.0+cpu |
| NumPy | 2.3.5 |
| SoundFile | 0.14.0 |
| Einops | 0.8.2 |
| CUDA available | **False** |

---

## 2. Checkpoint Loading

| Checkpoint | SHA256 |
|------------|--------|
| `checkpoints/model_trained_on_dns3.tar` | `a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79` |

DNS3 checkpoint loaded successfully with no errors.

---

## 3. Input Audio

| Property | Value |
|----------|-------|
| File | `test_wavs/mix.wav` |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Duration | 9.769 s |
| Samples | 156,302 |

---

## 4. Model Parameters

| Metric | Value |
|--------|-------|
| Total parameters | **48,245** |
| Trainable parameters | **23,669** |
| Non-trainable (ERB filters) | 24,576 |

These match the official reported values (48.2K total).

---

## 5. Inference Performance (CPU)

| Metric | Value |
|--------|-------|
| Device | CPU |
| Runs | 10 (after 3 warmup) |
| Mean inference time | **0.3141 s** |
| RTF | **0.0322** |

RTF << 1.0 indicates real-time capable inference on CPU.

---

## 6. Output Files

| File | Path | Samples |
|------|------|---------|
| Official output | `test_wavs/enh.wav` | 156,160 |
| Baseline output | `outputs/baseline/enh_dns3.wav` | 156,160 |
| Metrics JSON | `outputs/baseline/baseline_metrics.json` | — |

---

## 7. Output Consistency Check

| Metric | Value |
|--------|-------|
| Sample rate match | Yes |
| Length match | Yes |
| Max absolute error | **0.000000e+00** |
| Mean absolute error | **0.000000e+00** |

Outputs are **bit-exact identical** (error = 0). The compatibility wrapper patches `torch.istft` to auto-convert `(..., 2)` real tensors to complex, producing identical results to the official PyTorch 1.11 behavior.

---

## 8. Issues & Resolutions

### Issue 1: PyTorch API Incompatibility

**Problem**: `torch.istft()` in PyTorch 2.13 no longer accepts the `return_complex=False` parameter or `(F, T, 2)` real-valued tensors. The official `infer.py` written for PyTorch 1.11 fails with:
```
RuntimeError: istft requires a complex-valued input tensor
```

**Resolution**: Created `work2/scripts/run_infer.py` — a compatibility wrapper that monkey-patches `torch.istft` to auto-convert `(..., 2)` tensors to complex using `torch.view_as_complex()`. The wrapper runs the official `infer.py` via `runpy.run_path()`.

**Impact**: Zero — outputs are bit-exact identical as confirmed by test.

### Issue 2: OpenMP Library Conflict

**Problem**: NumPy and PyTorch both load `libiomp5md.dll`, causing:
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Resolution**: Set `KMP_DUPLICATE_LIB_OK=TRUE` in all scripts. This allows both libraries to coexist without crashes.

### Issue 3: STFT Deprecation Warning

`torch.stft(return_complex=False)` is deprecated in PyTorch 2.x. The function still returns the old `(F, T, 2)` format with a warning. This can be addressed in Phase 2 by migrating to the new API.

### Issue 4: Packages Version Drift

| Package | Required | Installed |
|---------|----------|-----------|
| torch | 1.11.0 | 2.13.0+cpu |
| numpy | 1.24.4 | 2.3.5 |
| einops | 0.7.0 | 0.8.2 |
| soundfile | 0.12.1 | 0.14.0 |
| ptflops | 0.7 | 0.7.5 |

Older torch versions are incompatible with Python 3.13. All newer versions work correctly.

---

## 9. Phase 1 Pass Criteria

| Criterion | Status |
|-----------|--------|
| Official `infer.py` runs successfully | PASS |
| `test_wavs/enh.wav` generated | PASS |
| `outputs/baseline/enh_dns3.wav` generated | PASS |
| `baseline_metrics.json` exists and complete | PASS |
| Max output error < 1e-4 | PASS (0.0) |
| `torch.cuda.is_available() == False` | PASS |
| Official `gtcrn.py`, `loss.py`, `infer.py` unmodified | PASS |

**Phase 1 result: PASSED**

---

## 10. Recommendations for Next Phase

1. Migrate STFT calls to the new PyTorch API (`return_complex=True` + `torch.view_as_real`) to eliminate deprecation warnings.
2. Move all STFT/iSTFT operations into a utility module to centralize compatibility handling.
3. Set `KMP_DUPLICATE_LIB_OK=TRUE` in environment setup scripts.
4. Consider pinning `torch >= 2.0` in future `requirements.txt` since Python 3.13 requires it.

---

## 11. Official File Checksums (for integrity verification)

```
gtcrn.py: 20bf0610702506fa45cd543454b84bbeb862d73c7c20a3f8d820fdc04da6099b
loss.py: 9156109b71c3a4cb27c1688b4b9798758fdbc65269b33514edacfbebd16f3948
infer.py: 1061bb6a9a6f1a27423f7603871e683aaf0cd3ffd40a75efa9f9ba0965330d7b
```
