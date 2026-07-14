"""
Unit tests for FrozenGTCRNFeatureExtractor.
"""
import sys; sys.path.insert(0, ".")
import pytest
import torch
from work2.models.frozen_gtcrn_extractor import FrozenGTCRNFeatureExtractor

CHECKPOINT = "checkpoints/model_trained_on_dns3.tar"


@pytest.fixture(scope="module")
def extractor():
    return FrozenGTCRNFeatureExtractor(checkpoint_path=CHECKPOINT, device="cpu")


class TestFeatureExtractor:
    def test_loads_checkpoint(self, extractor):
        for param in extractor.model.parameters():
            assert not param.requires_grad

    def test_all_params_frozen(self, extractor):
        for name, param in extractor.model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen"

    def test_forward_shape(self, extractor):
        spec = torch.randn(2, 257, 10, 2)
        out = extractor(spec)
        assert out["enhanced_spec"].shape == (2, 257, 10, 2)
        assert out["bottleneck"].shape[0] == 2
        assert out["bottleneck"].ndim == 4
        C = out["bottleneck"].shape[1]
        assert out["feature_mean"].shape == (2, C)
        assert out["feature_std"].shape == (2, C)
        assert out["stats"].shape == (2, 2 * C)

    def test_no_nan_inf(self, extractor):
        spec = torch.randn(1, 257, 10, 2)
        out = extractor(spec)
        for k, v in out.items():
            assert not v.isnan().any(), f"NaN in {k}"
            assert not v.isinf().any(), f"Inf in {k}"

    def test_stats_dim(self, extractor):
        d = extractor.get_stats_dim()
        assert d > 0
        assert d % 2 == 0

    def test_no_duplicate_hooks(self, extractor):
        initial_hooks = len(extractor.model.dpgrnn2._forward_hooks)
        spec = torch.randn(1, 257, 10, 2)
        _ = extractor(spec)
        _ = extractor(spec)
        final_hooks = len(extractor.model.dpgrnn2._forward_hooks)
        assert final_hooks == initial_hooks

    def test_invalid_input_raises(self, extractor):
        with pytest.raises(ValueError):
            extractor(torch.randn(1, 257, 10))
        with pytest.raises(ValueError):
            extractor(torch.randn(1, 257, 10, 3))

    def test_baseline_consistency(self, extractor):
        import soundfile as sf
        from work2.data.stft_utils import stft_to_ri, ri_to_istft
        from gtcrn import GTCRN

        ref_model = GTCRN().eval()
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        ref_model.load_state_dict(ckpt["model"])

        mix, _ = sf.read("test_wavs/mix.wav", dtype="float32")
        window = torch.hann_window(512).pow(0.5)
        spec = stft_to_ri(torch.from_numpy(mix), window=window)

        with torch.no_grad():
            ref_out = ref_model(spec.unsqueeze(0))[0]
            ours_out = extractor(spec.unsqueeze(0))["enhanced_spec"][0]

        max_err = (ref_out - ours_out).abs().max().item()
        assert max_err < 1e-4, f"Baseline mismatch: max_err={max_err:.2e}"
