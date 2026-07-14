"""
Frozen GTCRN feature extractor.

Loads official GTCRN with DNS3 weights, freezes all parameters, and uses
a forward hook to capture the dpgrnn2 bottleneck output. Returns enhanced
spectrogram, bottleneck features, and statistical (mean+std) features.
"""
import torch
import torch.nn as nn
from typing import Optional


class FrozenGTCRNFeatureExtractor(nn.Module):
    """
    Wraps official GTCRN, freezes it, and extracts bottleneck statistics.

    Captures the output of model.dpgrnn2 via a forward hook.
    Stats = concat(mean(bottleneck, dim=(T,F)), std(bottleneck, dim=(T,F)), dim=1)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        import sys
        import os as _os
        _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        from gtcrn import GTCRN

        self.device = torch.device(device)
        self.model = GTCRN().to(self.device).eval()

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        for param in self.model.parameters():
            param.requires_grad_(False)

        if not hasattr(self.model, "dpgrnn2"):
            raise AttributeError(
                "GTCRN model does not have attribute 'dpgrnn2'. "
                "Cannot capture bottleneck features."
            )

        self._bottleneck: Optional[torch.Tensor] = None
        self._hook_handle = self.model.dpgrnn2.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._bottleneck = output.detach().clone()

    @torch.no_grad()
    def forward(
        self,
        spec_ri: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through frozen GTCRN, capturing bottleneck.

        Args:
            spec_ri: (B, F, T, 2) real-imag spectrogram

        Returns:
            dict with keys:
                enhanced_spec: (B, F, T, 2) enhanced STFT
                bottleneck: (B, C, T, F) dpgrnn2 output
                feature_mean: (B, C) mean over (T, F) of bottleneck
                feature_std: (B, C) std over (T, F) of bottleneck
                stats: (B, 2*C) concatenated mean & std
        """
        if spec_ri.dim() != 4 or spec_ri.shape[-1] != 2:
            raise ValueError(
                f"Expected spec_ri shape (B, F, T, 2), got {spec_ri.shape}"
            )

        if spec_ri.isnan().any() or spec_ri.isinf().any():
            raise ValueError("Input spec_ri contains NaN or Inf")

        spec_ri = spec_ri.to(self.device)

        self._bottleneck = None

        enhanced_spec = self.model(spec_ri)

        if self._bottleneck is None:
            raise RuntimeError(
                "Bottleneck not captured. Hook may have failed on dpgrnn2."
            )

        bottleneck = self._bottleneck

        feature_mean = bottleneck.mean(dim=(2, 3))
        feature_std = bottleneck.std(dim=(2, 3), unbiased=False)
        stats = torch.cat([feature_mean, feature_std], dim=1)

        for name, tensor in [
            ("enhanced_spec", enhanced_spec),
            ("bottleneck", bottleneck),
            ("feature_mean", feature_mean),
            ("feature_std", feature_std),
            ("stats", stats),
        ]:
            if tensor.isnan().any() or tensor.isinf().any():
                raise RuntimeError(f"Output '{name}' contains NaN or Inf")

        return {
            "enhanced_spec": enhanced_spec,
            "bottleneck": bottleneck,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "stats": stats,
        }

    def get_stats_dim(self) -> int:
        """Return dimensionality of the stats vector by running one dummy forward."""
        dummy = torch.randn(1, 257, 10, 2, device=self.device)
        out = self.forward(dummy)
        return int(out["stats"].shape[1])

    def __del__(self):
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
