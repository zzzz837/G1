"""
Compatibility wrapper to run official infer.py with PyTorch 2.x.
Uses stft_utils to patch torch.istft for (..., 2) -> complex conversion.
Does NOT modify official files.

Note: This module temporarily patches torch.istft because the official
infer.py calls torch.istft directly. The patch is localized to the
runpy execution scope and does not persist after this script exits.
"""
import sys
import os
import runpy

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from work2.data.stft_utils import ri_to_istft

_original_istft = torch.istft

def _compat_istft(input_tensor, *args, **kwargs):
    kwargs.pop("return_complex", None)
    if not torch.is_complex(input_tensor) and input_tensor.shape[-1] == 2:
        input_tensor = torch.view_as_complex(input_tensor.contiguous())
    return _original_istft(input_tensor, *args, **kwargs)

torch.istft = _compat_istft

try:
    os.chdir(REPO_ROOT)
    runpy.run_path(os.path.join(REPO_ROOT, "infer.py"), run_name="__main__")
finally:
    torch.istft = _original_istft
