"""
STFT/iSTFT compatibility utilities for GTCRN.

Provides wrappers around torch.stft/istft that handle the new PyTorch 2.x
complex tensor API, without global monkey-patching.
"""
import torch


def stft_to_ri(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    window: torch.Tensor | None = None,
    center: bool = True,
) -> torch.Tensor:
    """
    Compute STFT and return real-imaginary representation.

    Uses torch.stft(return_complex=True) then torch.view_as_real
    to get (..., F, T, 2) shape.

    Args:
        waveform: input audio tensor
        n_fft: FFT size
        hop_length: hop length
        win_length: window length
        window: window function tensor
        center: whether to center-pad the input

    Returns:
        tensor of shape (..., F, T, 2) where last dim is (real, imag)
    """
    if window is None:
        window = torch.hann_window(win_length)

    spec_complex = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    return torch.view_as_real(spec_complex)


def ri_to_istft(
    spec_ri: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    window: torch.Tensor | None = None,
    center: bool = True,
    length: int | None = None,
) -> torch.Tensor:
    """
    Convert real-imaginary spectrogram back to waveform via iSTFT.

    Converts (..., F, T, 2) to complex via torch.view_as_complex,
    then calls torch.istft.

    Args:
        spec_ri: tensor of shape (..., F, T, 2)
        n_fft: FFT size
        hop_length: hop length
        win_length: window length
        window: window function tensor (must match stft)
        center: whether input was center-padded
        length: target output length (optional)

    Returns:
        waveform tensor

    Raises:
        ValueError: if spec_ri last dim is not 2
    """
    if spec_ri.shape[-1] != 2:
        raise ValueError(
            f"Last dimension must be 2 (real, imag), got shape {spec_ri.shape}"
        )

    if window is None:
        window = torch.hann_window(win_length)

    spec_complex = torch.view_as_complex(spec_ri.contiguous())

    return torch.istft(
        spec_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )
