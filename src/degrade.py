"""Physics-informed forward degradation model.

Emulates a low-field MRI acquisition from a high-resolution scan by:
  1. Resolution loss via k-space (Fourier) center-cropping: keep only the
     central low-frequency block, drop the high-frequency edges, invert.
  2. Realistic noise via the Rician model, the distribution that MRI
     magnitude images follow at low SNR.

Everything here is numpy so it has no torch/GPU dependency and is easy to
reason about and test. Inputs/outputs are float images scaled to [0, 1].

Scope note: this captures the resolution and noise characteristics of a
low-field scan. It does NOT reproduce field-strength contrast changes, so it
is a proof-of-concept forward model, not a substitute for real low-field data.
"""

from __future__ import annotations

import numpy as np


def _to_kspace(img: np.ndarray) -> np.ndarray:
    """Image -> centered k-space (fftshift so DC is in the middle)."""
    return np.fft.fftshift(np.fft.fft2(img))


def _from_kspace(k: np.ndarray) -> np.ndarray:
    """Centered k-space -> image magnitude."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k)))


def kspace_truncate(img: np.ndarray, factor: int) -> np.ndarray:
    """Lower spatial resolution the MRI-correct way.

    Keep only the central 1/factor of k-space along each axis (the low
    frequencies) and zero the high-frequency edges, then invert. This yields a
    genuinely low-resolution image on the original grid, with correct MRI
    blurring rather than a generic spatial blur.

    Args:
        img: 2D float image, values in [0, 1].
        factor: downsampling factor (e.g. 4 keeps the central 1/4 per axis).
    Returns:
        Low-resolution image on the same grid, same shape as ``img``.
    """
    assert img.ndim == 2, "expect a single 2D slice"
    h, w = img.shape
    k = _to_kspace(img)
    mask = np.zeros_like(k, dtype=np.float32)
    ch, cw = h // 2, w // 2
    kh, kw = max(1, h // (2 * factor)), max(1, w // (2 * factor))
    mask[ch - kh:ch + kh, cw - kw:cw + kw] = 1.0
    return _from_kspace(k * mask)


def add_rician_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Rician noise, the noise model of MRI magnitude images.

    A Rician-distributed magnitude arises from independent Gaussian noise added
    to the real and imaginary channels before taking the magnitude:
        out = sqrt((img + n_real)^2 + n_imag^2),  n ~ N(0, sigma).
    At high SNR this approaches Gaussian; at low SNR (the low-field case) it is
    distinctly Rician, which is why we model it explicitly.

    Args:
        img: 2D float image in [0, 1].
        sigma: noise standard deviation (in the same [0, 1] intensity units).
        rng: numpy random Generator (pass a seeded one for reproducibility).
    """
    n_real = rng.normal(0.0, sigma, size=img.shape)
    n_imag = rng.normal(0.0, sigma, size=img.shape)
    return np.sqrt((img + n_real) ** 2 + n_imag ** 2)


def degrade(
    img: np.ndarray,
    factor: int = 4,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Full forward model: resolution loss then Rician noise.

    Args:
        img: 2D float image in [0, 1] (high-resolution reference).
        factor: k-space truncation factor (resolution loss). 1 = no loss.
        sigma: Rician noise level. 0 = no noise.
        rng: numpy Generator; a default seeded one is used if None.
        clip: clip the result back to [0, 1].
    Returns:
        Degraded low-resolution, noisy image, same shape as ``img``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = img
    if factor and factor > 1:
        out = kspace_truncate(out, factor)
    if sigma and sigma > 0:
        out = add_rician_noise(out, sigma, rng)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)
