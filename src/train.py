"""Training loops for the segmentation and super-resolution networks.

Kept minimal and dependency-light. Batches come from a DataLoader yielding
dicts with 'hr' and 'mask'; the low-resolution input is produced on the fly by
the physics-informed degradation so we never store degraded copies.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .degrade import degrade, degrade_torch


def degrade_batch(hr: torch.Tensor, factor: int, sigma: float,
                  seed: int = 0, generator=None) -> torch.Tensor:
    """Apply the forward degradation to a (B,1,H,W) batch, returning LR.

    Runs the torch port, so the batch never leaves its device. Pass a
    ``generator`` (on the same device) for reproducible noise; ``seed`` is
    accepted for call-site compatibility with the numpy reference below.
    """
    return degrade_torch(hr, factor=factor, sigma=sigma, generator=generator)


def degrade_batch_numpy(hr: torch.Tensor, factor: int, sigma: float,
                        seed: int = 0) -> torch.Tensor:
    """Reference degradation via the numpy forward model, one sample at a time.

    Slower -- a CPU round trip per sample -- but seeded per sample and so
    exactly reproducible. Kept as the oracle the parity test checks against.
    """
    out = torch.empty_like(hr)
    for i in range(hr.shape[0]):
        rng = np.random.default_rng(seed + i)
        img = hr[i, 0].detach().cpu().numpy()
        out[i, 0] = torch.from_numpy(
            degrade(img, factor=factor, sigma=sigma, rng=rng)).to(hr.device)
    return out


def train_segmenter(model, dataset, epochs: int = 3, bs: int = 8, lr: float = 1e-3,
                    device: str = "cpu", log=print) -> None:
    """Train the segmentation U-Net on clean high-resolution images."""
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    for ep in range(epochs):
        tot = 0.0
        t0 = time.perf_counter()
        for batch in loader:
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            logits = model(hr)
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"[seg] epoch {ep + 1}/{epochs} loss {tot / len(loader):.4f} "
            f"({time.perf_counter() - t0:.1f}s)")


def train_sr(model, dataset, sr_loss, factor: int = 4, sigma: float = 0.02,
             epochs: int = 3, bs: int = 8, lr: float = 1e-3, device: str = "cpu",
             log=print, tag: str = "sr", seed: int = 1000) -> None:
    """Train a super-resolution U-Net with the given objective.

    Args:
        sr_loss: callable(pred, target, mask) -> scalar (see losses.make_sr_loss).
        factor, sigma: degradation used to make LR inputs each step.
        seed: seeds the degradation noise so a run is reproducible.
    """
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    gen = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        tot = 0.0
        t0 = time.perf_counter()
        for batch in loader:
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            lr_in = degrade_batch(hr, factor, sigma, generator=gen)
            pred = model(lr_in)
            loss = sr_loss(pred, hr, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"[{tag}] epoch {ep + 1}/{epochs} loss {tot / len(loader):.4f} "
            f"({time.perf_counter() - t0:.1f}s)")
