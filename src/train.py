"""Training loops for the segmentation and super-resolution networks.

Kept minimal and dependency-light. Batches come from a DataLoader yielding
dicts with 'hr' and 'mask'; the low-resolution input is produced on the fly by
the physics-informed degradation so we never store degraded copies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .degrade import degrade


def degrade_batch(hr: torch.Tensor, factor: int, sigma: float,
                  seed: int) -> torch.Tensor:
    """Apply the forward degradation to a (B,1,H,W) batch, returning LR."""
    out = torch.empty_like(hr)
    for i in range(hr.shape[0]):
        rng = np.random.default_rng(seed + i)
        img = hr[i, 0].cpu().numpy()
        out[i, 0] = torch.from_numpy(degrade(img, factor=factor, sigma=sigma, rng=rng))
    return out.to(hr.device)


def train_segmenter(model, dataset, epochs: int = 3, bs: int = 8, lr: float = 1e-3,
                    device: str = "cpu", log=print) -> None:
    """Train the segmentation U-Net on clean high-resolution images."""
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    for ep in range(epochs):
        tot = 0.0
        for batch in loader:
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            logits = model(hr)
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"[seg] epoch {ep + 1}/{epochs} loss {tot / len(loader):.4f}")


def train_sr(model, dataset, sr_loss, factor: int = 4, sigma: float = 0.02,
             epochs: int = 3, bs: int = 8, lr: float = 1e-3, device: str = "cpu",
             log=print, tag: str = "sr") -> None:
    """Train a super-resolution U-Net with the given objective.

    Args:
        sr_loss: callable(pred, target, mask) -> scalar (see losses.make_sr_loss).
        factor, sigma: degradation used to make LR inputs each step.
    """
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    step = 0
    for ep in range(epochs):
        tot = 0.0
        for batch in loader:
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            lr_in = degrade_batch(hr, factor, sigma, seed=1000 + step)
            step += 1
            pred = model(lr_in)
            loss = sr_loss(pred, hr, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"[{tag}] epoch {ep + 1}/{epochs} loss {tot / len(loader):.4f}")
