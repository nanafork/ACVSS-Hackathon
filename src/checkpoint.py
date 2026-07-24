"""Save / load trained models for the demo."""

from __future__ import annotations

import os

import torch

from .models import seg_unet, sr_unet


def save_models(path: str, seg, sr_d, sr_t, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "seg": seg.state_dict(),
        "sr_distortion": sr_d.state_dict(),
        "sr_tumor_aware": sr_t.state_dict(),
        "meta": meta or {},
    }, path)


def load_models(path: str, base_seg: int = 32, base_sr: int = 32,
                dropout: float = 0.2, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    seg = seg_unet(base=base_seg)
    sr_d = sr_unet(base=base_sr, dropout=dropout)
    sr_t = sr_unet(base=base_sr, dropout=dropout)
    seg.load_state_dict(ckpt["seg"])
    sr_d.load_state_dict(ckpt["sr_distortion"])
    sr_t.load_state_dict(ckpt["sr_tumor_aware"])
    for m in (seg, sr_d, sr_t):
        m.to(device).eval()
    return seg, sr_d, sr_t, ckpt.get("meta", {})
