"""Quality vs safety curve.

Image-quality metrics (PSNR/SSIM) do not measure whether a super-resolution model
keeps the tumor. This script makes that visible: it sweeps the degradation
severity, and for each level it measures image quality (PSNR) against safety
(tumor erasure rate) for both the distortion-optimal and the tumor-aware model.

The tumor-aware curve should sit below the distortion curve, which means fewer
erased lesions at the same image quality.

    python scripts/quality_safety_curve.py        # writes quality_safety_curve.png

Runs on CPU with the demo checkpoint. If no checkpoint exists it is trained once
on synthetic data by demo.py's helper.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data import make_dataset
from src.degrade import degrade
from src.metrics import lesion_records, psnr, to_mask_np

CKPT = "checkpoints/demo.pt"
FACTORS = [2, 3, 4, 5, 6, 8]          # k-space truncation: higher means blurrier
DI_COLOR = "#B4553B"                  # distortion-optimal (RISE low)
TA_COLOR = "#2E7D5B"                  # tumor-aware (RISE good)


def _load():
    from src.checkpoint import load_models
    if not os.path.exists(CKPT):
        from demo import _ensure_models
        seg, sr_d, sr_t, size, factor, sigma = _ensure_models("cpu")
        return seg, sr_d, sr_t, size, sigma
    seg, sr_d, sr_t, meta = load_models(CKPT, device="cpu")
    return seg, sr_d, sr_t, int(meta.get("size", 96)), float(meta.get("sigma", 0.03))


def measure(seg, sr_model, dataset, factor, sigma, device="cpu", seed=0):
    """Return (mean PSNR, erasure rate) for one model at one degradation level."""
    rng = np.random.default_rng(seed)
    psnrs, total, erased = [], 0, 0
    sr_model.eval(); seg.eval()
    for i in range(len(dataset)):
        s = dataset[i]
        hr = s["hr"][None].to(device)
        gt = s["mask"][0].cpu().numpy()
        lr_np = degrade(hr[0, 0].cpu().numpy(), factor=factor, sigma=sigma, rng=rng)
        lr = torch.from_numpy(lr_np)[None, None].float().to(device)
        with torch.no_grad():
            sr = sr_model(lr)
            pred = to_mask_np(seg(sr))
        psnrs.append(psnr(sr, hr))
        recs = lesion_records(gt, pred)
        total += len(recs)
        erased += sum(1 for r in recs if not r["detected"])
    erasure_rate = 100.0 * erased / total if total else float("nan")
    return float(np.mean(psnrs)), erasure_rate


def build(out="quality_safety_curve.png", device="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seg, sr_d, sr_t, size, sigma = _load()
    ds = make_dataset("synthetic", n=40, size=size, seed=999, tumor_frac=1.0)

    curves = {"distortion": [], "tumor-aware": []}
    print(f"{'factor':>6} {'model':<12} {'PSNR(dB)':>9} {'erasure%':>9}")
    for factor in FACTORS:
        for name, model in [("distortion", sr_d), ("tumor-aware", sr_t)]:
            q, safety = measure(seg, model, ds, factor, sigma, device=device, seed=factor)
            curves[name].append((q, safety, factor))
            print(f"{factor:>6} {name:<12} {q:>9.2f} {safety:>9.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, color in [("distortion", DI_COLOR), ("tumor-aware", TA_COLOR)]:
        pts = sorted(curves[name], key=lambda p: p[2])   # order by degradation factor
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=color, label=f"{name} SR", linewidth=2, markersize=6)
        for q, safety, factor in pts:
            ax.annotate(f"x{factor}", (q, safety), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)
    ax.set_xlabel("image quality  (PSNR, dB)  ->  better")
    ax.set_ylabel("tumor erasure rate (%)  ->  less safe")
    ax.set_title("Quality is not safety: erasure vs image quality")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(0.02, 0.02,
            "At matched PSNR the tumor-aware curve is lower:\nfewer lesions erased for the same image quality.",
            transform=ax.transAxes, fontsize=8, color="#444", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build(device="cuda" if torch.cuda.is_available() else "cpu")
