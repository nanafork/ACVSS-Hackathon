"""Generate notebooks/tumor_aware_sr.ipynb (valid JSON) from cell definitions."""
import json
import os

CELLS = [
    ("md", """# Tumor-Aware MRI Super-Resolution: a Safety Study

Pipeline: **degrade** a high-res brain scan (k-space truncation + Rician noise) ->
**super-resolve** with two objectives (distortion-optimal vs tumor-aware) ->
**segment** the tumor -> measure image quality, segmentation Dice, and the
**safety rates** (lesion erasure, hallucination), plus **MC-dropout uncertainty**
and a **CPU benchmark**.

**How to run on Kaggle**
1. Add this repository to the notebook (upload as a Dataset, or `git clone`) so
   that the `src/` package is importable.
2. Attach a BraTS dataset (e.g. the BraTS2020 subset on Kaggle, no registration).
3. Set `DATA_ROOT` and `DATA_KIND = "brats"` below. Leave `"synthetic"` to run
   with no download.
4. Run all. Turn on the GPU accelerator for faster training."""),

    ("code", """import sys, os
# Make the src/ package importable whether the repo is the working dir or added
# as a Kaggle dataset under /kaggle/input/.
for p in [".", "..", "/kaggle/working", "/kaggle/input"]:
    if os.path.isdir(os.path.join(p, "src")):
        sys.path.insert(0, p); break
import torch, numpy as np
from src.data import make_dataset
from src.models import seg_unet, sr_unet, count_params
from src.losses import make_sr_loss
from src.train import train_segmenter, train_sr
from src.evaluate import evaluate_pipeline, format_results
from src.uncertainty import mc_predict, cpu_benchmark
from src.degrade import degrade
from src.figures import comparison_figure, uncertainty_figure
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)"""),

    ("md", "## Config"),
    ("code", """DATA_KIND = "synthetic"     # "brats" for real data, "synthetic" for a no-download demo
DATA_ROOT = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SIZE      = 128
FACTOR    = 4          # k-space truncation (resolution loss)
SIGMA     = 0.03       # Rician noise level
EPOCHS_SEG = 8
EPOCHS_SR  = 12
TUMOR_WEIGHT = 30.0    # lesion up-weighting for the tumor-aware loss
torch.manual_seed(0)"""),

    ("md", "## Data\\nBraTS provides both the high-res images to degrade and the tumor masks for the downstream step."),
    ("code", """if DATA_KIND == "brats":
    full = make_dataset("brats", root=DATA_ROOT, modality="t1c", size=SIZE,
                        slices_per_case=12, min_tumor_pixels=20)
    n_test = max(8, len(full) // 5)
    test_ds  = torch.utils.data.Subset(full, range(n_test))
    train_ds = torch.utils.data.Subset(full, range(n_test, len(full)))
else:
    train_ds = make_dataset("synthetic", n=200, size=SIZE, seed=1)
    test_ds  = make_dataset("synthetic", n=40,  size=SIZE, seed=999)
print("train", len(train_ds), "test", len(test_ds))"""),

    ("md", "## 1. Segmenter (trained on clean high-res, then frozen)"),
    ("code", """seg = seg_unet(base=32)
print("seg params:", f"{count_params(seg):,}")
train_segmenter(seg, train_ds, epochs=EPOCHS_SEG, bs=8, device=DEVICE)
for p in seg.parameters(): p.requires_grad_(False)"""),

    ("md", "## 2. Two super-resolution models\\nSame architecture, same data, **different objective**. Aim to match PSNR/SSIM so the safety comparison is fair."),
    ("code", """sr_d = sr_unet(base=32, dropout=0.2)   # distortion-optimal (pixel L1)
sr_t = sr_unet(base=32, dropout=0.2)   # tumor-aware (lesion-weighted L1)
loss_d = make_sr_loss("distortion")
loss_t = make_sr_loss("tumor_aware", weight=TUMOR_WEIGHT)
train_sr(sr_d, train_ds, loss_d, factor=FACTOR, sigma=SIGMA, epochs=EPOCHS_SR, bs=8, device=DEVICE, tag="sr-distortion")
train_sr(sr_t, train_ds, loss_t, factor=FACTOR, sigma=SIGMA, epochs=EPOCHS_SR, bs=8, device=DEVICE, tag="sr-tumor-aware")"""),

    ("md", "## 3. Evaluation: the safety comparison"),
    ("code", """results = evaluate_pipeline({"distortion": sr_d, "tumor_aware": sr_t}, seg, test_ds,
                            factor=FACTOR, sigma=SIGMA, device=DEVICE, mc_passes=10,
                            size_edges=(50, 200))
print(format_results(results))"""),

    ("md", "## 4. Figures: comparison panel and uncertainty map"),
    ("code", """sample = test_ds[0]
hr = sample["hr"][None].to(DEVICE); mask = sample["mask"][None].to(DEVICE)
lr = torch.from_numpy(degrade(hr[0,0].cpu().numpy(), factor=FACTOR, sigma=SIGMA))[None,None].to(DEVICE)
sr_d.eval(); sr_t.eval()
with torch.no_grad():
    outs = {"distortion": sr_d(lr), "tumor_aware": sr_t(lr)}
comparison_figure(hr, mask, lr, outs, seg, save_path="comparison.png")
mean, unc = mc_predict(sr_t, lr, passes=15)
uncertainty_figure(lr, mean, unc, hr, save_path="uncertainty.png")"""),

    ("md", "## 5. CPU deployment benchmark"),
    ("code", """b = cpu_benchmark(sr_d, size=SIZE, reps=20)
print(f"CPU inference: {b['latency_ms']:.1f} ms/slice | params {b['params']:,} | {b['param_memory_mb']:.2f} MB")"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": src.split("\n")})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                          "execution_count": None, "source": src.split("\n")})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "tumor_aware_sr.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
