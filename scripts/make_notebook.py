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

**How to run on Kaggle with a GPU**
1. New Notebook, then in the right-hand panel set **Accelerator = GPU T4 x2**
   and **Internet = On**. (GPU requires a phone-verified Kaggle account.)
2. Run the clone cell below. It pulls the current branch, so you get the same
   code as the local demo.
3. **+ Add Input** and search `brats20-dataset-training-validation`, then add
   it. That mirror needs no registration.
4. Set `DATA_KIND = "brats"` in the config cell. Leave `"synthetic"` to run
   with no download.
5. Run All. On a T4 the synthetic run is a few minutes; BraTS is longer.

The last cell saves `checkpoints/demo.pt`. Download it from the notebook's
Output tab and drop it into `checkpoints/` locally, then `python main_demo.py`
regenerates the deck from the GPU-trained weights."""),

    ("code", """# Clone the repo so the src/ package is importable. Skip if you added it as
# a Kaggle Dataset instead. Private repo: use a token, or upload as a Dataset.
REPO = "https://github.com/nanafork/ACVSS-Hackathon.git"
BRANCH = "neuro-voxel-3d-viz"
import os
if not os.path.isdir("/kaggle/working/ACVSS-Hackathon") and os.path.isdir("/kaggle"):
    !git clone --depth 1 --branch $BRANCH $REPO /kaggle/working/ACVSS-Hackathon
    %cd /kaggle/working/ACVSS-Hackathon
!pip -q install nibabel"""),

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
# For the African population subset (BraTS-Africa / BraTS-SSA), attach that dataset
# and point DATA_ROOT at its training folder instead. The pipeline is identical:
# it takes high-resolution scans and degrades them to imitate a low-field scanner.
# DATA_ROOT = "/kaggle/input/brats-africa/.../TrainingData"
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

    ("md", """## 6. Upload the trained weights to Hugging Face

The person running this notebook pastes their own Hugging Face token. Create one
with **write** access at https://huggingface.co/settings/tokens. The token is
read through a hidden prompt and is not stored in the notebook or the repo."""),
    ("code", """import getpass
from huggingface_hub import HfApi, login
from src.checkpoint import save_models

# 1. Save the three trained models to one checkpoint.
CKPT_PATH = "checkpoints/demo.pt"
save_models(CKPT_PATH, seg, sr_d, sr_t,
            meta={"size": SIZE, "factor": FACTOR, "sigma": SIGMA,
                  "tumor_weight": TUMOR_WEIGHT, "data_kind": DATA_KIND})

# 2. Log in with the token entered at the prompt.
HF_TOKEN = getpass.getpass("Hugging Face token (write access): ").strip()
login(token=HF_TOKEN)

# 3. Create the model repo under the account that owns the token and upload.
api = HfApi()
username = api.whoami()["name"]
HF_REPO = f"{username}/tumor-aware-sr-weights"   # rename here if you prefer
api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=CKPT_PATH, path_in_repo="demo.pt",
                repo_id=HF_REPO, repo_type="model")
print(f"Uploaded weights to https://huggingface.co/{HF_REPO}")"""),

    ("md", "### Load the weights back later (for inference or the 3D demo)"),
    ("code", """# from huggingface_hub import hf_hub_download
# from src.checkpoint import load_models
# path = hf_hub_download(HF_REPO, "demo.pt")   # HF_REPO = "<username>/tumor-aware-sr-weights"
# seg, sr_d, sr_t, meta = load_models(path, device=DEVICE)"""),
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
