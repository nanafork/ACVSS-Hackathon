"""Regenerate notebooks/colab_brats.ipynb.

The notebook is the Colab entry point for a real BraTS run. Editing it here and
regenerating keeps it diffable in git, the same convention make_notebook.py
uses for the Kaggle notebook.

    python scripts/make_colab_notebook.py
"""

import json
import os

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# Tumor-Aware MRI Super-Resolution — BraTS on Colab

Trains the full pipeline on real BraTS data with a GPU.

**Before you start:** set the runtime to a GPU via
*Runtime → Change runtime type → Hardware accelerator → GPU*.

Order of operations: check GPU → install deps → get the code → mount Drive →
point at your BraTS folder → build a cached slice dataset → train → evaluate.

The dataset build decodes every NIfTI volume once and caches the extracted
slices to Drive. If the runtime disconnects, re-running loads the cache in
seconds instead of re-decoding everything."""),

    (MD, "## 0. Environment"),
    (CODE, """!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU - set Runtime > Change runtime type > GPU"
!pip -q install nibabel"""),

    (MD, """## 1. Get the code

Public repo: a plain clone. Private: use a personal access token, or upload the
folder to Drive and set `REPO_DIR` to that path instead.

**Colab clones from GitHub, not from your laptop.** Anything you have not
committed *and pushed* does not exist here. The next cell checks that the code
it just fetched is new enough to run this notebook, so a stale branch fails
immediately with a clear message instead of a confusing `TypeError` five cells
later."""),
    (CODE, """import os, sys

REPO_URL = "https://github.com/nanafork/ACVSS-Hackathon.git"
BRANCH   = "my-feature-branch"
REPO_DIR = "/content/ACVSS-Hackathon"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    !rm -rf $REPO_DIR
    !git clone --branch $BRANCH $REPO_URL $REPO_DIR
else:
    !cd $REPO_DIR && git fetch origin && git checkout $BRANCH && git pull --ff-only

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
!git log --oneline -1"""),

    (CODE, """# Fail fast if the pushed branch predates the BraTS training support.
import importlib
import src.data, src.degrade, src.train
for m in (src.data, src.degrade, src.train):
    importlib.reload(m)

missing = []
if not hasattr(src.data.BraTSSliceDataset, "split_by_case"):
    missing.append("BraTSSliceDataset.split_by_case")
if not hasattr(src.degrade, "degrade_torch"):
    missing.append("degrade.degrade_torch")
if "cache_path" not in src.data.BraTSSliceDataset.__init__.__code__.co_varnames:
    missing.append("BraTSSliceDataset(cache_path=...)")

if missing:
    raise SystemExit(
        "The cloned branch is out of date - missing: " + ", ".join(missing) +
        f"\\n\\nOn your machine, commit and push to '{BRANCH}':"
        "\\n    git add -A && git commit -m 'BraTS training support' && git push"
        "\\n\\nThen re-run this cell (it will pull the update)."
    )
print("code version OK")"""),

    (MD, """## 2. Mount Drive

Drive holds the BraTS data and, more importantly, the slice cache and the
trained checkpoint — so a disconnect does not cost you the run."""),
    (CODE, """from google.colab import drive
drive.mount('/content/drive')"""),

    (MD, """## 3. Point at your BraTS data

`BRATS_ROOT` must be the folder that contains **one subfolder per case**, each
holding `*_t1c.nii.gz` (or `*_t1ce.nii.gz`) and `*_seg.nii.gz`.

Set `SEARCH_FROM` to anywhere above your data and run the next cell — it walks
down looking for the folder that actually holds the cases and prints the value
to use. The usual gotcha is an extra nesting level: the BraTS2020 archive
unpacks to `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<cases>`."""),
    (CODE, """import glob

SEARCH_FROM = "/content/drive/MyDrive"     # <-- a folder somewhere above the data
WORK        = "/content/drive/MyDrive/tumor_aware_sr"
os.makedirs(WORK, exist_ok=True)


def find_brats_roots(start, max_depth=5):
    \"\"\"Return dirs whose immediate subfolders look like BraTS cases.\"\"\"
    hits = []
    for dirpath, dirnames, filenames in os.walk(start):
        rel = os.path.relpath(dirpath, start)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        n = sum(1 for d in dirnames
                if glob.glob(os.path.join(dirpath, d, "*_seg.nii*")))
        if n:
            hits.append((n, dirpath))
    return sorted(hits, reverse=True)


if not os.path.isdir(SEARCH_FROM):
    raise SystemExit(f"{SEARCH_FROM} does not exist. Is Drive mounted (cell above)?")

found = find_brats_roots(SEARCH_FROM)
if not found:
    print(f"No BraTS case folders anywhere under {SEARCH_FROM}.")
    print("A case folder must contain a file matching *_seg.nii*. Top level holds:")
    for e in sorted(os.listdir(SEARCH_FROM))[:40]:
        print("   ", e + ("/" if os.path.isdir(os.path.join(SEARCH_FROM, e)) else ""))
    print("\\nIf your data is still a .zip, unpack it first:")
    print("   !unzip -q '/content/drive/MyDrive/archive.zip' -d /content/brats")
    raise SystemExit("BraTS data not found - see the listing above.")

for n, p in found:
    print(f"{n:5d} cases   {p}")

BRATS_ROOT = found[0][1]
print(f"\\nUsing BRATS_ROOT = {BRATS_ROOT}")"""),

    (MD, "Confirm the files inside one case, so the modality tag is right."),
    (CODE, """cases = sorted(d for d in glob.glob(os.path.join(BRATS_ROOT, "*")) if os.path.isdir(d))
print(f"{len(cases)} case folders\\nfirst case: {os.path.basename(cases[0])}")
for f in sorted(os.listdir(cases[0])):
    print("   ", f)
print("\\nSet MODALITY below to match: t1c matches *_t1c* or *_t1ce*, "
      "flair matches *_t2f* or *_flair*.")"""),

    (MD, "## 4. Config"),
    (CODE, """import torch

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SIZE     = 128        # slices are center-cropped to SIZE x SIZE
FACTOR   = 4          # k-space truncation (resolution loss)
SIGMA    = 0.03       # Rician noise level
MODALITY = "t1c"

MAX_CASES       = 150       # None = all cases. Start smaller to time one epoch.
SLICES_PER_CASE = 12
SELECT          = "tumor"   # "tumor" keeps the largest-lesion slices; "middle" the central ones
MIN_TUMOR_PIX   = 20
NORMALIZE       = "slice"   # or "volume" to scale each scan by its own statistics

EPOCHS_SEG   = 8
EPOCHS_SR    = 12
BATCH        = 16
BASE         = 32           # U-Net width
TUMOR_WEIGHT = 30.0
TEST_FRAC    = 0.2          # fraction of CASES held out
SEED         = 0

CACHE = f"{WORK}/brats_{MODALITY}_{SIZE}_{SELECT}_{MAX_CASES}.npz"
CKPT  = f"{WORK}/brats_{SIZE}.pt"

torch.manual_seed(SEED)
print("device:", DEVICE)
if DEVICE == "cpu":
    print("WARNING: no GPU. Training will be very slow - switch the runtime.")"""),

    (MD, """## 5. Build the dataset (cached)

First run decodes every volume once; later runs load the `.npz`."""),
    (CODE, """from src.data import make_dataset

full = make_dataset(
    "brats", root=BRATS_ROOT, modality=MODALITY, size=SIZE,
    slices_per_case=SLICES_PER_CASE, min_tumor_pixels=MIN_TUMOR_PIX,
    max_cases=MAX_CASES, select=SELECT, normalize=NORMALIZE, cache_path=CACHE,
)
print(f"{len(full)} slices, tumor pixels/slice (mean): {full.masks.mean() * SIZE * SIZE:.0f}")"""),

    (MD, "Look at the data before training on it."),
    (CODE, """import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for j in range(5):
    i = j * max(1, len(full) // 5)
    axes[0, j].imshow(full.images[i], cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(f"case {full.case_ids[i]}")
    axes[1, j].imshow(full.images[i], cmap="gray", vmin=0, vmax=1)
    axes[1, j].imshow(full.masks[i], cmap="autumn", alpha=0.45 * full.masks[i])
    for ax in (axes[0, j], axes[1, j]):
        ax.axis("off")
axes[0, 0].set_ylabel("image"); axes[1, 0].set_ylabel("+ tumor")
plt.tight_layout(); plt.show()"""),

    (MD, """## 6. Split by case

Adjacent slices of one scan are near-duplicates, so splitting by slice would
leak anatomy across the split and flatter the test numbers. Whole cases are
held out instead."""),
    (CODE, """from torch.utils.data import Subset

train_idx, test_idx = full.split_by_case(val_frac=TEST_FRAC, seed=SEED)
train_ds, test_ds = Subset(full, train_idx), Subset(full, test_idx)

train_cases = {int(full.case_ids[i]) for i in train_idx}
test_cases  = {int(full.case_ids[i]) for i in test_idx}
print(f"train {len(train_ds)} slices / {len(train_cases)} cases")
print(f"test  {len(test_ds)} slices / {len(test_cases)} cases")
assert not train_cases & test_cases"""),

    (MD, "## 7. Segmenter (trained on clean high-res, then frozen)"),
    (CODE, """from src.models import seg_unet, sr_unet, count_params
from src.train import train_segmenter, train_sr

seg = seg_unet(base=BASE)
print("seg params:", f"{count_params(seg):,}")
train_segmenter(seg, train_ds, epochs=EPOCHS_SEG, bs=BATCH, device=DEVICE)
for p in seg.parameters():
    p.requires_grad_(False)"""),

    (MD, """## 8. Two SR models

Same architecture, same data, same schedule — **only the objective differs**.
That is what makes the safety comparison attributable to the loss.

The first epoch prints its wall time; multiply it out before committing to the
full run."""),
    (CODE, """from src.losses import make_sr_loss

sr_d = sr_unet(base=BASE, dropout=0.2)   # distortion-optimal (pixel L1)
sr_t = sr_unet(base=BASE, dropout=0.2)   # tumor-aware (lesion-weighted L1)

train_sr(sr_d, train_ds, make_sr_loss("distortion"),
         factor=FACTOR, sigma=SIGMA, epochs=EPOCHS_SR, bs=BATCH,
         device=DEVICE, tag="sr-distortion", seed=SEED + 1000)
train_sr(sr_t, train_ds, make_sr_loss("tumor_aware", weight=TUMOR_WEIGHT),
         factor=FACTOR, sigma=SIGMA, epochs=EPOCHS_SR, bs=BATCH,
         device=DEVICE, tag="sr-tumor-aware", seed=SEED + 1000)"""),

    (MD, "## 9. Save the checkpoint before evaluating"),
    (CODE, """from src.checkpoint import save_models

save_models(CKPT, seg, sr_d, sr_t, meta={
    "size": SIZE, "factor": FACTOR, "sigma": SIGMA, "base": BASE,
    "weight": TUMOR_WEIGHT, "kind": "brats", "modality": MODALITY,
    "max_cases": MAX_CASES, "select": SELECT,
})
print("saved", CKPT)"""),

    (MD, """## 10. The safety comparison

Read it as: at **comparable PSNR/SSIM**, is the tumor-aware erasure rate lower,
especially on small lesions? If the two models land at very different PSNR the
comparison is confounded — retune `TUMOR_WEIGHT` until they are close."""),
    (CODE, """from src.evaluate import evaluate_pipeline, format_results

results = evaluate_pipeline(
    {"distortion": sr_d, "tumor_aware": sr_t}, seg, test_ds,
    factor=FACTOR, sigma=SIGMA, device=DEVICE, mc_passes=10, size_edges=(50, 200))
print(format_results(results))"""),

    (CODE, """import json, math

def _clean(o):
    \"\"\"NaN is not valid JSON; a rate is None when that size bin had no lesions.\"\"\"
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, float) and math.isnan(o):
        return None
    return o

with open(f"{WORK}/results_{SIZE}.json", "w") as f:
    json.dump(_clean(results), f, indent=2)

def pct(x):
    return "n/a" if x is None else f"{x:.3f}"

d, t = results["distortion"]["safety"], results["tumor_aware"]["safety"]
gap = abs(results["distortion"]["psnr"] - results["tumor_aware"]["psnr"])
print(f"PSNR      distortion {results['distortion']['psnr']:.2f} "
      f"vs tumor-aware {results['tumor_aware']['psnr']:.2f}  (gap {gap:.2f} dB)")
print(f"Erasure   {pct(d['false_negative_erasure_rate'])} -> "
      f"{pct(t['false_negative_erasure_rate'])}")
print(f"  small   {pct(d['erasure_rate_by_size']['small'])} -> "
      f"{pct(t['erasure_rate_by_size']['small'])}")
print(f"lesions by size: {d['lesions_by_size']}")
if gap > 1.0:
    print(f"\\nNOTE: a {gap:.1f} dB PSNR gap confounds the safety comparison. "
          "Retune TUMOR_WEIGHT until the two models land within ~1 dB.")"""),

    (MD, "## 11. Figures"),
    (CODE, """from src.degrade import degrade_torch
from src.figures import comparison_figure, uncertainty_figure
from src.uncertainty import mc_predict

sample = test_ds[0]
hr   = sample["hr"][None].to(DEVICE)
mask = sample["mask"][None].to(DEVICE)
lr   = degrade_torch(hr, factor=FACTOR, sigma=SIGMA)

sr_d.eval(); sr_t.eval()
with torch.no_grad():
    outs = {"distortion": sr_d(lr), "tumor_aware": sr_t(lr)}

comparison_figure(hr, mask, lr, outs, seg, save_path=f"{WORK}/comparison.png")
mean, unc = mc_predict(sr_t, lr, passes=15)
uncertainty_figure(lr, mean, unc, hr, save_path=f"{WORK}/uncertainty.png")"""),

    (MD, "## 12. CPU deployment benchmark"),
    (CODE, """from src.uncertainty import cpu_benchmark

b = cpu_benchmark(sr_d, size=SIZE, reps=20)
print(f"CPU inference: {b['latency_ms']:.1f} ms/slice | "
      f"params {b['params']:,} | {b['param_memory_mb']:.2f} MB")"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "colab_brats.ipynb")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
