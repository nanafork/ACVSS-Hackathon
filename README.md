# Tumor-Aware MRI Super-Resolution: a Safety Study

Deep-learning super-resolution can sharpen cheap low-field brain MRI, but it is
optimized for image-quality metrics (PSNR/SSIM) that barely penalize erasing a
small tumor. This project tests that misalignment and offers a fix.

**Pipeline:** degrade a high-resolution scan (k-space truncation + Rician noise)
→ super-resolve with two objectives (distortion-optimal vs tumor-aware) →
segment the tumor → measure image quality, segmentation Dice, and **safety
rates** (lesion erasure, hallucination), plus **MC-dropout uncertainty** and a
**CPU benchmark**.

See `proposal.pdf` for the full write-up.

## Layout

```
src/
  degrade.py      physics-informed forward model (k-space truncation + Rician noise);
                  numpy reference + batched torch port for on-device training
  data.py         BraTS (.nii.gz) and synthetic 2D-slice datasets
  models.py       SR U-Net (with dropout) and segmentation U-Net
  losses.py       distortion-optimal vs tumor-aware (lesion-weighted) SR losses
  metrics.py      PSNR, SSIM, Dice, erasure + hallucination rates by lesion size
  uncertainty.py  MC-dropout uncertainty; uncertainty-vs-error AUROC; CPU benchmark
  evaluate.py     end-to-end comparison + summary table
  train.py        training loops
  checkpoint.py   save/load the three trained models
  figures.py      comparison panel and uncertainty figure
smoke_test.py     end-to-end run on synthetic data (CPU, no download)
tests/            unit tests; the BraTS ones build a fake case tree, no download
notebooks/colab_brats.ipynb      Colab orchestration (real BraTS, GPU)
notebooks/tumor_aware_sr.ipynb   Kaggle orchestration (real BraTS or synthetic)
scripts/train_demo.py            CLI trainer (synthetic or BraTS)
scripts/make_colab_notebook.py   regenerates the Colab notebook
scripts/make_notebook.py         regenerates the Kaggle notebook
```

## Quick start (local, no download)

```bash
pip install "numpy<2" torch scikit-image nibabel matplotlib
python smoke_test.py
```

This trains everything at tiny scale on synthetic data and prints the metric
table. It is a **correctness check**, not a scientific result.

```bash
python tests/test_pipeline.py    # or: python -m pytest tests/ -q
```

Unit tests for the degradation model, the BraTS loader, and the MC-dropout
path. The BraTS tests synthesise a fake case tree, so they need no download.

## Demo

```bash
python scripts/train_demo.py        # trains + saves checkpoints/demo.pt (synthetic; add --brats --root for real)
python demo.py                      # writes demo.html (self-contained, open in any browser / share)
python demo.py --gradio             # interactive app (needs: pip install gradio)
```

`demo.html` shows, per slice: the low-res / distortion / tumor-aware / true
panel, the predicted tumor masks (cyan = true outline), the uncertainty map, and
a table with PSNR/SSIM/Dice and **lesions erased** / **fabricated** counts. If no
checkpoint exists, the demo quick-trains one on synthetic data so it always runs.

## Real run (Colab, BraTS)

Open `notebooks/colab_brats.ipynb` in Colab, set the runtime to a GPU, and edit
`BRATS_ROOT` to the folder holding one subfolder per case. The notebook checks
the layout before training so a wrong modality tag fails in seconds, not after
an epoch.

Same thing from the command line:

```bash
python scripts/train_demo.py --brats --root /path/to/BraTS \
    --size 128 --max-cases 150 --select tumor \
    --cache cache/brats128.npz --evaluate
```

Useful flags: `--max-cases` (start small and time one epoch before committing),
`--select tumor` (keep the largest-lesion slices rather than the central ones),
`--cache` (see below), `--normalize volume`, `--test-frac`.

**Slice caching.** A BraTS volume decompresses to ~70 MB and `.nii.gz` is gzip,
so pulling one 2D slice costs a full decode. The loader therefore decodes each
volume exactly once, extracts the wanted slices, and — with `--cache` /
`cache_path` — writes them to an `.npz`. Later runs load that in seconds, which
matters when a Colab session disconnects mid-experiment.

**Splitting.** `split_by_case()` holds out whole cases. Adjacent slices of one
scan are near-duplicates, so a slice-level split leaks anatomy across the
boundary and inflates the test numbers.

## Real run (Kaggle, BraTS)

1. Add this repo to the notebook so `src/` is importable (upload as a Dataset or
   `git clone` into `/kaggle/working`).
2. Attach a BraTS dataset (e.g. the BraTS2020 subset on Kaggle, no registration
   required).
3. In `notebooks/tumor_aware_sr.ipynb`, set `DATA_KIND = "brats"` and point
   `DATA_ROOT` at the BraTS training folder.
4. Enable the GPU accelerator and Run All.

## The headline result

At **matched PSNR/SSIM**, does the tumor-aware model have a lower
**False Negative Erasure Rate** than the distortion-optimal model, especially on
small lesions? If yes, image quality is not a safety metric, and a tumor-aware
objective is the fix. The uncertainty map is checked as a complementary safety
signal (does high uncertainty coincide with error?).

## Scope

The low-field acquisition is **simulated** (resolution loss + Rician noise); it
does not reproduce field-strength contrast changes. This is a proof of concept.
Validation on real low-field or BraTS-Africa data is the next step.
