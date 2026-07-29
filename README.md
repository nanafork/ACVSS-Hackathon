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
  degrade.py      physics-informed forward model (k-space truncation + Rician noise)
  data.py         BraTS (.nii.gz) and synthetic 2D-slice datasets
  models.py       SR U-Net (with dropout) and segmentation U-Net
  losses.py       distortion-optimal vs tumor-aware (lesion-weighted) SR losses
  metrics.py      PSNR, SSIM, Dice, erasure + hallucination rates by lesion size
  uncertainty.py  MC-dropout uncertainty; uncertainty-vs-error AUROC; CPU benchmark
  evaluate.py     end-to-end comparison + summary table
  train.py        training loops
  figures.py      comparison panel and uncertainty figure
scripts/metric_blindness.py  what deleting the whole tumor costs in PSNR/SSIM
smoke_test.py     end-to-end run on synthetic data (CPU, no download)
notebooks/tumor_aware_sr.ipynb   Kaggle orchestration (real BraTS or synthetic)
scripts/make_notebook.py         regenerates the notebook
```

## Quick start (local, no download)

```bash
pip install "numpy<2" torch scikit-image nibabel matplotlib
python smoke_test.py
```

This trains everything at tiny scale on synthetic data and prints the metric
table. It is a **correctness check**, not a scientific result.

## Demo

```bash
python scripts/train_demo.py        # trains + saves checkpoints/demo.pt (synthetic; add --brats --root for real)
python main_demo.py                 # writes main_demo.html (self-contained, open in any browser / share)
python demo.py --gradio             # interactive app (needs: pip install gradio)
```

`main_demo.html` is the page we present, and [DECK.md](DECK.md) is how to
present it: the 7 minute running order with timings, what to say on the slides
that are hard to improvise, the speaker split, and the questions to have answers
ready for. Slide headlines are assertions and the figure under each one is its
evidence; keep it that way when editing `main_demo.py`.

The page carries the measured safety headline, four 3D viewports (ground truth, tumor-aware, distortion-optimal, and
the MC dropout uncertainty field), a rotating overlay, and the 2D evidence
underneath it: per-slice low-res / distortion / tumor-aware / true panels, the
predicted tumor masks (blue outline = true tumor), the uncertainty and error
maps, and a table with PSNR/SSIM/Dice and **lesions erased** / **fabricated**
counts. Colors come from `src/palette.py` so the 2D figures and the 3D renders
label the same model with the same hue. If no checkpoint exists, the demo
quick-trains one on synthetic data so it always runs.

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
