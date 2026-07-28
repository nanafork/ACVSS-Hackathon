# Experiment log

Every training run, what it was trained on, and what it actually showed. Append
a row when you train; never edit a past row. The point of this file is that a
number on a slide can always be traced back to a checkpoint and a command.

**The rule this file exists to enforce:** state the data source next to every
number. A result on the synthetic phantom and a result on BraTS are not
comparable, and conflating them is the single easiest way to mislead a judge.

---

## Status at a glance

| | |
|---|---|
| Checkpoint the local demo uses | `checkpoints/demo.pt` (run 2, **synthetic**) |
| Best real-data checkpoint | `checkpoints/demo_brats.pt` (run 4) — downloaded, md5 verified, loads and runs locally |
| Does the headline claim hold on real data? | **Not demonstrated.** See run 4. |
| Numbers currently in `paper/` | From an unreproducible older run. **Unverified.** |
| Numbers currently in the deck | Run 2 (synthetic). Labelled as synthetic. |

---

## Runs

### Run 1 — original quick checkpoint (superseded)
| | |
|---|---|
| Data | Synthetic phantom, size 80 |
| Command | unknown (predates this log) |
| Checkpoint | old `checkpoints/demo.pt`, backed up then replaced |
| Outcome | **Undertrained.** Segmenter fired only on clean HR, so the distortion model read 0.00 cm³ and every 2D panel showed an empty outline. This is the origin of the retired "erased entirely" claim. |

### Run 2 — CPU retrain, synthetic
| | |
|---|---|
| Data | Synthetic, size 96, n=240 |
| Command | `python scripts/train_demo.py --out checkpoints/demo_trained.pt` |
| Checkpoint | promoted to `checkpoints/demo.pt` (committed) |
| Held-out | n=64 synthetic slices, seed 999 |

| metric | distortion | tumor-aware |
|---|---|---|
| PSNR | 23.50 | 23.03 |
| SSIM | 0.771 | 0.733 |
| Dice | 0.473 | 0.646 |
| FNER | 0.253 | 0.177 |
| FPDR | 0.565 | 0.564 |
| AUROC | 0.939 | 0.934 |

Reads well, but see run 3 for why the substrate makes it meaningless.

### Run 3 — GPU, synthetic at scale
| | |
|---|---|
| Data | Synthetic, size 128, n=1200 |
| Command | `train_demo.py --size 128 --n 1200 --seg-epochs 40 --sr-epochs 60 --weight 40` |
| Checkpoint | `demo_gpu.pt` (sandbox only, not transferred) |
| Held-out | n=128 synthetic, 160 lesions |
| Wall clock | ~6 min (1.8 min GPU at 4.4 ms/step + 1.7 min CPU degradation at 0.71 ms/image) |

| metric | distortion | tumor-aware |
|---|---|---|
| PSNR | 32.60 | 31.99 |
| Dice | 0.657 | 0.753 |
| FNER | 0.163 | **0.050** |
| small-lesion erasure | 0.522 | 0.261 |
| FPDR | 0.330 | 0.481 |

**Do not quote these.** Probing the generator showed the task is close to
degenerate: a fixed threshold `img > 0.9` scores **Dice 0.878** on clean HR,
better than the trained U-Net manages on reconstructions. Tumor pixels are
hardcoded to exactly 0.95, and 55% of pixels are identical across every sample
because the brain ellipse never moves. "Erasure" here only measures whether SR
preserved a constant peak intensity.

Note also that FPDR flipped between runs 2 and 3 (flat, then 0.33 → 0.48). That
instability is itself evidence the synthetic numbers are not load-bearing.

### Run 4 — GPU, real BraTS (whole tumor)
| | |
|---|---|
| Data | MSD Task01_BrainTumour = BraTS, 484 cases, public, no registration |
| Prep | `prepare_msd.py --region wt --size 128` → 5,259 slices from 460 cases |
| Split | **By patient**: 368 train / 92 test, zero case overlap (asserted) |
| Command | `train_demo.py --cached slices_128.npz --size 128 --seg-epochs 40 --sr-epochs 60 --weight 40` |
| Checkpoint | `checkpoints/demo_brats.pt` (committed; md5 `d765a893fcc2d6d2b2891dd53ae90c73`) |
| Held-out | 1,074 slices, 92 unseen patients, 5,128 lesions |
| Raw results | `results/brats_wt_heldout.json` |

| metric | low-res | distortion | tumor-aware |
|---|---|---|---|
| PSNR | 21.58 | 25.10 | 24.32 |
| SSIM | 0.549 | 0.768 | 0.738 |
| Dice | 0.565 | **0.637** | 0.622 |
| FNER | 0.615 | 0.646 | 0.637 |
| FPDR | 0.711 | **0.482** | 0.596 |
| AUROC | — | 0.846 | 0.819 |

Re-run independently after downloading the checkpoint, to check the result was
not an artifact of MC-dropout randomness. It reproduces: FNER 0.645 / 0.637,
FPDR 0.478 / 0.595, Dice 0.633 / 0.624. The finding is stable.

**The central claim does not replicate.** Erasure 0.646 vs 0.637 is under one
percentage point over 5,128 lesions. It goes the wrong way on medium lesions
(0.505 → 0.546) and on Dice, and tumor-aware is clearly worse on hallucination.

Second finding, equally important: **both SR models erase more than the raw
low-res input** (0.646 and 0.637 vs 0.615), while improving Dice and PSNR. That
is the paper's own thesis landing harder than intended, and it lands against
our proposed fix as well as against the baseline.

Caveats that make this a weaker test than the claim deserves:
- The mask is whole tumor, which includes edema and covers ~10.9% of the image.
  The argument is about the *small enhancing* lesion being negligible in PSNR.
  Lesion-weighting a region that large is not the small-object regime.
- The segmenter is the measuring instrument and only reaches Dice 0.62–0.64 on
  real data. A weak instrument compresses the gap between the two objectives.
- 3,732 of 5,128 lesions are "small" and ~80% are erased by everything,
  including the low-res input.

### Run 5 — GPU, real BraTS (enhancing tumor only) — IN PROGRESS
| | |
|---|---|
| Prep | `prepare_msd.py --region et --min-tumor-pixels 10` |
| Why | Enhancing tumor (label 3) is the small, bright, easily-erased structure the proposal is actually about. This is the honest test of the hypothesis. |
| Status | Running on the sandbox. Results to be appended here, whichever way they come out. |

---

## Bugs found while doing the above

Both silently corrupted results rather than crashing, which is why they survived
until someone checked the numbers against the pictures.

1. **Dropout left on between slices** (`demo.py` `_infer`). `mc_predict` enables
   dropout for its Monte Carlo passes and never restores eval mode, so every
   slice after the first was super-resolved stochastically. The tumor-aware
   model was being judged on noise. After the fix, 8 of 64 held-out slices
   favour tumor-aware and none favour distortion; before it, one slice
   spuriously reversed. Same bug fixed in `viz_bridge.run_pipeline_3d`.
2. **Uncertainty volume transposed against the brain** (`render_3d.py`). The
   vendored analyzer maps array axis 0 to VTK x, so the grid needs the array's
   own axis order with Fortran-ordered cell data. The field was rendering
   rotated, outside the brain it was meant to sit inside. Also now normalized by
   a percentile (a few voxels sit ~60× above p99) and masked to tissue (60% of
   the raw uncertainty mass was in empty background).

## Reproducing a real-data run

```bash
curl -L -o Task01.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar xf Task01.tar
python scripts/prepare_msd.py --root Task01_BrainTumour --out slices_128.npz --size 128 --region et
python scripts/train_demo.py --cached slices_128.npz --size 128 --seg-epochs 40 --sr-epochs 60
```
