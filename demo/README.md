# Safety demo — "image quality is not a safety metric"

Self-contained page. Drag the slider to degrade the acquisition; both models
super-resolve it; watch the image-quality numbers stay flat while the lesions
disappear.

The moment to point at is **×3**:

| | PSNR | lesions found | model confidence |
|---|---|---|---|
| distortion-optimal | **26.83 dB** | **0 of 2** | **99.6 %** |
| tumor-aware | 25.69 dB | **2 of 2** | 0 % |

The distortion model has *better* image quality, lost *both* tumors, and is
maximally confident about it.

## Regenerating

Needs a trained checkpoint at `checkpoints/demo.pt` (see `scripts/train_demo.py`),
plus `torch`, `numpy`, `imageio`. GPU optional.

```bash
export TRUSTMRI_ROOT=$(pwd)          # repo root; defaults to two levels up
python demo/tools/build_demo_payload.py   # -> demo_payload.json (frames + metrics)
python demo/tools/make_demo_html.py       # -> demo_safety.html
```

`build_demo_payload.py` picks the display slice automatically: it scans the
held-out set for a slice where the distortion model loses a lesion the
tumor-aware model keeps. **That slice is chosen because it shows the effect
clearly** — the average effect is much more modest (see below), and the page
says so.

## The flag, and why its sign is backwards

The verdict uses **no ground truth**. It runs the SR network several times with
dropout active, segments every output, and measures how much the segmenter
disagrees with itself across passes.

The counter-intuitive part: **when a lesion is erased the segmenter becomes more
confident, not less.** It is confidently wrong, so *unusually low* disagreement
is the danger signal. This matches AGENT_GUIDE §3.2's prediction.

Measured on held-out synthetic data, single checkpoint (seed 0), 59 slices of
which 26 had an erased lesion:

| | slice-level AUROC |
|---|---|
| overall | **0.782** |
| medium lesions | 0.853 |
| large lesions | 0.821 |
| **small lesions** | **0.571 — chance** |

Uncertainty correlates +0.60 with lesion area, so the size-stratified numbers
are the honest ones: the signal survives within the medium and large bands, so
it is not merely measuring lesion size. It does **not** survive on the smallest
lesions, which is exactly where erasure matters most clinically. State that
gap rather than hiding it — it is the open problem, not a solved one.

## What was tried first and did not work

`src/consistency.py` detects erasure by re-applying the forward model to the SR
output and comparing inside the acquired k-space band. It reaches AUROC 0.948 on
the hand-erased phantom, but on this pipeline's own output it is **at chance at
the slice level** — sensitivity tracks the false-alarm rate at every threshold
tested (`eval_kspace_detector_sweep.py`, z_thresh 1.5–4.0 × min_area 4–32; best
point 65 % sensitivity at 45 % false alarm).

The reason is informative: that detector looks for *signal removal*. On the
phantom, lesions were cut out, so signal really was gone. These SR models are
trained with L1 and reproduce the measured band faithfully — the lesion is still
in the image, it just becomes unreadable to the **segmenter**. There is no
measurement inconsistency to find, so the detector is correctly silent.

## Honesty notes (keep these on the slide)

- **Synthetic phantom slices.** Not BraTS, not clinical data.
- Single checkpoint, one hand-picked display slice.
- Average effect across 3 seeds: small-lesion erasure −0.064 at equal PSNR
  (λ=5 vs λ=0), and roughly **doubled false positives** (0.305 → 0.393).
- Confidence on the page is the MC-dropout segmentation spread rescaled to
  0–100 across the frames shown; it is not a calibrated probability.

## Files

```
demo/tools/build_demo_payload.py        renders frames + metrics -> demo_payload.json
demo/tools/make_demo_html.py            payload -> demo_safety.html (one file, no assets)
demo/tools/eval_detector_candidates.py  slice-level AUROC for 5 reference-free scores
demo/tools/eval_detector_confound.py    size-confound check + stratified AUROC
demo/tools/eval_kspace_detector_sweep.py  threshold sweep that retired the k-space detector
```
