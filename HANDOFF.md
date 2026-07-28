# Branch `hassan/safety-layer` — read this first

Branches off `neuro-voxel-3d-viz`. Adds a **deployment-side safety layer**, the
experiments that say what it can and cannot do, and a demo you can open in a
browser with no setup.

Everything below was measured on **real BraTS (MSD Task01), enhancing tumor,
patient-level splits, VAL only**. The test split was left sealed.

---

## 1. Open the demo (30 seconds, no GPU, no downloads)

```bash
open safety/demo_safety.html          # or just double-click it
```

Two cases, a scan-quality slider (×2 → ×8), a SAFETY LAYER toggle.

**Case A is the run to present:**

| slider | what you see |
|---|---|
| ×2, ×3 | clear — nothing lost |
| **×4** | **the standard model's flag fires; ours stays clean** |
| ×5, ×6, ×8 | ⚠ cannot verify |

Toggle the safety layer **off** at ×4: the red outline disappears and the image
looks perfectly fine. That is what a radiologist sees today.

**Case B** (second button) shows the model difference in lesion counts: the
standard model finds 1 of 3 lesions where ours finds 2, at ×2/×3/×4.

---

## 2. The framing

> Enhancing resolution can make a tumor disappear. Super-resolution is judged by
> PSNR, and a small lesion is a fraction of a percent of the pixels, so the
> metric barely moves whether the tumor survives or not. On held-out patients,
> **339 lesions visible before enhancement could not be found after it** — 314 of
> them small. Our safety layer flags it at deployment, with no ground truth. Our
> tumor-aware training loses fewer in the first place.

**Say this too, before anyone asks:** enhancement also **recovers 774** lesions
the raw scan had lost. Net **+435**. The pitch is *"it helps, and it has a silent
failure mode"* — not *"it is dangerous"*. That framing is stronger and it is what
the data supports.

**Do not say** the model "deletes the signal". We tested that specifically and it
is false: the enhanced image still contains the lesion, it stops being
*detectable*. "Disappears", "is lost", "can no longer be found" are accurate.

---

## 3. What the safety layer is

```
read the ACQUIRED scan   -> p_before
read the ENHANCED image  -> p_after      (same segmenter, which it does not own)
flag any region present in p_before and gone from p_after
```

No ground truth, no retraining, no access to the enhancement model. It wraps any
black-box tool. The verdict has **three** states, in precedence order, all
computed from the acquisition factor and the image:

1. no usable reference in the acquired read → **cannot verify**
2. a region vanished → **flag**
3. strict majority of lesions below the ~N px the acquisition measured → **cannot verify**
4. otherwise → **clear**

State 3 is the physics: at truncation ×N the scan never sampled structures under
about N px, so *no* image-space check can confirm them. Silence there means
unmeasured, not safe. Saying so is stronger than letting a green tick imply a
guarantee — and it removed a false positive we would otherwise have presented
(slice 1960 at ×8 flagged a spurious region; the rule now suppresses it).

### Why something this simple

Four more sophisticated detectors were built and measured first. All failed:

| detector | result on real data |
|---|---|
| k-space consistency (physics) | **at chance** at slice level (0.948 on a phantom) |
| MC-dropout uncertainty | 18% recall on small lesions, and the sign is **inverted** — an erasing model is *more* confident |
| log-odds evidence drop + conformal calibration | **AUROC 0.478** |
| requiring flags to persist across slices | precision got *worse* (11.3% → 1.6%) |

They fail for one reason, and it is the finding: **the enhancement does not
delete pixels.** It reproduces the measured k-space faithfully; the lesion is
still in the image and stops being readable. Nothing that inspects pixels,
physics, or model confidence can see that. Only the downstream task can.

---

## 4. Numbers (val, 9,490 lesions, 70 cases)

| | |
|---|---|
| lesions destroyed by enhancement | **339 (3.6%)**, 314 of them small |
| lesions recovered by enhancement | **774 (8.2%)** |
| invisible in both images (segmenter floor) | 5,116 (53.9%) |
| flag precision, HR-only reader | 11.3% |
| flag precision, domain-robust reader | **20.3%** |
| 8-way test-time augmentation | precision ×2 **and** recall 63% → 78% |

`safety/results/summary.json` has all of it machine-readable with provenance.

**Limits to state on the slide, not hide:** precision ~20%, so it is a **review
prompt, not a diagnosis**; only lesions the reader saw before enhancement are
flaggable (6.2% of those missing afterwards); demo cases are chosen for clarity
from an exhaustive scan (~880 slices; 4 produce a clean flag frame).

---

## 5. A finding that affects the paper's numbers

`train_segmenter` trains on **clean HR only**, then scores LR and SR. Measured
on val, same weights: Dice **0.842** on HR, **0.680** on the acquired scan,
0.779 on SR.

Training one segmenter across all three domains
(`safety/scripts/train_robust_segmenter.py`) gives 0.825 / **0.801** / 0.810 —
and **measured "real vanishings" drop from 340 to 266, −22%**, with the
super-resolution untouched.

So a fifth of the erasure everyone is measuring is the reader being out of its
training domain, not damage done by enhancement. Every safety rate in the paper
runs through the HR-only segmenter and inherits that bias. Same class of issue as
the dropout leak in the 2026-07-28 audit. **Worth re-running the headline numbers
with the robust reader.**

---

## 6. What is left

1. **Re-run the headline numbers with the domain-robust segmenter** (§5). The
   current instrument is biased against the acquired scan.
2. **Combine TTA + robust reader.** Each roughly doubles flag precision on its
   own; they are independent and have never been measured together.
3. **Update the paper's abstract, conclusion and limitations** — they still
   describe a synthetic study and quote retired numbers (27.8% → 15.2%), while
   the results section is real BraTS with p=0.007. These are the sections judges
   read.
4. **Test stays sealed** until one final evaluation.

---

## 7. Layout

```
safety/demo_safety.html      the demo — open it, no setup
safety/app.py                live GPU version: marimo run safety/app.py
safety/README.md             fuller version of this document
safety/results/summary.json  every number with provenance
safety/scripts/
  fetch_assets.py            pulls the slice cache + checkpoints from HF
  build_demo_html.py         regenerates demo_safety.html
  train_robust_segmenter.py  the domain-robust reader (~25 min on a GPU)
  eval_trigger.py            flag precision / recall
  eval_sr_damage.py          destroyed vs recovered lesions
  eval_uncertainty_detector.py   MC-dropout, three granularities
  eval_conformal_monitor.py  the log-odds monitor that failed
```

Assets are gitignored; `python safety/scripts/fetch_assets.py` fetches them
(~290 MB) from `douyeszn/tumor-aware-sr` on Hugging Face. The demo page needs
none of it.
