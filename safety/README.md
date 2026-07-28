# The safety layer: catching tumors that vanish during enhancement

**Branch owner:** Hassan · **Base:** `neuro-voxel-3d-viz` · **Data:** real BraTS (MSD Task01), enhancing tumor, patient-level splits

This directory adds a deployment-side safety layer to the existing pipeline, plus
the experiments that tell you what it can and cannot do. Everything here was
measured on the **val** split; **test was left sealed**.

---

## ▶ Run the demo

`safety/demo_safety.html` is the one to present: a single self-contained file,
no server, no GPU, no network. Open it in any browser.

Two cases, a scan-quality slider (×2 → ×8) and a SAFETY LAYER toggle.

**Case A (opens by default) — the run to present**

| slider | verdict |
|---|---|
| ×2, ×3 | clear |
| **×4** | **flag on the standard model, none on ours** |
| ×5, ×6, ×8 | cannot verify |

**Case B** — the standard model loses a lesion ours keeps (flags at ×2, ×3).

The verdict has **three** states, in precedence order, all computed from the
slider and the image rather than keyed to a slice:

1. no usable reference in the acquired scan → **cannot verify**
2. a region present before enhancement is gone after → **flag**
3. most lesions below the ~N px the acquisition actually measured → **cannot verify**
4. otherwise → **clear**

The third state is the honest part: at ×N the acquisition never sampled
structures under about N px, so *no* image-space check can confirm them. Silence
there means unmeasured, not safe — and saying so out loud is stronger than
letting a green tick imply a guarantee.

To regenerate it, or to point it at different slices:

```bash
pip install torch numpy imageio marimo huggingface_hub
python safety/scripts/fetch_assets.py        # ~290 MB from HF, one time
python safety/scripts/build_demo_html.py     # -> safety/demo_safety.html
```

There is also a live GPU version, which recomputes on any of the 2,650 held-out
slices — useful for showing the rule is a rule and not a lookup table:

```bash
marimo run safety/app.py                  # booth mode, code hidden
```

Opens on `http://localhost:2718`. Four live panels — **acquired → standard model
→ ours → ground truth** — with a slice slider, a degradation slider (×2–×8), and
a **SAFETY LAYER** toggle. Every frame is real inference; nothing is precomputed.
Runs on CPU if there is no GPU, just slower.

**The beat to present, on the default slice (843):** the standard model loses a
small lesion and the layer fires red; the tumor-aware model keeps it and stays
green. Then drag the degradation slider to show it appear and disappear.

Optional but worth it: `python safety/scripts/train_robust_segmenter.py`
(~25 min on an RTX 6000) writes `safety/robust_seg.pt`, and the app picks it up
automatically — it roughly halves the false alarms.

Verified end to end with `marimo export html safety/app.py` on a fresh clone:
all cells execute, both verdict states render.

---

## 1. The framing

### The one-sentence pitch

> Super-resolution sometimes makes a tumor disappear. Image-quality metrics
> cannot see it happen. We add a layer that flags it at deployment, using no
> ground truth — and we train a model that does it less often.

### The three facts the talk rests on

**Fact 1 — enhancement helps on net.** Against the scan the machine actually
produced, super-resolution *recovers* far more lesions than it destroys:

| | lesions |
|---|---|
| recovered by enhancement (invisible before, found after) | **774** |
| destroyed by enhancement (found before, invisible after) | **339** |
| net | **+435 (+4.6 points)** |

Do not pitch this as "enhancement is dangerous." It isn't. The pitch is
*"enhancement is useful, and it has a silent failure mode."*

**Fact 2 — the failures are invisible to the metrics the field uses.** A lesion
is a tiny fraction of the pixels, so PSNR and SSIM barely move when one is lost.
That is the entire problem, and it is why a safety layer has to exist at all.

**Fact 3 — the damage is real but rare.** 339 destroyed lesions is **3.6%** of
all lesions, and **314 of them are small**. Small lesions are where erasure
bites; large ones are essentially never lost (1 of 1,718).

### What the safety layer actually is

```
segment the ACQUIRED scan   ->  lesion present
segment the ENHANCED image  ->  lesion gone
                                ─────────────
                                flag: "possible erasure — review the original"
```

That is the whole method. No training, no ground truth, no access to the
enhancement model's weights. It wraps any black-box enhancement tool.

**Say the simplicity out loud — it is the selling point.** A radiology
department can switch this on over software they already run.

### Why this simple thing, and not something cleverer

Four more sophisticated detectors were built and measured first. All failed:

| detector | result on real BraTS |
|---|---|
| k-space consistency (re-apply the forward model, compare in-band) | **at chance** at slice level |
| MC-dropout uncertainty | **18%** recall on small lesions; and the sign is *inverted* |
| log-odds evidence drop + conformal calibration | **AUROC 0.478** |
| task comparison (segment both sides) | **the one that works** |

The failures explain the winner. The k-space check fails because the model does
**not** remove signal from the measured band — the reconstruction is faithful.
MC-dropout fails because an erasing model is **confidently** wrong: uncertainty
*drops* when a lesion is lost (oracle lesion-level AUROC 0.287, i.e. 0.713
inverted). The damage is not in the pixels and not in the model's confidence. It
is in the downstream reading — so you have to look at the task, not the image.

> **Task-level consistency, not pixel-level consistency.** That is the
> one-line contribution.

### Be honest about the limits — they are on the slide, not hidden

- **Precision is 20%.** About 1 in 5 flags marks a real vanished lesion. Present
  the layer as a **review prompt**, never as a diagnosis.
- **Coverage is partial.** Of lesions absent after enhancement, only those the
  segmenter could see in the acquired scan are flaggable. The rest were
  invisible in both images — that is the segmenter's own floor, not damage done
  by super-resolution.
- **The demo case is hand-picked.** Slice 843 was chosen from ~300 scanned
  because it shows the effect cleanly. Roughly 2 in 300 slices do. State the
  aggregate (3.6% of lesions) alongside it.

---

## 2. A finding that affects the main results

**A large part of the measured "erasure" is the segmenter's domain gap, not
damage from super-resolution.**

`train_segmenter` trains on **clean high-resolution images only** — it never
sees a degraded or super-resolved one. It is then used to score both. Measured
on val, same weights:

| input | Dice |
|---|---|
| clean HR (its training domain) | 0.842 |
| the acquired low-field scan | **0.680** |
| the super-resolved image | 0.779 |

Retraining one segmenter on all three domains (`scripts/train_robust_segmenter.py`)
closes the gap and changes the numbers:

| | HR-only segmenter | domain-robust |
|---|---|---|
| Dice on LR | 0.680 | **0.801** |
| Dice on SR | 0.779 | **0.810** |
| Dice on HR | 0.842 | 0.825 |
| **measured "real vanishings"** | **340** | **266 (−22%)** |
| flags raised | 1493 | 403 |
| flag precision | 11.3% | **20.3%** |
| flag recall | 49.4% | 30.8% |

**Fixing the reader removed 22% of the measured erasure without touching the
super-resolution model at all.** Every safety rate in the paper — the erasure
rates, the 65% small-lesion floor, the +3.2-point headline — is measured through
the HR-only segmenter. A domain-robust reader is a fairer instrument and would
likely move those numbers. This is the same class of issue as the dropout leak
found in the 2026-07-28 audit, and it is better found by us than by a judge.

---

## 3. The demo

`app.py` is a marimo app. It runs real inference on the GPU — nothing is
precomputed.

```bash
marimo run safety/app.py          # booth mode, no code visible
marimo edit safety/app.py         # to tweak
```

Four live panels: **acquired → standard model → ours → ground truth**, with the
vanished region outlined in red and a verdict banner under each model. Controls:
slice, degradation factor (×2–×8), and a SAFETY LAYER switch.

**The demo beat:** on the default slice the standard model loses a small lesion
and the layer fires; the tumor-aware model keeps it. Both of the framings above
in one screen.

**Answer ready for the obvious question.** "How often is it right?" → 20%
precision, ~0.15 flags per slice, 3.6% of lesions damaged. Have that on the slide
straight after the demo, not in an appendix.

---

## 4. Reproducing

Everything needs the slice cache and a checkpoint from
`douyeszn/tumor-aware-sr` on Hugging Face (now public).

```bash
pip install torch numpy imageio marimo huggingface_hub
python safety/scripts/fetch_assets.py             # cache + shared-segmenter checkpoint
python safety/scripts/train_robust_segmenter.py   # ~25 min on an RTX 6000
python safety/scripts/eval_trigger.py             # precision/recall table
marimo run safety/app.py
```

`fetch_assets.py` pulls `data/et_full.npz` (288 MB) and `shared/sh_w40_sl0.0.pt`.
The robust segmenter writes to `safety/robust_seg.pt`.

---

## 5. Numbers, with provenance

All on **val** (70 cases, 2,650 slices, 9,490 lesions), enhancing tumor, shared
frozen segmenter `sh_w40_sl0.0.pt`, degradation ×4 σ=0.03. Test untouched.

| quantity | value |
|---|---|
| lesions destroyed by enhancement | 339 (3.6%) |
| lesions recovered by enhancement | 774 (8.2%) |
| invisible in both images (segmenter floor) | 5,116 (53.9%) |
| flag precision, HR-only reader | 11.3% |
| flag precision, domain-robust reader | 20.3% |
| flag recall, domain-robust reader | 30.8% |
| MC-dropout recall on small lesions | 18% |
| MC-dropout oracle AUROC (inverted) | 0.287 |
| k-space consistency, slice level | at chance |
| conformal log-odds monitor AUROC | 0.478 |

`results/summary.json` carries these machine-readable. Raw per-experiment JSON
lives in the sandbox that produced it and is regenerable from the scripts.

---

## 6. What to do next

1. **Re-run the headline numbers with the domain-robust segmenter.** The current
   instrument is biased against the acquired scan, and the erasure rates inherit
   that bias.
2. **Push flag precision past 20%.** The blocker is the reader's false positives
   on the acquired scan, not the comparison rule. A better reader is the lever.
3. **Leave test sealed** until one final evaluation.
