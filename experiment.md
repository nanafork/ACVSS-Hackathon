# Experiment log — TrustMRI / ACVSS Hackathon

Running record of what was actually run, what it produced, and what is still
open. Append new entries at the bottom of §6; keep the summary tables current.

**Rule for this file (AGENT_GUIDE R1):** every number here carries its data
source and the run that produced it. Nothing in this file is a validated
clinical result. All runs so far are **synthetic data only** — no BraTS has been
touched yet.

---

## 1. Setup

| | |
|---|---|
| Repo | `github.com/nanafork/ACVSS-Hackathon` (push access) |
| Local clone | `~/Desktop/ACVSS_HACKATHON/repo` |
| My branch | `hassan/dev`, branched from `origin/main` (81738a8), pushed and tracking |
| Friend's branch | `neuro-voxel-3d-viz` (author: douyeszn) |
| Friend worktree | `~/Desktop/ACVSS_HACKATHON/friend-viz` (detached, read-only use) |

`main` is a single commit (81738a8, "Initial commit").

### Untracked local work (NOT in git, NOT backed up)

Six files exist in the local clone but are untracked. They are the second layer
of the project and are documented in `AGENT_GUIDE.md` §2:

| file | status per AGENT_GUIDE | lines |
|---|---|---|
| `src/consistency.py` | tested on phantom | 299 |
| `src/frc.py` | tested on phantom | 156 |
| `src/sweep.py` | never executed | 163 |
| `src/judge.py` | never executed | 148 |
| `src/split.py` | logic unit-tested only | ~130 |
| `AGENT_GUIDE.md` | the work contract | 277 |

⚠️ **Still untracked as of the latest entry.** A `git clone` of any branch does
not bring these. This blocked getting `sweep.py` into a remote sandbox; worked
around by uploading over the marimo connection instead.

---

## 2. Compute — marimo molab sandboxes

Sessions are ephemeral. **A terminated sandbox returns HTTP 410 and everything
inside it is gone.** `execute-code.sh` exits 0 even when the session is dead, so
empty output is the only symptom — check with `curl -o /dev/null -w "%{http_code}"`.

| session | GPU | fate |
|---|---|---|
| `s_mvtatl` | none (128 CPU, 1080 GB RAM) | superseded |
| `s_x0jxcq` | RTX PRO 6000 Blackwell, 102 GB | **died (410)** — took all artifacts with it |
| `s_iz10lj` | RTX PRO 6000 Blackwell, 102 GB | current |

Sandbox env: Python 3.13, torch 2.13.0+cu130, numpy 2.4.6 (README pins
`numpy<2` — divergence not yet a problem), skimage 0.26.0. **`nibabel` is
absent** — needed for BraTS, install before any real-data run. `pyvista` +
`imageio` must be pip-installed each new session for the 3D render.

Pairing: `uvx marimo@latest pair prompt --url <url> --session <id> --with-token --claude`,
then `bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <url> --session <id> --token "$(cat <tokenfile>)"`.
`npx` is unavailable locally; install the skill with `uvx deno -A npm:skills add marimo-team/marimo-pair`.

Getting code into a sandbox:
- Committed code → `git clone --branch <b> --depth 1` inside the sandbox (fast, includes the 23 MB checkpoint).
- Untracked code → tar+base64 into a generated Python script, run via `execute-code.sh`.

---

## 3. Bugs found

| # | file | bug | status |
|---|---|---|---|
| 1 | `src/metrics.py` | `_gaussian_window(win).to(pred.dtype)` set dtype but not device → SSIM crashes on any CUDA run | **fixed by friend** in 549abb6 (`.to(device=pred.device, dtype=pred.dtype)`) |
| 2 | `scripts/quality_safety_curve.py` | `build(device="cuda")` but `_load()` hardcoded `device="cpu"` → models on CPU, tensors on GPU | **fixed by friend** in 549abb6 (`_load(device)`, threads the arg through) |

Both are GPU-only and invisible on CPU, which is why the CPU smoke test never
caught them. Found by me on the GPU sandbox, reported, fixed and pushed by
douyeszn. His `_load(device)` fix is cleaner than my local `.to(device)` patch.

⚠️ Fix #1 is applied locally in `repo/src/metrics.py` but that edit is
**uncommitted** on `hassan/dev`. It is already upstream on the friend's branch.

---

## 4. Task 1 (λ sweep) — plumbing only, NOT a result

Per AGENT_GUIDE §4. Run on `s_x0jxcq`, synthetic, size 96, n=48 train / 16 test,
base=16, 5 epochs, `src/sweep.py` first execution ever.

**GATE 1 passes literally:** 6 points ✓, λ=0 present ✓, all `n_lesions`>0 (17) ✓,
no NaN in `erasure_small` ✓.

**But the run is scientifically empty and must not be quoted:**

| λ | PSNR | SSIM | Dice | erasure_small | fp_rate |
|---|---|---|---|---|---|
| 0 | 22.16 | 0.436 | 0.048 | 1.000 | 0.933 |
| 1 | 22.11 | 0.411 | 0.039 | 1.000 | 0.894 |
| 5 | 21.58 | 0.345 | 0.001 | 1.000 | 0.977 |
| 20 | 20.75 | 0.321 | 0.010 | 1.000 | 0.953 |
| 50 | 18.46 | 0.277 | 0.002 | 1.000 | 0.968 |
| 200 | 16.76 | 0.254 | 0.003 | 1.000 | 0.945 |

Why it is empty:
1. `erasure_small` is 1.000 at every λ — a flat line cannot show a knee.
2. Dice ≈ 0.005–0.05 — the segmenter is effectively not working, and every
   safety metric is computed through it.
3. PSNR falls monotonically with λ; `matched_psnr_pair(tol=0.15)` returns λ=1
   with `erasure_small_drop = 0.0` — a match with zero effect.

Cost: **4.3 s** for all six points. AGENT_GUIDE budgeted overnight on a T4, so
there is enormous headroom to scale (more slices, more epochs, larger `base`).

**Status: incomplete.** Needs a rerun at scale, and a segmenter that reaches a
non-trivial Dice before its safety numbers mean anything. `sweep.json` was lost
with session `s_x0jxcq`.

---

## 5. Friend's branch — `neuro-voxel-3d-viz`

Pure additions over `main` (1297 lines): `viz/` (vendored neuro-voxel),
`viz_bridge.py`, `render_3d.py`, `demo_3d.py`, `scripts/quality_safety_curve.py`,
`TEAM_GUIDE.md`, and later `paper/`. No files from `main` modified.

PyVista is imported lazily, so `viz_bridge.py` (volume math, pure numpy) runs
without it; only the 3D render needs it. Rendering works headless — VTK warns
`bad X server connection. DISPLAY=` and falls back to offscreen. Verify output
is non-blank rather than trusting rc=0: decode the base64 images out of
`demo_3d.html` and check colour variance. `brain3d_compare.png` is never written
to disk; it is embedded in the HTML only.

### 5.1 Run on committed checkpoint (82df2a1, before retraining)

`viz_bridge.py` reproduced `TEAM_GUIDE.md` §1 **exactly**: true 2.80 / tumor-aware
0.48 (17.2%) / distortion **0.00 cm³ (0%)**.

### 5.2 Run after retraining (`scripts/train_demo.py`, GPU, defaults)

The 0.00 cm³ headline **did not reproduce**:

| | committed ckpt | retrained |
|---|---|---|
| true | 2.80 cm³ | 2.82 cm³ |
| tumor-aware | 0.48 cm³ (17%) | **5.47 cm³ (194%)** |
| distortion | **0.00 cm³ (0%)** | **3.23 cm³ (115%)** |

The distortion model did not erase — it over-segmented. The original 0.00 came
from an undertrained quick checkpoint whose segmenter found nothing; it showed
the mechanism by accident, not by effect. Reported to douyeszn, who retired the
claim in 908aa53 ("Reframe onto measured result").

### 5.3 Rerun on fixed branch (f226cd1, session `s_iz10lj`)

Both fixes confirmed working on GPU, no patching needed. Training: 18.6 s.

Quality-safety curve (`quality_safety_curve.png`, synthetic n=40, seed 999):

| factor | distortion PSNR / erasure | tumor-aware PSNR / erasure |
|---|---|---|
| ×2 | 29.55 dB / 19.2% | 27.33 dB / 21.2% |
| ×3 | 29.08 dB / 46.2% | 27.11 dB / **30.8%** |
| ×4 | 27.97 dB / 90.4% | 26.35 dB / **50.0%** |
| ×5 | 25.64 dB / 100% | 24.84 dB / 88.5% |
| ×6 | 24.64 dB / 100% | 24.11 dB / 94.2% |
| ×8 | 22.42 dB / 100% | 22.36 dB / 100% |

3D volumes this run: true 2.82 / tumor-aware 0.60 (21.3%) / distortion 0.01 cm³ (0.2%).
This *does* match the original erasure story — but see §5.4; that is partly luck
of the draw.

### 5.4 ⚠️ Reproducibility failure — the most important finding so far

Three identical `train_demo.py` runs, same `torch.manual_seed(0)`, same GPU,
same code, evaluated on the same held-out set (synthetic n=64, seed 999):

| trial | distortion Dice | distortion erasure | tumor-aware Dice | tumor-aware erasure |
|---|---|---|---|---|
| 0 | 0.567 | 30.4% | 0.683 | 19.0% |
| 1 | 0.151 | 82.3% | 0.261 | 64.6% |
| 2 | 0.118 | 92.4% | 0.389 | 50.6% |

- **PSNR is stable** (26.3–27.0 dB). SR training is fine.
- **Everything computed through the segmenter is not.** Erasure ranges 30%→92%
  for the same model and seed.
- Cause: `train_demo.py` calls `torch.manual_seed(0)` once, but DataLoader
  shuffle order, the per-batch degradation RNG, and cuDNN nondeterminism are
  uncontrolled; a 10-epoch segmenter amplifies the noise.

**What survives:** in all three trials tumor-aware had **lower erasure and higher
Dice** than distortion-optimal. The *direction* of the claim reproduces every
time. Only the magnitudes are unusable.

**Consequence: no single-run number from this pipeline belongs on a slide.**
Report mean ± spread over N seeds instead.

### 5.5 Held-out evaluation, single run (n=64, seed 999) — magnitudes unstable, see §5.4

| | PSNR | SSIM | Dice | erasure | small-lesion erasure | FP rate | unc. AUROC |
|---|---|---|---|---|---|---|---|
| distortion | 26.91 | 0.774 | 0.147 | 83.5% | 82.4% | 0.130 | 0.897 |
| tumor-aware | 22.65 | 0.768 | 0.234 | 72.2% | 70.6% | 0.566 | 0.928 |

Note the **FP rate roughly quadruples** (0.13 → 0.57). The tumor-aware objective
trades hallucination for erasure. A judge will ask about this; it needs an
answer, and it is visible in every run, not just this one.

### 5.6 Variance root cause — diagnosed, and mostly fixed

Ablation on `s_iz10lj`. Metric is segmenter Dice on **clean** held-out HR
(synthetic n=64, seed 999), 3 runs per config, **identical seed every run**:

| config | runs | spread |
|---|---|---|
| A — current (BCE, 10 ep, cuDNN default) | 0.856 / 0.623 / 0.602 | 0.254 |
| B — as A but `cudnn.deterministic=True` | 0.555 / 0.555 / 0.555 | **0.000** |
| C — BCE+soft Dice, 10 ep | 0.784 / 0.825 / 0.444 | 0.380 |
| D — BCE+soft Dice, 30 ep | 0.892 / 0.870 / 0.850 | 0.042 |
| **E — BCE+soft Dice, 30 ep, deterministic** | **0.9006 ×3** | **0.000** |

**Cause 1 — cuDNN nondeterminism.** B isolates it: one flag takes the spread to
exactly zero. Seeding was never at fault — `degrade_batch` is seeded by `step`
and DataLoader shuffle draws from the seeded global RNG. Nondeterministic conv
backward kernels were the whole source of drift.

**Cause 2 — undertrained segmenter amplifying it.** Training masks are **2.4%
foreground** (1 lesion pixel per 41). Plain BCE at 10 epochs sits near the
decision boundary where small numeric differences flip whole lesions. Soft Dice
+ 30 epochs raises Dice 0.55 → 0.90 *and* cuts spread to 0.042 without the flag.

**Remaining, genuine instability.** Config E across **different** seeds (0,1,2):
`0.901 / 0.505 / 0.896` — mean 0.767 ± 0.185. One seed in three collapses.
Determinism buys reproducibility, not stability. Single-run numbers are still
not quotable; error bars over seeds are mandatory.

**Proposed fix (not yet applied to any branch):**
1. `torch.backends.cudnn.deterministic = True`, `benchmark = False` in `train_demo.py`.
2. `train_segmenter`: BCE + soft Dice, epochs 10 → 30.
3. Report mean ± spread over N=5 seeds.
4. If collapsed runs are excluded, do it by a stated rule (e.g. train-Dice floor)
   and disclose it.

### 5.7 Variance fix applied + 5-seed run

Applied locally on `hassan/dev` (uncommitted) and mirrored into the sandbox clone
of the friend's branch:

- `src/train.py`: new `set_deterministic(seed)` (seeds torch+numpy, sets
  `cudnn.deterministic=True`, `benchmark=False`) and `soft_dice_loss()`.
- `src/train.py`: `train_segmenter(..., dice_weight=1.0)` adds soft Dice to BCE.
  `dice_weight=0.0` restores the original BCE-only objective.
- `scripts/train_demo.py`: `--seg-epochs` default 10 → 30, new `--seed`,
  `torch.manual_seed(0)` → `set_deterministic(args.seed)`.

Five seeds, full `train_demo.py` each (~27 s per seed), evaluated on synthetic
n=64 seed 999:

| seed | clean-HR Dice | distortion erasure | tumor-aware erasure |
|---|---|---|---|
| 0 | 0.901 | 0.608 | 0.228 |
| 1 | **0.505** | 0.911 | 0.873 |
| 2 | 0.896 | 0.835 | 0.772 |
| 3 | 0.759 | 0.139 | 0.101 |
| 4 | 0.861 | 0.240 | 0.152 |

Aggregate (mean ± sd over 5 seeds):

| metric | distortion | tumor-aware |
|---|---|---|
| PSNR | 26.404 ± 0.357 | 22.739 ± 1.575 |
| SSIM | 0.797 ± 0.008 | 0.696 ± 0.038 |
| Dice | 0.339 ± 0.226 | 0.421 ± 0.247 |
| erasure | 0.547 ± 0.310 | 0.425 ± 0.329 |
| erasure_small | 0.776 ± 0.094 | 0.659 ± 0.146 |
| fp_rate | 0.329 ± 0.238 | 0.561 ± 0.160 |

**tumor-aware erasure < distortion erasure in 5/5 seeds.** The direction is the
defensible claim; absolute magnitudes still are not (erasure spans 0.139–0.911).
Remaining spread tracks segmenter quality — seed 1 collapses to clean Dice 0.505
and yields the worst erasure for both models. PSNR and SSIM are stable.

Raw per-seed JSON: `/marimo/friend/seeds5.json` (sandbox only, not downloaded).

### 5.8 Cosine schedule added — segmenter collapse eliminated, and the effect shrinks

Ablation, segmenter Dice on clean held-out HR, 5 seeds per config:

| config | Dice | spread | collapses (<0.70) |
|---|---|---|---|
| 30 ep, lr 1e-3 (previous) | 0.784 | ±0.149 | 1/5 |
| 30 ep, lr 3e-4 | 0.788 | ±0.146 | 1/5 |
| 60 ep, lr 1e-3 | 0.874 | ±0.034 | 0/5 |
| **30 ep + cosine annealing** | **0.889** | **±0.009** | **0/5** |

Cosine wins on every axis at no extra cost. Added as
`train_segmenter(..., cosine=True)`; pass `False` for the old constant LR.

**5-seed rerun with cosine** (synthetic n=64 held out, seed 999):

segmenter Dice on clean HR: **0.889 ± 0.009** — [0.886, 0.893, 0.904, 0.878, 0.881].
Collapse gone; compare §5.7's 0.505–0.901.

| metric | distortion | tumor-aware |
|---|---|---|
| PSNR | 26.491 ± 0.500 | 22.738 ± 1.524 |
| SSIM | 0.797 ± 0.009 | 0.697 ± 0.030 |
| Dice | 0.436 ± 0.060 | 0.465 ± 0.138 |
| erasure (all) | 0.405 ± 0.065 | 0.354 ± 0.130 |
| erasure (small) | 0.776 ± 0.069 | 0.682 ± 0.047 |
| fp_rate | 0.297 ± 0.079 | 0.500 ± 0.206 |

⚠️ **The effect is much weaker than §5.7 suggested.** Overall erasure now favours
tumor-aware in only **3/5** seeds (was 5/5 with the broken segmenter). Likely
explanation: with a weak segmenter the tumor-aware model's lesion-preserving
output is easier to segment, inflating its apparent advantage — exactly the
distribution-matching confound `src/judge.py` and AGENT_GUIDE Task 5 describe.
Fixing the segmenter removed part of it.

What survives, per-seed paired on **small** lesions:

| seed | small erasure dist | small erasure TA | delta | PSNR dist | PSNR TA |
|---|---|---|---|---|---|
| 0 | 0.706 | 0.647 | +0.059 | 25.96 | 25.06 |
| 1 | 0.706 | 0.647 | +0.059 | 27.35 | 20.64 |
| 2 | 0.824 | 0.647 | +0.176 | 26.06 | 21.72 |
| 3 | 0.882 | 0.706 | +0.176 | 26.39 | 22.65 |
| 4 | 0.765 | 0.765 | 0.000 | 26.69 | 23.63 |

Paired delta on small lesions: **+0.094 ± 0.079**, better in **4/5** seeds
(never worse). This is the strongest surviving claim, and it is modest.

⚠️ **Not a matched-PSNR comparison.** Tumor-aware runs 3.75 dB below distortion
(22.74 vs 26.49). Some of the erasure difference is simply the quality gap. The
λ sweep (§4, Task 1) is the only run that can separate these, and it has not
been done at scale — that is now the highest-value experiment left.

Also note **fp_rate 0.297 → 0.500**: the tumor-aware objective roughly doubles
fabricated detections. Present alongside any erasure claim, not after it.

---

## 6. Open items

1. **Apply the variance fix (highest priority).** See §5.6 — diagnosed and
   verified, but not yet applied to `hassan/dev` or the friend's branch. Until
   it lands, no safety magnitude is defensible.
2. **Commit or back up the six untracked files.** They exist only on this laptop.
   Decision so far: transfer to sandboxes by direct upload, no push (AGENT_GUIDE
   R3). That leaves them unbacked-up.
3. **`src/metrics.py` fix is uncommitted** on `hassan/dev` (already upstream on
   the friend's branch).
4. **Task 1 needs a real run** — see §4. `sweep.py` is proven to execute; it now
   needs scale plus a working segmenter.
5. **Tasks 2–5 of AGENT_GUIDE not started** — subject-level split, detector
   validation on BraTS, detector in `demo.py`, independent judge.
6. **No BraTS run yet.** Everything above is synthetic. `nibabel` must be
   installed in the sandbox first.
7. **Answer the FP-rate trade-off** (§5.5) before it is asked from the floor.

---

## 7. Deviations from AGENT_GUIDE

Recorded so they are not mistaken for oversights:

- **R3 (never push).** `hassan/dev` was pushed to origin before the guide was
  read — a new branch only, no history rewritten, nothing overwritten. No pushes
  since; sandbox transfer is by direct upload instead.
- **R4 (one task at a time, in order).** Task 1 was left at the plumbing stage to
  run the friend's branch, at the user's explicit request.
- **R5 (free tier, 16 GB T4).** The sandbox GPU is an RTX PRO 6000 Blackwell with
  102 GB — 6× the assumed VRAM. More headroom than planned, no scope change.
