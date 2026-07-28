# Pending

Running list. Tick items off, add new ones as they surface. Results and
checkpoint provenance live in [EXPERIMENTS.md](EXPERIMENTS.md).

## Blocking the presentation

- [ ] **Paper results section is half-rewritten.** `tumor_aware_sr_paper.tex`
      now carries the real ET and WT tables, but the abstract, discussion and
      limitations still describe the synthetic study. PDFs not rebuilt.
- [ ] **`proposal.tex` still has the old synthetic numbers** (27.8% → 15.2%,
      PSNR 26.23/19.89). Those come from a checkpoint nothing in the repo
      reproduces. They are the least defensible figures we have.
- [ ] **Quality-safety sweep not rerun on real data.** `quality_safety_curve.png`
      and the paper's sweep table are synthetic. Either regenerate with
      `scripts/quality_safety_curve.py` against `demo_et.pt` or drop the claim.

## Demo

- [x] Consolidate into `main_demo.py` → `main_demo.html`, slide deck with nav
- [x] Validated colorblind-safe palette (`src/palette.py`)
- [x] Repoint demo at the real ET checkpoint and real held-out slices
- [x] Real 3D case (BRATS_038) instead of the phantom
- [x] Zero-pad the volume before marching cubes (open faces at the crop
      boundary were rendering as flat grey sheets across the scene)
- [x] Bigger viewports: 2×2 grid, renders at 820×780
- [x] Regenerate the rotating GIF from the padded 192-wide case
- [x] Reword the config caption so it refers to training, not the render
- [ ] The 2×2 grid pushes the bottom row below the fold at 1400×1150. Fine when
      scrolling, not ideal for a projector. Either shrink the panel height or
      split into two slides.
- [ ] **Cache the 3D pipeline output.** Every rebuild reruns inference on 80
      slices for 3 models plus 10 MC passes, marching cubes, and a 36-frame
      GIF: 5-8 minutes. Writing `build_patient_volumes` output to an npz and
      reusing it would make visual iteration seconds instead of minutes. This
      is what made today's demo work slow.

## Data and compute

- [x] **Lesion-centred slice window** (was: middle 12 slices of the volume,
      which used 8% of each case and silently dropped 130 of 484 patients whose
      enhancing tumor sat outside the mid-axial window). Default raised to 48.
      **Not yet re-extracted or retrained** — needs a GPU box.
- [ ] **The sandbox terminated (HTTP 410).** Everything downloaded and verified
      first; lost: the 48-slice extraction in progress, `slices_128.npz`,
      `demo_gpu.pt`, and the 7.6 GB MSD tar. All regenerable.
- [ ] **Need a new GPU box** to redo extraction + training on the full data.
- [ ] **Need an HF token** (write scope) + target repo to push checkpoints
      during training, so the next termination does not cost a run.
- [ ] **Uncertainty-gated output**: flag or abstain where MC-dropout std is
      high, instead of presenting a uniformly crisp image that hides its own
      doubt. AUROC 0.85/0.82 on real data says the signal is there. Needs no
      GPU; the most defensible improvement still available.

- [x] Real BraTS via MSD Task01 (no registration wall), 484 cases
- [x] Case-level train/test split, leakage asserted
- [x] `demo_brats.pt`, `demo_et.pt`, results JSON, logs, ET slice cache — all
      downloaded and md5-verified
- [ ] **`slices_128.npz` (98 MB, whole-tumor cache) not downloaded.** Only
      needed to re-run the WT evaluation locally; regenerable from
      `prepare_msd.py`.
- [ ] **`demo_gpu.pt` not downloaded.** Synthetic size-128 model; no
      evidentiary value given the phantom is degenerate. Probably let it go.
- [ ] **The sandbox is ephemeral.** Anything still only on it is at risk.

## Open questions worth an experiment

- [ ] The segmenter is the measuring instrument and only reaches Dice 0.69 on
      real data. A stronger segmenter might widen or shrink the gap between the
      two objectives. Currently unknown.
- [ ] Tumor weight is fixed at 40. No sweep has been run on real data, so we
      cannot say whether the ET effect is near its optimum.
- [ ] `seg_consistency_loss` exists in `src/losses.py` and has never been
      enabled in any run.
- [ ] Enhancing tumor erasure is still 45% at best. Both objectives miss
      roughly half of small enhancing lesions. That is the honest ceiling of
      this pipeline and is worth stating rather than burying.

## Housekeeping

- [ ] `data/slices_et.npz` (60 MB) was committed once before being untracked;
      it is still in git history. Left alone deliberately — rewriting history
      on a shared branch is worse than the bloat.
- [ ] `paper/PROPOSAL.md` has no contributions section and has drifted from
      `proposal.tex`.
- [ ] Optional courtesy PR upstreaming the neuro-voxel fixes.
