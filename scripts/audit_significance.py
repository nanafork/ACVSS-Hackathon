"""Re-test the headline result with the correct unit of analysis.

The safety rates are computed per connected component per 2D slice. One tumor
spanning twenty axial slices contributes twenty "lesions", and those twenty
outcomes rise and fall together: if the reconstruction loses that tumor, it
loses it on every slice. Treating them as independent samples inflates the
effective sample size by roughly the number of slices a tumor spans, which
makes any difference look far more significant than it is.

This script recomputes the comparison with the patient as the unit:
  * a per-patient erasure rate for each objective,
  * a paired test across patients (each patient sees both models),
  * a patient-level bootstrap confidence interval on the difference.

    python scripts/audit_significance.py --ckpt checkpoints/demo_et.pt \
        --cache data/slices_et.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import load_models  # noqa: E402
from src.data import make_dataset  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.metrics import lesion_records, to_mask_np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/demo_et.pt")
    ap.add_argument("--cache", default="data/slices_et.npz")
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg, sr_d, sr_t, meta = load_models(args.ckpt, device=device)
    for m in (seg, sr_d, sr_t):
        m.eval()
    ds = make_dataset("cached", path=args.cache, split=args.split)
    factor, sigma = int(meta["factor"]), float(meta["sigma"])
    print(f"{len(ds)} slices from {ds.n_cases()} patients ({args.ckpt})")

    # Per-patient tallies: lesions seen, lesions erased, for each objective.
    tally = {}
    for i in range(len(ds)):
        s = ds[i]
        case = int(ds.case_id[i])
        hr_np = s["hr"][0].numpy()
        gt = s["mask"][0].numpy()
        lr_np = degrade(hr_np, factor=factor, sigma=sigma,
                        rng=np.random.default_rng(7 + i))
        lr = torch.from_numpy(lr_np)[None, None].float().to(device)
        with torch.no_grad():
            preds = {"distortion": to_mask_np(seg(sr_d(lr))),
                     "tumor-aware": to_mask_np(seg(sr_t(lr)))}
        t = tally.setdefault(case, {k: [0, 0] for k in preds})
        for name, pm in preds.items():
            for rec in lesion_records(gt, pm):
                t[name][0] += 1
                t[name][1] += 0 if rec["detected"] else 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    cases = sorted(tally)
    rate = {n: np.array([tally[c][n][1] / max(1, tally[c][n][0]) for c in cases])
            for n in ("distortion", "tumor-aware")}
    n_les = sum(tally[c]["distortion"][0] for c in cases)

    print(f"\nlesion cross-sections: {n_les}   patients: {len(cases)}")
    print(f"  pooled erasure   distortion {np.concatenate([[tally[c]['distortion'][1]] for c in cases]).sum()/n_les:.4f}"
          f"   tumor-aware {np.concatenate([[tally[c]['tumor-aware'][1]] for c in cases]).sum()/n_les:.4f}")

    d = rate["distortion"] - rate["tumor-aware"]
    print("\n--- patient as the unit of analysis ---")
    print(f"  mean per-patient erasure   distortion {rate['distortion'].mean():.4f}"
          f"   tumor-aware {rate['tumor-aware'].mean():.4f}")
    print(f"  mean paired difference     {d.mean():+.4f}")
    print(f"  patients helped {int((d > 0).sum())}, hurt {int((d < 0).sum())}, "
          f"unchanged {int((d == 0).sum())}")

    # Paired bootstrap over patients: resample patients, not lesions.
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(args.boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  95% CI (patient bootstrap) [{lo:+.4f}, {hi:+.4f}]")
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    print(f"  two-sided bootstrap p      {p_boot:.4f}")

    # Wilcoxon signed-rank, no distributional assumption.
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        if len(nz):
            print(f"  Wilcoxon signed-rank p     {wilcoxon(nz).pvalue:.4f} (n={len(nz)})")
    except ImportError:
        pass

    naive = np.sqrt(0.25 / n_les)
    print(f"\n  naive SE if lesions were independent: {naive:.4f}")
    print(f"  actual SE across patients:            {d.std(ddof=1)/np.sqrt(len(d)):.4f}")
    print(f"  the naive figure is optimistic by ~{(d.std(ddof=1)/np.sqrt(len(d)))/naive:.1f}x")


if __name__ == "__main__":
    main()
