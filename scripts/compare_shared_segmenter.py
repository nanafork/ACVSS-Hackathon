"""Re-score every sweep config against ONE shared segmenter.

The sweep trained a fresh segmenter per config, which broke the comparison: the
segmenter is the instrument that measures both safety rates, so each config was
graded with a different ruler. The giveaway was the low-resolution baseline,
which uses no reconstruction at all and depends only on the segmenter, yet came
out at erasure 0.622 / 0.715 / 0.719 across three configs where it should have
been identical.

Retraining every config against a shared segmenter is the complete fix and needs
a GPU. This is the part that does not: take the *already trained* SR model pairs
and score all of them with a single frozen segmenter. The measurement confound
disappears entirely -- by construction the low-resolution row is now identical
for every config, which is the check that this worked.

What this does not fix: for the seg_lambda > 0 configs, the consistency term used
that config's own segmenter *during training*. So the models are not identical to
what a fully shared-segmenter run would produce. The comparison of outcomes is
now fair; the training conditions still differ slightly. Stated rather than
hidden.

    python scripts/compare_shared_segmenter.py --seg checkpoints/demo_et_v2.pt \\
        --cache data/slices_et_full.npz --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import load_models  # noqa: E402
from src.data import make_dataset  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.metrics import (aggregate_safety, dice, hallucination_stats,  # noqa: E402
                         lesion_records, to_mask_np)

CONFIGS = [
    ("w=40 segl=0.0", "checkpoints/demo_et_v2.pt"),
    ("w=40 segl=0.5", "checkpoints/ck_w40_sl0.5.pt"),
    ("w=80 segl=0.5", "checkpoints/ck_w80_sl0.5.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="checkpoints/demo_et_v2.pt",
                    help="checkpoint whose segmenter becomes the shared instrument")
    ap.add_argument("--cache", default="data/slices_et_full.npz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="results/shared_segmenter_val.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg, _, _, meta = load_models(args.seg, device=device)
    seg.eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    factor, sigma = int(meta["factor"]), float(meta["sigma"])

    ds = make_dataset("cached", path=args.cache, split=args.split)
    print(f"{len(ds)} slices / {ds.n_cases()} patients, shared segmenter from "
          f"{os.path.basename(args.seg)}", flush=True)

    models = {}
    for label, path in CONFIGS:
        _, sr_d, sr_t, m = load_models(path, device=device)
        sr_d.eval(); sr_t.eval()
        models[label] = (sr_d, sr_t, m)

    # Every version of every slice is scored by the same segmenter in the same
    # pass, so the degraded input is identical across configs too.
    acc = {}
    for label in models:
        for v in ("lowres", "distortion", "tumor-aware"):
            acc[(label, v)] = {"records": [], "halluc": [], "dice": []}
    floor = {"records": [], "halluc": [], "dice": []}
    per_pat = {}

    for i in range(len(ds)):
        s = ds[i]
        case = int(ds.case_id[i])
        hr = s["hr"][None].to(device)
        gt = s["mask"][0].numpy()
        lr_np = degrade(hr[0, 0].cpu().numpy(), factor=factor, sigma=sigma,
                        rng=np.random.default_rng(7 + i))
        lr = torch.from_numpy(lr_np)[None, None].float().to(device)

        with torch.no_grad():
            pm = to_mask_np(seg(hr))
        floor["records"].extend(lesion_records(gt, pm))
        floor["halluc"].append(hallucination_stats(gt, pm))
        floor["dice"].append(dice(pm, gt))
        t = per_pat.setdefault(case, {})
        f = t.setdefault("floor", [0, 0])
        for r in lesion_records(gt, pm):
            f[0] += 1
            f[1] += 0 if r["detected"] else 1

        for label, (sr_d, sr_t, _) in models.items():
            with torch.no_grad():
                vers = {"lowres": lr, "distortion": sr_d(lr), "tumor-aware": sr_t(lr)}
            for v, img in vers.items():
                with torch.no_grad():
                    pred = to_mask_np(seg(img))
                a = acc[(label, v)]
                recs = lesion_records(gt, pred)
                a["records"].extend(recs)
                a["halluc"].append(hallucination_stats(gt, pred))
                a["dice"].append(dice(pred, gt))
                c = t.setdefault((label, v), [0, 0])
                for r in recs:
                    c[0] += 1
                    c[1] += 0 if r["detected"] else 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    def rate(a):
        s = aggregate_safety(a["records"], a["halluc"])
        return s, float(np.mean(a["dice"]))

    fs, fdice = rate(floor)
    out = {"split": args.split, "n_slices": len(ds), "n_patients": ds.n_cases(),
           "shared_segmenter": args.seg,
           "floor": {"fner": fs["false_negative_erasure_rate"],
                     "fpdr": fs["false_positive_detection_rate"], "dice": fdice},
           "configs": {}}

    cases = sorted(per_pat)
    fl_pp = np.array([per_pat[c]["floor"][1] / max(1, per_pat[c]["floor"][0]) for c in cases])
    print(f"\nshared segmenter floor: FNER {fs['false_negative_erasure_rate']:.4f}  "
          f"Dice {fdice:.3f}   (per-patient mean {fl_pp.mean():.4f})")
    print(f"\n{'config':16s} {'lowres':>8} {'distort':>8} {'ours':>8} {'gain':>8} "
          f"{'halluc+':>8} {'DiceOurs':>9}")
    for label in models:
        row = {}
        for v in ("lowres", "distortion", "tumor-aware"):
            s, d = rate(acc[(label, v)])
            row[v] = {"fner": s["false_negative_erasure_rate"],
                      "fpdr": s["false_positive_detection_rate"], "dice": d,
                      "by_size": s["erasure_rate_by_size"]}
        gain = row["distortion"]["fner"] - row["tumor-aware"]["fner"]
        pen = row["tumor-aware"]["fpdr"] - row["distortion"]["fpdr"]
        # Patient-level paired difference, resampling patients not lesions.
        d = np.array([
            per_pat[c][(label, "distortion")][1] / max(1, per_pat[c][(label, "distortion")][0])
            - per_pat[c][(label, "tumor-aware")][1] / max(1, per_pat[c][(label, "tumor-aware")][0])
            for c in cases])
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row["patient_level"] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                                "helped": int((d > 0).sum()), "hurt": int((d < 0).sum())}
        out["configs"][label] = row
        print(f"{label:16s} {row['lowres']['fner']:8.4f} {row['distortion']['fner']:8.4f} "
              f"{row['tumor-aware']['fner']:8.4f} {100*gain:+7.2f}pp {pen:+8.3f} "
              f"{row['tumor-aware']['dice']:9.3f}")

    lows = {round(out["configs"][l]["lowres"]["fner"], 6) for l in models}
    print(f"\nlowres identical across configs: {len(lows) == 1}  {lows}")
    print("  (this is the check: lowres uses no SR model, so with one shared")
    print("   segmenter it MUST be identical. In the original sweep it was not.)")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
