"""Does the MC-dropout flag actually catch a tumor that vanished?

The claim Toufiq's framing needs is lesion-level: "this tumor disappeared, and we
caught it." A pixel-level uncertainty-vs-error AUROC does NOT establish that --
it can be high simply because uncertainty concentrates near lesion borders in
slices that are already bad. Today's synthetic run showed exactly that gap:
pixel AUROC 0.824 while slice-level detection sat at chance.

So we measure three granularities, weakest assumption last:

  (1) PIXEL   uncertainty vs segmentation error, all pixels pooled.
              This is the number that already exists (0.85/0.82).
  (2) SLICE   does this slice contain an erased lesion?  Score = aggregate
              uncertainty over the brain. DEPLOYABLE: no ground truth used.
  (3) REGION  threshold the uncertainty map, keep blobs, and ask whether a blob
              lands on the erased lesion.  DEPLOYABLE and localising -- this is
              what a radiologist would actually be shown.

  (0) ORACLE  uncertainty measured inside the true lesion mask. NOT deployable
              (it needs the answer to find the question); reported only as the
              ceiling the deployable numbers are chasing.

Real BraTS (MSD Task01), enhancing tumor, test split only -- cases no model has
seen. Dropout is restored to eval mode after every MC block: leaving it on was
the bug that corrupted the earlier published safety numbers.
"""
import sys, json, os
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "safety", "assets")
CACHE = os.path.join(ASSETS, "data", "et_full.npz")
CKPT = os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt")
sys.path.insert(0, ROOT)

import numpy as np, torch

from src.data import make_dataset
from src.metrics import to_mask_np
from src.models import enable_mc_dropout

DEV = "cuda"
PASSES = 16
CKPT = CKPT
CACHE = CACHE

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
meta = ck.get("meta", {})
FACTOR = int(meta.get("factor", 4))
SIG = float(meta.get("sigma", 0.03))
print("checkpoint:", os.path.basename(CKPT), "meta:", meta, flush=True)

from src.models import seg_unet, sr_unet
seg = seg_unet(base=32).to(DEV); seg.load_state_dict(ck["seg"]); seg.eval()
models = {}
for key, sk in (("distortion", "sr_distortion"), ("tumor_aware", "sr_tumor_aware")):
    m = sr_unet(base=32, dropout=0.2).to(DEV)
    m.load_state_dict(ck[sk]); m.eval()
    models[key] = m

ds = make_dataset("cached", path=CACHE, split="test")
print(f"test slices: {len(ds)}", flush=True)

from src.degrade import degrade


def components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); out = []; nxt = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                nxt += 1; stack = [(i, j)]; lab[i, j] = nxt; pix = []
                while stack:
                    y, x = stack.pop(); pix.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and lab[ny,nx]==0:
                            lab[ny,nx] = nxt; stack.append((ny,nx))
                m = np.zeros((h,w), bool)
                for y,x in pix: m[y,x] = True
                out.append(m)
    return out


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts); start = cum - counts
    ranks = ((start + cum + 1) / 2.0)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos*(npos+1)/2.0) / (npos*nneg))


def size_bin(n):
    return "small" if n < 50 else ("medium" if n <= 200 else "large")


LIMIT = int(os.environ.get("LIMIT", "1200"))
out = {}
for name, model in models.items():
    slice_scores, slice_labels = [], []
    pix_u, pix_e = [], []
    lesion_rows = []
    region_hit = {"small": [0,0], "medium": [0,0], "large": [0,0]}
    n_done = 0
    for i in range(min(LIMIT, len(ds))):
        s = ds[i]
        hr = s["hr"][0].numpy().astype(np.float32)
        gt = s["mask"][0].numpy() > 0.5
        if not gt.any():
            continue
        lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=np.random.default_rng(1000+i))
        x = torch.from_numpy(lr)[None, None].float().to(DEV)

        model.eval()                                   # deterministic reconstruction
        with torch.no_grad():
            sr = model(x)
            pred = to_mask_np(seg(sr)) > 0.5

        enable_mc_dropout(model)                       # MC block
        with torch.no_grad():
            outs = torch.stack([model(x) for _ in range(PASSES)], 0)
            probs = torch.stack([torch.sigmoid(seg(outs[p])) for p in range(PASSES)], 0)
        model.eval()                                   # RESTORE (the old leak)

        pstd = probs.std(0)[0, 0].cpu().numpy()
        brain = lr > 0.05

        les = components(gt)
        erased = [m for m in les if (m & pred).sum()/max(1, m.sum()) < 0.1]

        # (1) pixel: uncertainty vs segmentation error
        err = (pred != gt) & brain
        if brain.any():
            pix_u.append(pstd[brain][::7]); pix_e.append(err[brain][::7])

        # (2) slice: deployable aggregate, no ground truth
        slice_scores.append(float(pstd[brain].mean()) if brain.any() else 0.0)
        slice_labels.append(bool(erased))

        # (3) region: threshold uncertainty, does a blob land on the erased lesion?
        thr = np.percentile(pstd[brain], 99.0) if brain.any() else 1.0
        flag = (pstd >= thr) & brain
        blobs = components(flag)
        for m in les:
            b = size_bin(int(m.sum()))
            was_erased = (m & pred).sum()/max(1, m.sum()) < 0.1
            hit = any((bl & m).any() for bl in blobs)
            if was_erased:
                region_hit[b][0] += int(hit); region_hit[b][1] += 1
            # (0) oracle: uncertainty inside the true lesion
            lesion_rows.append({"size": int(m.sum()), "bin": b,
                                "erased": bool(was_erased),
                                "u_in": float(pstd[m].mean())})
        n_done += 1

    pu = np.concatenate(pix_u); pe = np.concatenate(pix_e)
    res = {
        "n_slices": n_done,
        "pixel_auroc": auroc(pu, pe),
        "slice_auroc": auroc(slice_scores, slice_labels),
        "n_slices_with_erasure": int(sum(slice_labels)),
        "oracle_lesion_auroc": auroc([r["u_in"] for r in lesion_rows],
                                     [r["erased"] for r in lesion_rows]),
        "n_lesions": len(lesion_rows),
        "region_recall": {b: (region_hit[b][0]/region_hit[b][1] if region_hit[b][1] else float("nan"))
                          for b in region_hit},
        "region_n": {b: region_hit[b][1] for b in region_hit},
    }
    for b in ("small", "medium", "large"):
        rows = [r for r in lesion_rows if r["bin"] == b]
        res[f"oracle_auroc_{b}"] = auroc([r["u_in"] for r in rows], [r["erased"] for r in rows])
        res[f"n_{b}"] = len(rows)
    out[name] = res
    print(f"\n=== {name} ===", flush=True)
    print(f"  slices scored {res['n_slices']}, lesions {res['n_lesions']}, "
          f"slices containing an erased lesion {res['n_slices_with_erasure']}")
    print(f"  (1) PIXEL  uncertainty vs error AUROC      {res['pixel_auroc']:.3f}")
    print(f"  (2) SLICE  'contains erased lesion' AUROC  {res['slice_auroc']:.3f}   <- deployable")
    print(f"  (0) ORACLE lesion-level AUROC              {res['oracle_lesion_auroc']:.3f}   "
          f"(small {res['oracle_auroc_small']:.3f}, med {res['oracle_auroc_medium']:.3f}, "
          f"large {res['oracle_auroc_large']:.3f})")
    print(f"  (3) REGION recall on erased lesions        "
          f"small {100*res['region_recall']['small']:.0f}% (n={res['region_n']['small']}), "
          f"med {100*res['region_recall']['medium']:.0f}% (n={res['region_n']['medium']}), "
          f"large {100*res['region_recall']['large']:.0f}% (n={res['region_n']['large']})",
          flush=True)

json.dump(out, open(os.path.join(ROOT, "safety", "results", "out.json"), "w"), indent=2)
print("\nwrote /marimo/lesion_detect.json")
