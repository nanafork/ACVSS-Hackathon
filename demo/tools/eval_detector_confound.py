"""Is low segmenter-uncertainty really a danger signal, or just small lesions?

Test: (a) detector AUROC using -seg_std, (b) does seg_std track lesion area,
(c) does the detector still work WITHIN a narrow lesion-size band (size held
roughly constant, so it cannot be explained by size).
"""
import sys
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import numpy as np, torch

from src.checkpoint import load_models
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import to_mask_np
from src.models import enable_mc_dropout
from src.consistency import brain_roi

DEV = "cuda"; PASSES = 16
seg, sr_d, sr_t, meta = load_models(os.path.join(ROOT, "checkpoints", "demo.pt"), device=DEV)
SIZE, FACTOR, SIG = int(meta["size"]), int(meta["factor"]), float(meta["sigma"])
ds = make_dataset("synthetic", n=64, size=SIZE, seed=999)


def components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); out = []; nxt = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                nxt += 1; stack = [(i, j)]; lab[i, j] = nxt; pix = []
                while stack:
                    y, x = stack.pop(); pix.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = nxt; stack.append((ny, nx))
                m = np.zeros((h, w), bool)
                for y, x in pix: m[y, x] = True
                out.append(m)
    return out


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if y.all() or not y.any(): return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts); start = cum - counts
    ranks = ((start + cum + 1) / 2.0)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


rows = []
sr_d.eval(); seg.eval()
rng = np.random.default_rng(0)
for i in range(len(ds)):
    s = ds[i]
    hr = s["hr"][0].numpy(); gt = s["mask"][0].numpy() > 0.5
    if not gt.any(): continue
    lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=rng)
    x = torch.from_numpy(lr)[None, None].float().to(DEV)
    with torch.no_grad():
        det = sr_d(x); pred = to_mask_np(seg(det)) > 0.5
    enable_mc_dropout(sr_d)
    with torch.no_grad():
        outs = torch.stack([sr_d(x) for _ in range(PASSES)], 0)
        probs = torch.stack([torch.sigmoid(seg(outs[p])) for p in range(PASSES)], 0)
    sr_d.eval()
    roi = brain_roi(lr)
    pstd = (probs.std(0)[0, 0].cpu().numpy() * roi)
    les = components(gt)
    erased = [m for m in les if (m & pred).sum()/max(1, m.sum()) < 0.1]
    rows.append({"has_erasure": bool(erased),
                 "seg_std": float(pstd[roi].mean()),
                 "gt_area": int(gt.sum()),
                 "min_les": int(min(m.sum() for m in les)),
                 "n_les": len(les)})

y = [r["has_erasure"] for r in rows]
print(f"{len(rows)} slices, {sum(y)} with erasure\n")
print(f"(a) detector AUROC using -seg_std : {auroc([-r['seg_std'] for r in rows], y):.3f}")

areas = np.array([r["gt_area"] for r in rows], float)
stds = np.array([r["seg_std"] for r in rows], float)
print(f"(b) corr(seg_std, total lesion area) : {np.corrcoef(stds, areas)[0,1]:+.3f}")
print(f"    corr(seg_std, smallest lesion)   : "
      f"{np.corrcoef(stds, [r['min_les'] for r in rows])[0,1]:+.3f}")
print(f"    mean lesion area | erased {areas[np.array(y)].mean():7.1f} px")
print(f"    mean lesion area | clean  {areas[~np.array(y)].mean():7.1f} px")

# (c) control for size: split into area terciles, test within each
qs = np.quantile(areas, [1/3, 2/3])
print("\n(c) within lesion-size bands (size held roughly constant):")
for lo, hi, label in ((-1, qs[0], "small"), (qs[0], qs[1], "medium"), (qs[1], 1e9, "large")):
    idx = [k for k, r in enumerate(rows) if lo < r["gt_area"] <= hi]
    yy = [rows[k]["has_erasure"] for k in idx]
    a = auroc([-rows[k]["seg_std"] for k in idx], yy)
    print(f"    {label:<7} n={len(idx):>3}  erased={sum(yy):>3}  AUROC={a:.3f}")
