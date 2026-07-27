"""Can a ground-truth-free score tell which slices had a lesion erased?

Same slice-level test that the k-space detector failed. Ground truth is used
ONLY to label slices for scoring; every candidate score is computed from the
model's own behaviour.

Scores tested (all reference-free):
  unc_mean / unc_max  SR MC-dropout pixel std over the brain ROI
  seg_std             mean std of the segmenter's probability across MC passes
  seg_area_cv         coefficient of variation of predicted lesion area across
                      passes -- "does the segmenter change its mind about how
                      much tumor is here?"
  seg_count_flip      fraction of passes whose predicted lesion count differs
                      from the modal count
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

DEV = "cuda"
PASSES = 16
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
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts); start = cum - counts
    ranks = ((start + cum + 1) / 2.0)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


for name, model in (("distortion", sr_d), ("tumor_aware", sr_t)):
    seg.eval()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        hr = s["hr"][0].numpy(); gt = s["mask"][0].numpy() > 0.5
        if not gt.any():
            continue
        lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=rng)
        x = torch.from_numpy(lr)[None, None].float().to(DEV)

        model.eval()
        with torch.no_grad():
            det = model(x)
            pred_det = to_mask_np(seg(det)) > 0.5

        enable_mc_dropout(model)
        with torch.no_grad():
            outs = torch.stack([model(x) for _ in range(PASSES)], 0)      # (P,1,1,H,W)
            probs = torch.stack([torch.sigmoid(seg(outs[p])) for p in range(PASSES)], 0)
        model.eval()

        roi = brain_roi(lr)
        unc = outs.std(0)[0, 0].cpu().numpy() * roi
        pstd = probs.std(0)[0, 0].cpu().numpy() * roi
        masks = (probs[:, 0, 0] > 0.5).cpu().numpy()
        areas = masks.reshape(PASSES, -1).sum(1).astype(float)
        counts = [len(components(m)) for m in masks]
        modal = max(set(counts), key=counts.count)

        les = components(gt)
        erased = [m for m in les if (m & pred_det).sum() / max(1, m.sum()) < 0.1]

        rows.append({
            "has_erasure": bool(erased),
            "unc_mean": float(unc[roi].mean()) if roi.any() else 0.0,
            "unc_max": float(unc.max()),
            "seg_std": float(pstd[roi].mean()) if roi.any() else 0.0,
            "seg_area_cv": float(areas.std() / areas.mean()) if areas.mean() > 0 else 0.0,
            "seg_count_flip": float(np.mean([c != modal for c in counts])),
        })

    y = [r["has_erasure"] for r in rows]
    print(f"\n=== {name} ===  {len(rows)} slices, {sum(y)} with an erased lesion")
    for k in ("unc_mean", "unc_max", "seg_std", "seg_area_cv", "seg_count_flip"):
        a = auroc([r[k] for r in rows], y)
        flag = "  <-- usable" if a == a and (a > 0.70 or a < 0.30) else ""
        print(f"  slice-level AUROC  {k:<15} {a:.3f}{flag}")
