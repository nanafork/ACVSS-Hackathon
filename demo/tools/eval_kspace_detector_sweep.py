"""Threshold sweep for the reference-free detector, on the pipeline's own output.

AGENT_GUIDE Task 4: do NOT just lower the threshold until something fires --
run the sweep and report it. Sensitivity = fraction of slices that really had an
erased lesion and got flagged. False alarm = fraction of slices with no erased
lesion that got flagged anyway.
"""
import sys
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import numpy as np, torch, json

from src.checkpoint import load_models
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import to_mask_np
from src.consistency import consistency_maps, flag_regions

DEV = "cuda"
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


# Cache the z-maps once; the sweep is then pure thresholding.
cache = {}
for name, model in (("distortion", sr_d), ("tumor_aware", sr_t)):
    model.eval(); seg.eval()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        hr = s["hr"][0].numpy(); gt = s["mask"][0].numpy() > 0.5
        if not gt.any():
            continue
        lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=rng)
        with torch.no_grad():
            out = model(torch.from_numpy(lr)[None, None].float().to(DEV))
            pred = to_mask_np(seg(out)) > 0.5
        sr = out[0, 0].cpu().numpy()
        les = components(gt)
        erased = [m for m in les if (m & pred).sum()/max(1, m.sum()) < 0.1]
        mm = consistency_maps(sr, lr, FACTOR)
        rows.append({"z_erased": mm["erased"], "has_erasure": bool(erased)})
    cache[name] = rows
    print(f"{name}: {len(rows)} slices, "
          f"{sum(r['has_erasure'] for r in rows)} with erasure", flush=True)

print(f"\n{'z_thr':>6} {'min_area':>9} | "
      f"{'DIST sens':>10} {'DIST FA':>8} | {'TA sens':>8} {'TA FA':>7}")
best = None
for z in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    for a in (4, 8, 16, 32):
        line = [f"{z:>6.1f} {a:>9d} |"]
        stats = {}
        for name in ("distortion", "tumor_aware"):
            rows = cache[name]
            pos = [r for r in rows if r["has_erasure"]]
            neg = [r for r in rows if not r["has_erasure"]]
            sens = np.mean([flag_regions(r["z_erased"], z, a)[1] != [] for r in pos]) if pos else float("nan")
            fa = np.mean([flag_regions(r["z_erased"], z, a)[1] != [] for r in neg]) if neg else float("nan")
            stats[name] = (sens, fa)
            line.append(f" {100*sens:>9.0f}% {100*fa:>7.0f}% |")
        print("".join(line), flush=True)
        sd, fd = stats["distortion"]
        score = sd - fd
        if best is None or score > best[0]:
            best = (score, z, a, stats)

print(f"\nbest by (sensitivity - false alarm) on the distortion model: "
      f"z_thresh={best[1]}, min_area={best[2]}")
print(f"  distortion sens={100*best[3]['distortion'][0]:.0f}% fa={100*best[3]['distortion'][1]:.0f}%")
print(f"  tumor_aware sens={100*best[3]['tumor_aware'][0]:.0f}% fa={100*best[3]['tumor_aware'][1]:.0f}%")
