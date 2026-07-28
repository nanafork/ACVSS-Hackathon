"""How much damage does super-resolution itself do, and can we see it without labels?

Three lesion outcomes, per lesion, using the SAME frozen segmenter throughout:

  found on the acquired (LR) scan?   found on the enhanced (SR) image?
  ------------------------------------------------------------------
  yes / yes   fine
  yes / NO    <- SR DAMAGE. The tumor was visible in what the scanner produced
                 and the enhancement removed it. This is the thing a safety
                 layer must catch, and it is DIRECTLY OBSERVABLE at inference:
                 segment both images and compare. No ground truth required.
  NO  / no    the segmenter's own floor -- not caused by SR
  NO  / yes   SR recovered a lesion the acquired scan lost (the upside)

Everything is computed from the cached arrays: lr, sr, p_sr, p_lr, gt, pred.
"""
import sys, json
sys.path.insert(0, ROOT)
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "safety", "assets")
CACHE = os.path.join(ASSETS, "data", "et_full.npz")
CKPT = os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt")

import numpy as np

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


def size_bin(n):
    return "small" if n < 50 else ("medium" if n <= 200 else "large")


d = np.load(os.path.join(ASSETS, "val_cache.npz"))
x, gt_all, pred_all = d["x"], d["gt"], d["pred"]
print(f"val slices: {len(x)}", flush=True)

cells = {b: {"ok": 0, "damage": 0, "floor": 0, "rescued": 0} for b in ("small","medium","large")}
caught = {b: [0, 0] for b in ("small","medium","large")}   # [caught, damage]
fa = {b: [0, 0] for b in ("small","medium","large")}       # false alarms on healthy lesions

for i in range(len(x)):
    p_lr = x[i, 3].astype(np.float32)
    gt = gt_all[i] > 0
    sr_pred = pred_all[i] > 0
    lr_pred = p_lr > 0.5
    if not gt.any():
        continue
    for m in components(gt):
        b = size_bin(int(m.sum()))
        in_lr = (m & lr_pred).sum() / max(1, m.sum()) >= 0.1
        in_sr = (m & sr_pred).sum() / max(1, m.sum()) >= 0.1
        if in_lr and in_sr:
            cells[b]["ok"] += 1
        elif in_lr and not in_sr:
            cells[b]["damage"] += 1
            caught[b][1] += 1
            # the observable rule: lesion-shaped region present in LR seg, absent in SR seg
            caught[b][0] += 1          # by construction the rule sees exactly this
        elif not in_lr and not in_sr:
            cells[b]["floor"] += 1
        else:
            cells[b]["rescued"] += 1

print("\n=== lesion outcomes, acquired (LR) vs enhanced (SR), same segmenter ===")
print(f"{'size':<8}{'n':>7}{'both found':>12}{'SR DAMAGE':>12}{'floor (neither)':>17}{'SR rescued':>12}")
tot = {"ok":0,"damage":0,"floor":0,"rescued":0}
for b in ("small","medium","large"):
    c = cells[b]; n = sum(c.values())
    for k in tot: tot[k] += c[k]
    if n:
        print(f"{b:<8}{n:>7}{c['ok']:>12}{c['damage']:>12}{c['floor']:>17}{c['rescued']:>12}")
n = sum(tot.values())
print(f"{'ALL':<8}{n:>7}{tot['ok']:>12}{tot['damage']:>12}{tot['floor']:>17}{tot['rescued']:>12}")
print(f"\n  SR damage rate      : {100*tot['damage']/n:.2f}% of all lesions")
print(f"  segmenter floor     : {100*tot['floor']/n:.2f}% (invisible in BOTH -- not SR's fault)")
print(f"  SR rescued          : {100*tot['rescued']/n:.2f}% (enhancement recovered a lost lesion)")
print(f"  net effect of SR    : {tot['rescued'] - tot['damage']:+d} lesions "
      f"({100*(tot['rescued']-tot['damage'])/n:+.2f} points)")

json.dump({b: cells[b] for b in cells} | {"total": tot},
          open(os.path.join(ROOT, "safety", "results", "out.json"), "w"), indent=2)
print("\nwrote /marimo/watchdog/sr_damage.json")
