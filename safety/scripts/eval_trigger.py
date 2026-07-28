"""How trustworthy is the erasure alarm?

The trigger, exactly as it would run in a hospital:

  regions = connected components of  seg(acquired scan) > 0.5
  for each region:
      if it has (almost) no overlap with seg(enhanced image) > 0.5:
          RAISE FLAG

No ground truth is used to raise the flag. Ground truth is used only here, to
audit the flags afterwards:

  TRUE  the region sits on a real lesion  -> a real tumor vanished
  FALSE the region sits on no lesion      -> the segmenter hallucinated it in the
                                             input and the flag is noise

Also measured: how many real vanishings the trigger MISSES (a lesion the
acquired-scan segmentation never found cannot be flagged), because that is the
honest ceiling of this design.
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

OP = 0.5


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
x, gt_all = d["x"], d["gt"]

flags_true = flags_false = 0
flag_by_bin = {b: [0, 0] for b in ("small", "medium", "large")}   # [true, total]
slices_with_flag = 0
n_slices = len(x)
missed_vanish = caught_vanish = 0

for i in range(n_slices):
    p_lr = x[i, 3].astype(np.float32)
    p_sr = x[i, 2].astype(np.float32)
    gt = gt_all[i] > 0
    lr_hit = p_lr > OP
    sr_hit = p_sr > OP

    fired_here = False
    for region in components(lr_hit):
        if region.sum() < 4:
            continue
        survived = (region & sr_hit).sum() / max(1, region.sum()) >= 0.1
        if survived:
            continue
        fired_here = True
        b = size_bin(int(region.sum()))
        on_tumor = (region & gt).sum() / max(1, region.sum()) >= 0.1
        flag_by_bin[b][1] += 1
        if on_tumor:
            flags_true += 1; flag_by_bin[b][0] += 1
        else:
            flags_false += 1
    slices_with_flag += int(fired_here)

    # ceiling: real lesions that vanished but were never visible in the input
    for m in components(gt):
        in_lr = (m & lr_hit).sum() / max(1, m.sum()) >= 0.1
        in_sr = (m & sr_hit).sum() / max(1, m.sum()) >= 0.1
        if not in_sr:                       # gone after enhancement
            if in_lr:
                caught_vanish += 1          # flaggable
            else:
                missed_vanish += 1          # invisible in the input too -> unflaggable

tot = flags_true + flags_false
print(f"slices scored: {n_slices}\n")
print("=== when the alarm fires, is it real? ===")
print(f"  flags raised            {tot}")
print(f"  on a real lesion        {flags_true}   ({100*flags_true/max(1,tot):.1f}%  <- precision)")
print(f"  on nothing (segmenter FP in the input) {flags_false}   ({100*flags_false/max(1,tot):.1f}%)")
print(f"  slices raising >=1 flag {slices_with_flag} of {n_slices} "
      f"({100*slices_with_flag/n_slices:.1f}%)")
print(f"\n  by flagged-region size:")
for b in ("small", "medium", "large"):
    t, n = flag_by_bin[b]
    if n:
        print(f"    {b:<7} {n:>5} flags, {100*t/n:>5.1f}% on a real lesion")

print("\n=== ceiling of this design ===")
tot_v = caught_vanish + missed_vanish
print(f"  lesions absent after enhancement      {tot_v}")
print(f"  visible in the acquired scan (flaggable) {caught_vanish} "
      f"({100*caught_vanish/max(1,tot_v):.1f}%)")
print(f"  invisible in both (segmenter floor)      {missed_vanish} "
      f"({100*missed_vanish/max(1,tot_v):.1f}%)  <- this design cannot see these")

json.dump({"flags": tot, "precision": flags_true/max(1,tot),
           "flags_true": flags_true, "flags_false": flags_false,
           "slices_with_flag": slices_with_flag, "n_slices": n_slices,
           "flaggable": caught_vanish, "unflaggable": missed_vanish},
          open(os.path.join(ROOT, "safety", "results", "out.json"), "w"), indent=2)
print("\nwrote /marimo/watchdog/trigger.json")
