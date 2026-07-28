"""Domain-robust segmenter, then re-test the erasure trigger with it.

The deployed segmenter is trained on clean HR only (Dice 0.842 HR / 0.680 LR /
0.779 SR). Every comparison-based erasure trigger reads the acquired scan with
it, so its false positives there become false alarms.

Fix: train ONE segmenter on all three domains -- clean HR, the degraded
acquisition, and the super-resolved output -- with the same masks. One reader,
no distribution shift, usable on both sides of the enhancement.

Then re-run the trigger and compare precision/recall against the 11.3% / 49.6%
the HR-only segmenter produced.
"""
import sys, os, json, time
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "safety", "assets")
CACHE = os.path.join(ASSETS, "data", "et_full.npz")
CKPT = os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt")
sys.path.insert(0, ROOT)

import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import seg_unet, sr_unet
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import to_mask_np, dice

DEV = "cuda"
OP = 0.5
CK = CKPT
ck = torch.load(CK, map_location=DEV, weights_only=False)
meta = ck["meta"]; FACTOR = int(meta["factor"]); SIG = float(meta["sigma"])

sr_d = sr_unet(base=32, dropout=0.2).to(DEV); sr_d.load_state_dict(ck["sr_distortion"]); sr_d.eval()
for p in sr_d.parameters():
    p.requires_grad_(False)
old_seg = seg_unet(base=32).to(DEV); old_seg.load_state_dict(ck["seg"]); old_seg.eval()

train_ds = make_dataset("cached", path=CACHE, split="train")
val_ds = make_dataset("cached", path=CACHE, split="val")
print(f"train {len(train_ds)}  val {len(val_ds)}", flush=True)


def soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.dim()))
    inter = (p * target).sum(dim=dims)
    d = (2 * inter + eps) / (p.sum(dim=dims) + target.sum(dim=dims) + eps)
    return 1.0 - d.mean()


def degrade_batch(hr, seed):
    out = torch.empty_like(hr)
    for i in range(hr.shape[0]):
        out[i, 0] = torch.from_numpy(
            degrade(hr[i, 0].cpu().numpy(), factor=FACTOR, sigma=SIG,
                    rng=np.random.default_rng(seed + i)))
    return out.to(hr.device)


torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seg = seg_unet(base=32).to(DEV)
opt = torch.optim.Adam(seg.parameters(), lr=1e-3)
EPOCHS = 12
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loader = DataLoader(train_ds, batch_size=16, shuffle=True)

t0 = time.time(); step = 0
for ep in range(EPOCHS):
    seg.train(); tot = 0.0
    for batch in loader:
        hr = batch["hr"].to(DEV); mask = batch["mask"].to(DEV)
        lr_in = degrade_batch(hr, 90000 + step * 97); step += 1
        with torch.no_grad():
            sr_im = sr_d(lr_in)
        # one batch containing all three domains, same masks
        x = torch.cat([hr, lr_in, sr_im], 0)
        y = torch.cat([mask, mask, mask], 0)
        logits = seg(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + soft_dice(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    sched.step()
    print(f"  [robust-seg] epoch {ep+1}/{EPOCHS} loss {tot/len(loader):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

seg.eval()
torch.save({"seg": seg.state_dict(), "meta": meta}, os.path.join(ROOT, "safety", "robust_seg.pt"))

# ---- domain check
d = {"HR": [], "LR": [], "SR": []}
o = {"HR": [], "LR": [], "SR": []}
for i in range(600):
    s = val_ds[i]; hr = s["hr"][0].numpy().astype(np.float32); gt = s["mask"][0].numpy() > 0.5
    if not gt.any():
        continue
    lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=np.random.default_rng(500+i))
    thr = torch.from_numpy(hr)[None, None].float().to(DEV)
    tlr = torch.from_numpy(lr)[None, None].float().to(DEV)
    with torch.no_grad():
        tsr = sr_d(tlr)
        for k, t in (("HR", thr), ("LR", tlr), ("SR", tsr)):
            d[k].append(dice(to_mask_np(seg(t)) > 0.5, gt))
            o[k].append(dice(to_mask_np(old_seg(t)) > 0.5, gt))
print("\n=== Dice by domain (val) ===")
print(f"{'domain':<8}{'HR-only seg':>14}{'robust seg':>13}")
for k in ("HR", "LR", "SR"):
    print(f"{k:<8}{np.mean(o[k]):>14.3f}{np.mean(d[k]):>13.3f}")

# ---- re-run the trigger with the robust segmenter on BOTH sides
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


def trigger_stats(reader, tag):
    flags = tp = real = 0
    for i in range(len(val_ds)):
        s = val_ds[i]; hr = s["hr"][0].numpy().astype(np.float32)
        gt = s["mask"][0].numpy() > 0.5
        if not gt.any():
            continue
        lr = degrade(hr, factor=FACTOR, sigma=SIG, rng=np.random.default_rng(7000+i))
        tlr = torch.from_numpy(lr)[None, None].float().to(DEV)
        with torch.no_grad():
            sr_im = sr_d(tlr)
            p_lr = to_mask_np(reader(tlr)) > 0.5
            p_sr = to_mask_np(reader(sr_im)) > 0.5
        for region in components(p_lr):
            if region.sum() < 4:
                continue
            if (region & p_sr).sum()/max(1, region.sum()) >= 0.1:
                continue
            flags += 1
            if (region & gt).sum()/max(1, region.sum()) >= 0.1:
                tp += 1
        for m in components(gt):
            if (m & p_lr).sum()/max(1, m.sum()) >= 0.1 and (m & p_sr).sum()/max(1, m.sum()) < 0.1:
                real += 1
    prec = tp/max(1, flags); rec = tp/max(1, real)
    print(f"  {tag:<14} flags {flags:>5}  precision {100*prec:>5.1f}%  "
          f"recall {100*rec:>5.1f}%  (real vanishings {real})", flush=True)
    return {"flags": flags, "precision": prec, "recall": rec, "real": real}


print("\n=== erasure trigger ===")
res_old = trigger_stats(old_seg, "HR-only seg")
res_new = trigger_stats(seg, "robust seg")
json.dump({"old": res_old, "new": res_new,
           "dice_old": {k: float(np.mean(o[k])) for k in o},
           "dice_new": {k: float(np.mean(d[k])) for k in d}},
          open(os.path.join(ROOT, "safety", "results", "robust_trigger.json"), "w"), indent=2)
print("\nwrote /marimo/robust_trigger.json")
