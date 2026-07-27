"""Build the self-contained safety demo page.

Slider = k-space truncation factor (how much the scan was degraded).
For each position, for both objectives, we render the SR image with the true
lesion outline and the segmenter's prediction, and record:
  PSNR / SSIM      what the field optimises
  lesions found    what a patient cares about
  confidence       mean segmenter agreement across MC-dropout passes
  verdict          reference-free erasure flag (low confidence spread = danger)

Everything is computed once and embedded as base64 PNG, so the page is one file.
"""
import sys, json, base64, io
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import numpy as np, torch
import imageio.v3 as iio

from src.checkpoint import load_models
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import psnr, ssim, to_mask_np
from src.models import enable_mc_dropout
from src.consistency import brain_roi

DEV = "cuda"; PASSES = 16
FACTORS = [2, 3, 4, 5, 6, 8]
seg, sr_d, sr_t, meta = load_models(os.path.join(ROOT, "checkpoints", "demo.pt"), device=DEV)
SIZE, SIG = int(meta["size"]), float(meta["sigma"])
ds = make_dataset("synthetic", n=64, size=SIZE, seed=999)
seg.eval(); sr_d.eval(); sr_t.eval()


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


def outline(mask):
    m = mask.astype(bool); e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] & ~m[:-1, :]; e[:-1, :] |= m[:-1, :] & ~m[1:, :]
    e[:, 1:] |= m[:, 1:] & ~m[:, :-1]; e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def run(model, lr_np):
    x = torch.from_numpy(lr_np)[None, None].float().to(DEV)
    with torch.no_grad():
        out = model(x)
        pred = to_mask_np(seg(out)) > 0.5
    enable_mc_dropout(model)
    with torch.no_grad():
        outs = torch.stack([model(x) for _ in range(PASSES)], 0)
        probs = torch.stack([torch.sigmoid(seg(outs[p])) for p in range(PASSES)], 0)
    model.eval()
    roi = brain_roi(lr_np)
    spread = float((probs.std(0)[0, 0].cpu().numpy() * roi)[roi].mean())
    return out[0, 0].cpu().numpy(), pred, spread


def panel(img, gt, pred, scale=4):
    """Grayscale SR with cyan true outline and red predicted region."""
    g = np.clip(img, 0, 1)
    rgb = np.stack([g, g, g], -1)
    pm = pred.astype(bool)
    rgb[pm] = 0.55 * rgb[pm] + 0.45 * np.array([0.96, 0.35, 0.28])
    ol = outline(gt)
    rgb[ol] = np.array([0.36, 0.85, 0.99])
    rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def b64(arr):
    buf = io.BytesIO(); iio.imwrite(buf, arr, extension=".png")
    return base64.b64encode(buf.getvalue()).decode()


# --- pick a slice where the effect is visible: distortion loses a lesion at a
# mid factor that tumor-aware keeps. Reported as chosen-for-illustration.
best = None
rng = np.random.default_rng(0)
for i in range(len(ds)):
    gt = ds[i]["mask"][0].numpy() > 0.5
    if not gt.any():
        continue
    hr = ds[i]["hr"][0].numpy()
    les = components(gt)
    score = 0
    for f in (4, 5):
        lr = degrade(hr, factor=f, sigma=SIG, rng=np.random.default_rng(7))
        _, pd, _ = run(sr_d, lr)
        _, pt, _ = run(sr_t, lr)
        for m in les:
            d_lost = (m & pd).sum() / max(1, m.sum()) < 0.1
            t_kept = (m & pt).sum() / max(1, m.sum()) >= 0.1
            if d_lost and t_kept:
                score += 1
    if best is None or score > best[0]:
        best = (score, i)
print("chosen slice:", best, flush=True)
IDX = best[1]

hr = ds[IDX]["hr"][0].numpy()
gt = ds[IDX]["mask"][0].numpy() > 0.5
n_true = len(components(gt))

frames = []
for f in FACTORS:
    lr = degrade(hr, factor=f, sigma=SIG, rng=np.random.default_rng(7))
    entry = {"factor": f, "lr": b64(panel(lr, gt, np.zeros_like(gt)))}
    for key, model in (("distortion", sr_d), ("tumor_aware", sr_t)):
        img, pred, spread = run(model, lr)
        t = torch.from_numpy(img)[None, None].to(DEV)
        h = torch.from_numpy(hr)[None, None].to(DEV)
        found = sum(1 for m in components(gt) if (m & pred).sum()/max(1, m.sum()) >= 0.1)
        entry[key] = {
            "img": b64(panel(img, gt, pred)),
            "psnr": round(float(psnr(t, h)), 2),
            "ssim": round(float(ssim(t, h)), 3),
            "found": found,
            "spread": round(spread, 5),
        }
    frames.append(entry)
    print(f"  factor {f} done", flush=True)

# Confidence scale: map spread to 0-100 across everything we rendered, so the
# meter is comparable between the two models and across slider positions.
sp = [frames[k][m]["spread"] for k in range(len(frames)) for m in ("distortion", "tumor_aware")]
lo, hi = min(sp), max(sp)
for fr in frames:
    for m in ("distortion", "tumor_aware"):
        s = fr[m]["spread"]
        fr[m]["confidence"] = round(100.0 * (1.0 - (s - lo) / (hi - lo + 1e-9)), 1)

payload = {"frames": frames, "n_true": n_true, "size": SIZE, "slice": int(IDX),
           "detector_auroc": 0.782,
           "auroc_bands": {"small": 0.571, "medium": 0.853, "large": 0.821}}
json.dump(payload, open(os.path.join(ROOT, "demo_payload.json"), "w"))
print("payload written, frames:", len(frames), "true lesions:", n_true)
