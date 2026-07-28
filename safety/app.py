"""Live safety-layer demo.

    marimo run safety/app.py      # booth mode
    marimo edit safety/app.py     # to tweak

Runs real inference: the acquired scan is degraded from a held-out BraTS slice,
both enhancement models reconstruct it, and one segmenter reads all of them. The
safety layer compares the segmentation before and after enhancement. It never
sees the ground-truth mask.
"""
import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import base64
    import io
    import os
    import sys

    import marimo as mo
    import numpy as np
    import torch

    ROOT = os.environ.get(
        "TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, ROOT)
    ASSETS = os.path.join(ROOT, "safety", "assets")

    from src.data import make_dataset
    from src.degrade import degrade
    from src.metrics import psnr, ssim, to_mask_np
    from src.models import seg_unet, sr_unet

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    _ck = torch.load(os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt"),
                     map_location=DEV, weights_only=False)
    SIGMA = float(_ck["meta"]["sigma"])

    sr_base = sr_unet(base=32, dropout=0.2).to(DEV)
    sr_base.load_state_dict(_ck["sr_distortion"]); sr_base.eval()
    sr_ours = sr_unet(base=32, dropout=0.2).to(DEV)
    sr_ours.load_state_dict(_ck["sr_tumor_aware"]); sr_ours.eval()

    # The reader. Prefer the domain-robust segmenter when it has been trained
    # (scripts/train_robust_segmenter.py); fall back to the shipped HR-only one.
    reader = seg_unet(base=32).to(DEV)
    _rp = os.path.join(ROOT, "safety", "robust_seg.pt")
    if os.path.exists(_rp):
        reader.load_state_dict(torch.load(_rp, map_location=DEV,
                                          weights_only=False)["seg"])
        READER = "domain-robust"
    else:
        reader.load_state_dict(_ck["seg"])
        READER = "HR-only (run train_robust_segmenter.py for the better one)"
    reader.eval()

    VAL = make_dataset("cached", path=os.path.join(ASSETS, "data", "et_full.npz"),
                       split="val")
    return (ASSETS, DEV, READER, SIGMA, VAL, base64, degrade, io, mo, np,
            psnr, reader, sr_base, sr_ours, ssim, to_mask_np, torch)


@app.cell(hide_code=True)
def _(DEV, SIGMA, VAL, base64, degrade, io, np, psnr, reader, ssim,
      to_mask_np, torch):
    def cc(mask):
        h, w = mask.shape
        lab = np.zeros((h, w), np.int32); out = []; n = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and lab[i, j] == 0:
                    n += 1; st = [(i, j)]; lab[i, j] = n; px = []
                    while st:
                        y, x = st.pop(); px.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = n; st.append((ny, nx))
                    m = np.zeros((h, w), bool)
                    for y, x in px:
                        m[y, x] = True
                    out.append(m)
        return out

    def edge(m):
        e = np.zeros_like(m)
        e[1:, :] |= m[1:, :] & ~m[:-1, :]; e[:-1, :] |= m[:-1, :] & ~m[1:, :]
        e[:, 1:] |= m[:, 1:] & ~m[:, :-1]; e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
        return e

    def grow(m, k=1):
        o = m.copy()
        for _ in range(k):
            d = o.copy()
            d[1:, :] |= o[:-1, :]; d[:-1, :] |= o[1:, :]
            d[:, 1:] |= o[:, :-1]; d[:, :-1] |= o[:, 1:]
            o = d
        return o

    def png(img, overlays=(), scale=5):
        import imageio.v3 as iio
        g = np.clip(img, 0, 1)
        rgb = np.stack([g, g, g], -1)
        for mask, colour, solid in overlays:
            if mask is None or not mask.any():
                continue
            c = np.array(colour, float) / 255.0
            if solid:
                rgb[mask] = 0.72 * rgb[mask] + 0.28 * c
            else:
                rgb[grow(edge(mask), 1)] = c
        arr = (np.clip(np.repeat(np.repeat(rgb, scale, 0), scale, 1), 0, 1) * 255)
        buf = io.BytesIO(); iio.imwrite(buf, arr.astype(np.uint8), extension=".png")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def run_case(idx, factor, model):
        s = VAL[int(idx)]
        hr = s["hr"][0].numpy().astype(np.float32)
        gt = s["mask"][0].numpy() > 0.5
        lr = degrade(hr, factor=int(factor), sigma=SIGMA,
                     rng=np.random.default_rng(7))
        t = torch.from_numpy(lr)[None, None].float().to(DEV)
        with torch.no_grad():
            out = model(t)
            p_lr = to_mask_np(reader(t)) > 0.5
            p_sr = to_mask_np(reader(out)) > 0.5
        vanished = np.zeros_like(p_lr)
        for reg in cc(p_lr):
            if reg.sum() >= 4 and (reg & p_sr).sum() / max(1, reg.sum()) < 0.1:
                vanished |= reg
        h = torch.from_numpy(hr)[None, None].to(DEV)
        return dict(hr=hr, gt=gt, lr=lr, sr=out[0, 0].cpu().numpy(),
                    p_lr=p_lr, p_sr=p_sr, vanished=vanished,
                    psnr=float(psnr(out, h)), ssim=float(ssim(out, h)),
                    n_lr=len(cc(p_lr)), n_sr=len(cc(p_sr)), n_gone=len(cc(vanished)))
    return cc, png, run_case


@app.cell(hide_code=True)
def _(VAL, mo):
    case_pick = mo.ui.slider(0, len(VAL) - 1, value=843, label="held-out BraTS slice")
    factor_pick = mo.ui.slider(2, 8, value=4, step=1, label="acquisition degradation (x)")
    safety_on = mo.ui.switch(True, label="SAFETY LAYER")
    mo.hstack([case_pick, factor_pick, safety_on], justify="start", gap=2.5)
    return case_pick, factor_pick, safety_on


@app.cell(hide_code=True)
def _(READER, case_pick, factor_pick, mo, png, run_case, safety_on,
      sr_base, sr_ours):
    rb = run_case(case_pick.value, factor_pick.value, sr_base)
    ro = run_case(case_pick.value, factor_pick.value, sr_ours)

    BLUE, RED, CYAN = (96, 165, 250), (255, 90, 70), (92, 217, 253)
    acq = png(rb["lr"], [(rb["p_lr"], BLUE, True)])
    std = png(rb["sr"], [(rb["p_sr"], BLUE, True)] +
              ([(rb["vanished"], RED, False)] if safety_on.value else []))
    our = png(ro["sr"], [(ro["p_sr"], BLUE, True)] +
              ([(ro["vanished"], RED, False)] if safety_on.value else []))
    tru = png(rb["hr"], [(rb["gt"], CYAN, False)])

    def card(title, sub, img, tone="#8fa0b3"):
        return mo.vstack([
            mo.md(f"<div style='font-size:15px;font-weight:700;color:{tone}'>{title}</div>"
                  f"<div style='font-size:12px;color:#78889c;margin-bottom:6px'>{sub}</div>"),
            mo.image(img)], gap=0.2)

    def verdict(r, name):
        n = r["n_gone"] if safety_on.value else 0
        if n:
            return (f"<div style='padding:11px 13px;border-radius:9px;font-weight:600;"
                    f"font-size:13px;background:rgba(255,90,70,.12);color:#ff6b57;"
                    f"border:1px solid rgba(255,90,70,.35)'>&#9888; {name}: {n} region(s) "
                    f"vanished during enhancement &mdash; review the original</div>")
        return ("<div style='padding:11px 13px;border-radius:9px;font-weight:600;"
                "font-size:13px;background:rgba(62,207,142,.12);color:#3ecf8e;"
                f"border:1px solid rgba(62,207,142,.32)'>&#10003; {name}: nothing vanished</div>")

    mo.vstack([
        mo.md("<div style='font-size:26px;font-weight:800;letter-spacing:-.02em'>"
              "Enhancement can make a tumor disappear</div>"
              "<div style='color:#8fa0b3;font-size:14px;margin-top:2px'>Real held-out "
              "BraTS. The layer compares the segmentation before and after enhancement "
              f"&mdash; no ground truth. Reader: {READER}.</div>"),
        mo.hstack([
            card("Acquired", "what the scanner produced", acq),
            card("Standard model", "PSNR-optimal", std, "#ff8f7a"),
            card("Ours", "tumor-aware", our, "#3ecf8e"),
            card("Ground truth", "never shown to the layer", tru, "#5cd9fd"),
        ], gap=1.2, justify="start"),
        mo.hstack([mo.Html(verdict(rb, "standard")), mo.Html(verdict(ro, "ours"))],
                  gap=1, justify="start"),
        mo.md(f"<span style='color:#8fa0b3;font-size:13px'>standard &nbsp; PSNR "
              f"<b>{rb['psnr']:.2f} dB</b> &nbsp; SSIM <b>{rb['ssim']:.3f}</b> "
              f"&nbsp;|&nbsp; ours &nbsp; PSNR <b>{ro['psnr']:.2f} dB</b> &nbsp; SSIM "
              f"<b>{ro['ssim']:.3f}</b> &nbsp;&mdash;&nbsp; regions found: acquired "
              f"<b>{rb['n_lr']}</b> &rarr; standard <b>{rb['n_sr']}</b>, ours "
              f"<b>{ro['n_sr']}</b></span>"),
        mo.md("<span style='color:#78889c;font-size:12px'>Measured on held-out patients: "
              "the layer raises ~0.15 flags per slice, of which <b>20% mark a real "
              "vanished lesion</b> &mdash; a review prompt, not a diagnosis. Enhancement "
              "destroys 3.6% of lesions and recovers 8.2%. This slice is chosen because "
              "it shows the effect clearly.</span>"),
    ])
    return


if __name__ == "__main__":
    app.run()
