"""Demo for the tumor-aware super-resolution safety pipeline.

Two modes:
  * Static (default, always works, no extra deps): generates ``demo.html``, a
    self-contained page with before/after panels, the uncertainty map, and the
    per-slice safety readout for a few test slices. Open it in any browser or
    share the file. Great for a hackathon booth or a link to judges.
  * Interactive (``--gradio``): launches a Gradio app if gradio is installed,
    with sliders to pick a slice and the degradation level.

Models are loaded from checkpoints/demo.pt if present; otherwise a quick
synthetic model is trained on the fly so the demo always runs.

    python demo.py                 # writes demo.html
    python demo.py --gradio        # interactive app (needs `pip install gradio`)
"""

from __future__ import annotations

import argparse
import base64
import io
import os

import numpy as np
import torch

from src.checkpoint import load_models
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import (dice, hallucination_stats, lesion_records, psnr,
                         ssim, to_mask_np)
from src.uncertainty import mc_predict

CKPT = "checkpoints/demo.pt"


def _ensure_models(device: str):
    """Load checkpoint, or quick-train on synthetic if none exists."""
    if os.path.exists(CKPT):
        seg, sr_d, sr_t, meta = load_models(CKPT, device=device)
        size = meta.get("size", 96)
        factor = meta.get("factor", 4)
        sigma = meta.get("sigma", 0.03)
        return seg, sr_d, sr_t, size, factor, sigma
    print("No checkpoint found; quick-training on synthetic data (one-off)...")
    from src.losses import make_sr_loss
    from src.models import seg_unet, sr_unet
    from src.train import train_segmenter, train_sr
    size, factor, sigma = 96, 4, 0.03
    ds = make_dataset("synthetic", n=160, size=size, seed=1)
    seg = seg_unet(base=32)
    train_segmenter(seg, ds, epochs=8, bs=8, device=device)
    for p in seg.parameters():
        p.requires_grad_(False)
    sr_d = sr_unet(base=32, dropout=0.2)
    sr_t = sr_unet(base=32, dropout=0.2)
    train_sr(sr_d, ds, make_sr_loss("distortion"), factor=factor, sigma=sigma,
             epochs=15, bs=8, device=device, tag="sr-distortion")
    train_sr(sr_t, ds, make_sr_loss("tumor_aware", weight=40.0), factor=factor,
             sigma=sigma, epochs=15, bs=8, device=device, tag="sr-tumor-aware")
    return seg, sr_d, sr_t, size, factor, sigma


def _infer(seg, sr_d, sr_t, hr, factor, sigma, device, seed=0):
    """Return dict of versions, per-version stats, and uncertainty for the
    tumor-aware model."""
    rng = np.random.default_rng(seed)
    lr_np = degrade(hr[0, 0].cpu().numpy(), factor=factor, sigma=sigma, rng=rng)
    lr = torch.from_numpy(lr_np)[None, None].to(device)
    with torch.no_grad():
        versions = {"low-res": lr, "distortion": sr_d(lr), "tumor-aware": sr_t(lr),
                    "true HR": hr}
    mean, unc = mc_predict(sr_t, lr, passes=15)
    return versions, lr, mean, unc


def _slice_stats(seg, versions, gt, hr):
    rows = {}
    for name, img in versions.items():
        pred = to_mask_np(seg(img)) if img.requires_grad is False else to_mask_np(seg(img))
        rec = lesion_records(gt, pred)
        erased = sum(0 if r["detected"] else 1 for r in rec)
        hal = hallucination_stats(gt, pred)
        rows[name] = {
            "psnr": psnr(img, hr), "ssim": ssim(img, hr),
            "dice": dice(pred, gt),
            "lesions": len(rec), "erased": erased,
            "fabricated": hal["fabricated_components"],
        }
    return rows


# --------------------------------------------------------------------------- #
# Static HTML demo
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_static(n_slices: int = 3, device: str = "cpu", out: str = "demo.html"):
    import matplotlib
    matplotlib.use("Agg")
    from src.figures import comparison_figure, uncertainty_figure

    seg, sr_d, sr_t, size, factor, sigma = _ensure_models(device)
    test_ds = make_dataset("synthetic", n=n_slices, size=size, seed=999)

    blocks = []
    for i in range(len(test_ds)):
        s = test_ds[i]
        hr = s["hr"][None].to(device)
        mask = s["mask"][None].to(device)
        gt = s["mask"][0].cpu().numpy()
        versions, lr, mean, unc = _infer(seg, sr_d, sr_t, hr, factor, sigma, device, seed=100 + i)
        rows = _slice_stats(seg, versions, gt, hr)

        cfig = comparison_figure(hr, mask, lr, {"distortion": versions["distortion"],
                                                "tumor-aware": versions["tumor-aware"]}, seg)
        ufig = uncertainty_figure(lr, mean, unc, hr)
        cimg, uimg = _fig_to_b64(cfig), _fig_to_b64(ufig)

        table = "".join(
            f"<tr><td>{k}</td><td>{v['psnr']:.1f}</td><td>{v['ssim']:.3f}</td>"
            f"<td>{v['dice']:.3f}</td><td>{v['erased']}/{v['lesions']}</td>"
            f"<td>{v['fabricated']}</td></tr>"
            for k, v in rows.items())
        blocks.append(f"""
        <section>
          <h3>Slice {i + 1}</h3>
          <img src="data:image/png;base64,{cimg}"/>
          <img src="data:image/png;base64,{uimg}"/>
          <table>
            <tr><th>version</th><th>PSNR</th><th>SSIM</th><th>Dice</th>
                <th>lesions erased</th><th>fabricated</th></tr>
            {table}
          </table>
        </section>""")

    html = f"""<!doctype html><meta charset=utf-8>
    <title>Tumor-Aware SR Demo</title>
    <style>
      body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
           margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
      h1{{margin-bottom:.2rem}} .sub{{color:#666}}
      img{{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:.4rem 0}}
      table{{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}}
      td,th{{border:1px solid #ddd;padding:4px 10px;text-align:center}}
      section{{border-top:1px solid #eee;padding-top:1rem;margin-top:1rem}}
      code{{background:#f4f4f2;padding:1px 5px;border-radius:4px}}
    </style>
    <h1>Tumor-Aware MRI Super-Resolution</h1>
    <p class=sub>Does sharpening a low-field scan hide the tumor? Comparing a
    distortion-optimal model against a tumor-aware one, then checking the
    downstream segmentation and an uncertainty map.
    (config: size {size}, k-space factor {factor}, Rician &sigma;={sigma})</p>
    <p><b>Read the "lesions erased" column:</b> lower is safer. The cyan outline
    in the mask row is the true tumor; the red overlay is the model's prediction.</p>
    {''.join(blocks)}
    <p class=sub>Synthetic demo data unless trained on BraTS. See <code>README.md</code>.</p>
    """
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Interactive Gradio demo
# --------------------------------------------------------------------------- #
def build_gradio(device: str = "cpu"):
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")
    from src.figures import comparison_figure, uncertainty_figure

    seg, sr_d, sr_t, size, def_factor, def_sigma = _ensure_models(device)
    test_ds = make_dataset("synthetic", n=12, size=size, seed=999)

    def run(idx, factor, sigma):
        s = test_ds[int(idx)]
        hr = s["hr"][None].to(device); mask = s["mask"][None].to(device)
        gt = s["mask"][0].cpu().numpy()
        versions, lr, mean, unc = _infer(seg, sr_d, sr_t, hr, int(factor), float(sigma), device, seed=int(idx))
        rows = _slice_stats(seg, versions, gt, hr)
        cfig = comparison_figure(hr, mask, lr, {"distortion": versions["distortion"],
                                                "tumor-aware": versions["tumor-aware"]}, seg)
        ufig = uncertainty_figure(lr, mean, unc, hr)
        md = "| version | PSNR | SSIM | Dice | erased/lesions | fabricated |\n|---|---|---|---|---|---|\n"
        md += "".join(f"| {k} | {v['psnr']:.1f} | {v['ssim']:.3f} | {v['dice']:.3f} "
                      f"| {v['erased']}/{v['lesions']} | {v['fabricated']} |\n"
                      for k, v in rows.items())
        return cfig, ufig, md

    with gr.Blocks(title="Tumor-Aware MRI Super-Resolution") as app:
        gr.Markdown("# Tumor-Aware MRI Super-Resolution\nDoes sharpening a low-field "
                    "scan hide the tumor? Distortion-optimal vs tumor-aware.")
        with gr.Row():
            idx = gr.Slider(0, len(test_ds) - 1, value=0, step=1, label="test slice")
            factor = gr.Slider(2, 8, value=def_factor, step=1, label="k-space factor (resolution loss)")
            sigma = gr.Slider(0.0, 0.1, value=def_sigma, step=0.01, label="Rician noise")
        btn = gr.Button("Run", variant="primary")
        cplot = gr.Plot(label="low-res / distortion / tumor-aware / true + masks")
        uplot = gr.Plot(label="uncertainty and error")
        table = gr.Markdown()
        btn.click(run, [idx, factor, sigma], [cplot, uplot, table])
        app.load(run, [idx, factor, sigma], [cplot, uplot, table])
    app.launch(share=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradio", action="store_true", help="launch interactive app")
    ap.add_argument("--slices", type=int, default=3, help="slices in static demo")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gradio:
        try:
            build_gradio(device)
        except ImportError:
            print("gradio not installed. Run `pip install gradio`, or use the "
                  "static demo: `python demo.py`")
    else:
        build_static(n_slices=args.slices, device=device)


if __name__ == "__main__":
    main()
