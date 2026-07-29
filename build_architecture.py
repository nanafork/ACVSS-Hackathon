"""Build ``architecture.html``: the pipeline as an academic process figure.

Every picture on the page is real output from this repository. One held-out
BraTS slice is pushed through the actual trained models, and each stage of the
pipeline is saved as its own panel: the true scan, the simulated low-field scan,
both reconstructions, what the frozen detector finds in each of them, the MC
dropout uncertainty and the error map. The 3D renders are the ones already in
the repo root. Nothing here is an illustration or a stock diagram.

    python build_architecture.py                 # -> architecture.html
    python build_architecture.py --pool 48       # search harder for a slice

Why panels are re-rendered rather than cropped out of ``demo.html``: cropping a
composite figure means guessing at pixel gutters, and the guess breaks the next
time a title wraps or matplotlib changes its padding. Rendering each panel from
the same arrays ``src/figures.py`` uses gives the identical picture with exact
bounds, and it does not require ``demo.html`` to exist.

The second half of the page is the code structure, read at build time from the
modules themselves (their docstrings and their line counts), so it cannot drift
away from the repository the way a hand-written diagram does.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import os

import numpy as np
import torch

from demo import _ensure_models, _infer, _slice_stats
from main_demo import _pick_slices
from src.metrics import to_mask_np
from src.palette import (FIG, LIGHT, UNCERTAINTY_RAMP, error_cmap,
                         uncertainty_cmap)

OUT = "architecture.html"

# The 3D renders already in the repo root, and what each one is.
RENDERS = [
    ("brain3d_true.png", "Ground truth", "true lesions, from the untouched scan"),
    ("brain3d_tumor_aware.png", "Tumor-aware (ours)", "restacked predicted masks"),
    ("brain3d_distortion.png", "Distortion-optimal (baseline)", "restacked predicted masks"),
    ("brain3d_uncertainty.png", "MC dropout uncertainty", "where the model is least sure"),
]

# The five ways into this repository, in the order you would use them.
ENTRY_POINTS = [
    ("scripts/prepare_msd.py",
     "Build the slice cache from the raw BraTS volumes: pick the region "
     "(enhancing tumor or whole tumor), centre a window on the lesion, crop and "
     "cache as an npz. Splits are by patient and asserted, never by slice."),
    ("scripts/train_demo.py",
     "Train the three networks: one segmenter, then the two SR U-Nets that differ "
     "only in their loss. Writes a single checkpoint the demos load. "
     "<code>--seg-from</code> freezes one segmenter across configurations so a "
     "loss comparison is measured with one ruler."),
    ("smoke_test.py",
     "End-to-end correctness check on synthetic data, CPU, no download. Trains "
     "everything at tiny scale and prints the metric table. It proves the "
     "pipeline runs; it is not a scientific result."),
    ("demo.py",
     "The 2D inference helpers every other page reuses, plus a static page and an "
     "optional Gradio app. This is where a slice is degraded, super-resolved, "
     "segmented and scored."),
    ("main_demo.py",
     "The deck we present: ten slides plus backups, self-contained, every image "
     "embedded. Reuses the helpers in <code>demo.py</code> and the renders from "
     "<code>render_3d.py</code> rather than duplicating them."),
]


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _panel(img, cmap="gray", vmin=0.0, vmax=1.0, overlay=None, contour=None,
           inches=2.3, dpi=150) -> str:
    """One pipeline stage as a bare square panel, base64 PNG, no axes.

    ``overlay`` is an RGBA array drawn on top (a predicted mask in a model's own
    hue); ``contour`` is a binary mask drawn as an outline (the true tumor).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(inches, inches), facecolor="white")
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if overlay is not None:
        ax.imshow(overlay)
    if contour is not None:
        ax.contour(contour, levels=[0.5], colors=[FIG["true"]], linewidths=1.4)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _overlay_frame(path: str = "brain3d_rotate.gif", frame: int = 4) -> str:
    """One still frame of the rotating overlay, as a base64 PNG.

    The overlay of all three reconstructions on one brain only exists as the
    rotating GIF. A single frame carries the same information as a plate without
    embedding four megabytes of animation that loops behind the text.
    """
    if not os.path.exists(path):
        return ""
    from PIL import Image

    im = Image.open(path)
    im.seek(min(frame, getattr(im, "n_frames", 1) - 1))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mask_overlay(pred: np.ndarray, hue: str) -> np.ndarray:
    """Solid fill in one model's hue wherever it predicted tumor."""
    from matplotlib.colors import to_rgba

    fill = np.zeros((*pred.shape, 4))
    fill[...] = to_rgba(hue, alpha=0.62)
    fill[..., 3] *= (pred >= 0.5)
    return fill


def stage_panels(device: str, pool: int) -> tuple[dict, dict, dict]:
    """Run one real held-out slice through the real models, panel by panel.

    Returns the panels as base64 PNGs, the per-version metrics for that slice,
    and the degradation settings the models were trained with.
    """
    seg, sr_d, sr_t, size, factor, sigma = _ensure_models(device)
    ds, picks = _pick_slices(seg, sr_d, sr_t, size, factor, sigma, device,
                             n_slices=1, pool=pool)
    i = picks[0]
    s = ds[i]
    hr = s["hr"][None].to(device)
    gt = s["mask"][0].cpu().numpy()
    versions, lr, mean, unc = _infer(seg, sr_d, sr_t, hr, factor, sigma, device,
                                     seed=100 + i)
    rows = _slice_stats(seg, versions, gt, hr)

    def arr(t):
        return t.detach().squeeze().cpu().numpy()

    def pred_of(t):
        with torch.no_grad():
            return to_mask_np(seg(t if t.dim() == 4 else t[None]))

    u = arr(unc)
    err = np.abs(arr(mean) - arr(hr))
    hi = lambda a: max(float(np.percentile(a, 99.0)), 1e-6)

    panels = {
        "hr": _panel(arr(hr)),
        "lr": _panel(arr(lr)),
        "sr_d": _panel(arr(versions["distortion"])),
        "sr_t": _panel(arr(versions["tumor-aware"])),
        "seg_d": _panel(arr(versions["distortion"]),
                        overlay=_mask_overlay(pred_of(versions["distortion"]),
                                              FIG["distortion"]), contour=gt),
        "seg_t": _panel(arr(versions["tumor-aware"]),
                        overlay=_mask_overlay(pred_of(versions["tumor-aware"]),
                                              FIG["tumor_aware"]), contour=gt),
        "unc": _panel(u, cmap=uncertainty_cmap(on_dark=False), vmax=hi(u)),
        "err": _panel(err, cmap=error_cmap(), vmax=hi(err)),
    }
    meta = {"size": size, "factor": factor, "sigma": sigma,
            "case": i, "lesions": rows["low-res"]["lesions"]}
    return panels, rows, meta


def code_structure() -> list[tuple[str, str, int]]:
    """Module name, first line of its docstring, and its length, read live."""
    out = []
    for name in sorted(os.listdir("src")):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        path = os.path.join("src", name)
        src = open(path).read()
        doc = ast.get_docstring(ast.parse(src)) or ""
        first = doc.strip().split("\n")[0] if doc else "(no docstring)"
        out.append((name, first, len(src.splitlines())))
    return out


# --------------------------------------------------------------------------- #
# page                                                                        #
# --------------------------------------------------------------------------- #

CSS = """
  /* Tokens. Neutrals carry a slight cool bias toward the blue the palette uses
     for ground truth, so the page and the figures read as one system. The figure
     canvas is deliberately exempt: it stays black ink on white paper in both
     themes, the way a printed figure does. */
  :root{
    --paper:#F2F3F5; --card:#FFFFFF; --rule:#D7D9DE; --rule-soft:#E4E6EA;
    --ink:#14161A; --ink-mid:#3E434A; --ink-soft:#5C6169; --chip:#E5E7EB;
    --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
    --serif:Georgia,'Times New Roman',Times,serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{--paper:#14161A; --card:#1C1F24; --rule:#2C3036; --rule-soft:#262A2F;
      --ink:#ECEEF1; --ink-mid:#B7BCC3; --ink-soft:#9AA0A8; --chip:#24282D}
  }
  :root[data-theme="dark"]{--paper:#14161A; --card:#1C1F24; --rule:#2C3036;
    --rule-soft:#262A2F; --ink:#ECEEF1; --ink-mid:#B7BCC3; --ink-soft:#9AA0A8;
    --chip:#24282D}
  :root[data-theme="light"]{--paper:#F2F3F5; --card:#FFFFFF; --rule:#D7D9DE;
    --rule-soft:#E4E6EA; --ink:#14161A; --ink-mid:#3E434A; --ink-soft:#5C6169;
    --chip:#E5E7EB}

  *{box-sizing:border-box}
  body{margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
    line-height:1.55; -webkit-font-smoothing:antialiased}
  .wrap{max-width:1180px; margin:0 auto; padding:2.4rem 1.4rem 4rem}
  h1{font-size:clamp(1.7rem,3.6vw,2.5rem); letter-spacing:-.02em; margin:.2rem 0 .3rem;
    text-wrap:balance}
  h2{font-size:1.15rem; margin:2.6rem 0 .5rem; letter-spacing:-.01em; text-wrap:balance}
  .kicker{font-family:var(--mono); font-size:.7rem; letter-spacing:.22em;
    text-transform:uppercase; color:var(--ink-soft)}
  .lede{color:var(--ink-mid); max-width:74ch}
  .lede b{color:var(--ink)}
  .small{font-size:.82rem; color:var(--ink-soft); max-width:80ch}
  code{font-family:var(--mono); font-size:.86em; background:var(--chip);
    padding:1px 5px; border-radius:4px}

  /* ---- the figure canvas: a printed academic figure, black ink on white ---- */
  .figure{background:#fff; border:1px solid var(--rule); border-radius:6px;
    padding:1.8rem 1.4rem 1.2rem; margin-top:1.2rem; overflow-x:auto;
    color:#14161A; position:relative; left:50%; transform:translateX(-50%);
    width:min(1420px, calc(100vw - 2.6rem))}
  .fig-inner{min-width:1300px}
  .sweep{display:block; position:relative; z-index:2; margin-top:64px}

  /* ---- exploded stack, landscape ----------------------------------------
     Flexbox throughout: the stack is a row of stages, each stage a column of
     [plate | 2D panels | numbered legend]. The dotted horizontals are the
     alignment guides an exploded axonometric uses to say "these are the same
     object, pulled apart", turned on their side with the layout. */
  .stack{display:flex; flex-direction:row; align-items:flex-start;
    justify-content:center; position:relative}
  .level{flex:0 0 280px; width:280px; display:flex; flex-direction:column}
  .plate{height:216px; display:flex; align-items:center; justify-content:center;
    gap:.6rem; position:relative}
  .pcol{flex:0 0 auto; max-width:176px}
  .pcol.small{max-width:128px}
  /* the guides: two hairlines running the width of the stack, behind the plates */
  .stack::before,.stack::after{content:""; position:absolute; left:4px; right:4px;
    height:0; border-top:1px dotted #9A9EA4; z-index:0}
  .stack::before{top:6px}
  .stack::after{top:210px}

  .block{border:2.2px solid #000; background:#0E1116; padding:4px; position:relative;
    z-index:1}
  .block img{display:block; width:168px; height:168px; object-fit:cover}
  .block.pair-item img{width:120px; height:120px}
  .plate-cap{font-family:var(--serif); font-weight:700; font-size:.72rem;
    text-align:center; margin-top:.3rem; line-height:1.2; color:#14161A}

  /* the 2D panels this stage produced, in a row under its plate */
  .thumbs{display:flex; flex-direction:row; justify-content:center; gap:.5rem;
    margin-top:.5rem; position:relative; z-index:1}
  .thumb{margin:0; border:1.6px solid #000; background:#fff; padding:2px}
  .thumb img{display:block; width:84px; height:84px; object-fit:cover}
  .thumb figcaption{font-family:var(--sans); font-size:.58rem; color:#4A4F56;
    text-align:center; margin:2px 0 0; padding:0; border:0; max-width:none;
    line-height:1.2; word-break:normal}

  /* the legend, under the stage it describes */
  .legend{margin-top:1rem}
  .bar{display:flex; align-items:baseline; gap:.5rem; padding:.28rem .7rem; color:#fff}
  .bar .no{font-family:var(--sans); font-weight:700; font-size:1.55rem; line-height:1.2}
  .bar .ttl{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em;
    text-transform:uppercase}
  .items{list-style:none; margin:.45rem 0 0; padding:0}
  .items li{display:flex; align-items:baseline; gap:.5rem; padding:.2rem 0 .2rem .7rem;
    font-family:var(--mono); font-size:.62rem; letter-spacing:.04em; line-height:1.35;
    text-transform:uppercase; color:#2A2E33}
  .items li span{flex:1 1 auto}
  .items li i{flex:0 0 auto; width:7px; height:7px; border-radius:50%; display:block;
    transform:translateY(-1px)}
  .items li em{font-style:normal; text-transform:none; letter-spacing:0;
    font-size:.62rem; color:#6A6E73}
  .items li .path{text-transform:none; letter-spacing:0; color:#3E434A}

  /* the sweep arrows sit in the plate column, between levels */
  .sweep{display:block; position:relative; z-index:2}

  figcaption{font-family:var(--serif); font-size:.8rem; color:#3E434A;
    margin-top:1.4rem; padding-top:.9rem; border-top:1px solid #E4E5E2;
    max-width:100ch}

  /* ---- code structure ---- */
  .scroll{overflow-x:auto}
  table{border-collapse:collapse; width:100%; min-width:520px; margin-top:.9rem;
    font-size:.86rem}
  th{text-align:left; font-family:var(--mono); font-weight:500; font-size:.66rem;
    letter-spacing:.16em; text-transform:uppercase; color:var(--ink-soft);
    border-bottom:1px solid var(--rule); padding:.4rem .6rem}
  td{border-bottom:1px solid var(--rule-soft); padding:.45rem .6rem;
    color:var(--ink-mid); vertical-align:top}
  td:first-child{white-space:nowrap; color:var(--ink)}
  td.n,th.n{text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums;
    color:var(--ink-soft); white-space:nowrap}
  .entries{display:grid; gap:.7rem; margin-top:1rem}
  .entry{background:var(--card); border:1px solid var(--rule); border-radius:10px;
    padding:.85rem 1rem}
  .entry b{font-family:var(--mono); font-size:.86rem}
  .entry p{margin:.3rem 0 0; font-size:.85rem; color:var(--ink-mid)}
  a:focus-visible,button:focus-visible{outline:2px solid #2a78d6; outline-offset:2px}
  @media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""

def _block(b64: str, cap: str, small: bool = False, mime: str = "png") -> str:
    """A plate in the exploded stack: the 3D volume this stage produced."""
    cls = "block pair-item" if small else "block"
    col = "pcol small" if small else "pcol"
    return (f'<div class="{col}"><div class="{cls}">'
            f'<img src="data:image/{mime};base64,{b64}" alt="{cap}"></div>'
            f'<div class="plate-cap">{cap}</div></div>')


def _thumb(b64: str, cap: str) -> str:
    """A 2D panel from the same stage, small, beside the plate."""
    return (f'<figure class="thumb"><img src="data:image/png;base64,{b64}" '
            f'alt="{cap}"><figcaption>{cap}</figcaption></figure>')


def _sweep(bulge: str = "up", w: int = 66, h: int = 96) -> str:
    """A hand-drawn style sweep arrow between two stages of the stack.

    A single cubic curve with a small open V head, the way an architectural
    massing diagram carries you from one stage to the next. Landscape, so it
    travels left to right; the head is computed from the curve's own end tangent
    rather than hand-placed, and the bulge alternates so four of them do not read
    as a printed repeat.
    """
    import math

    # Both variants leave and arrive travelling rightward: the bulge alternates
    # above and below the line, but the last control point stays close in y to
    # the tip, so the head never ends up pointing back up the page.
    y_mid = h / 2
    d = 28 if bulge == "up" else -28
    x0, y0 = 6, y_mid + d / 7
    x1, y1 = 22, y_mid - d
    x2, y2 = w - 30, y_mid - d / 5
    x3, y3 = w - 10, y_mid + d / 14

    # end tangent of a cubic is 3 * (P3 - P2)
    tx, ty = x3 - x2, y3 - y2
    n = math.hypot(tx, ty) or 1.0
    tx, ty = tx / n, ty / n
    head = 10.0
    barbs = []
    for deg in (28, -28):
        a = math.radians(deg)
        # rotate the reversed tangent, so both barbs open behind the tip
        bx = (-tx) * math.cos(a) - (-ty) * math.sin(a)
        by = (-tx) * math.sin(a) + (-ty) * math.cos(a)
        barbs.append(f'M{x3:.1f},{y3:.1f} L{x3 + bx * head:.1f},{y3 + by * head:.1f}')

    return (f'<svg class="sweep" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'aria-hidden="true"><g fill="none" stroke="#14161A" stroke-width="1.5" '
            f'stroke-linecap="round">'
            f'<path d="M{x0},{y0:.1f} C{x1},{y1:.1f} {x2},{y2:.1f} {x3},{y3:.1f}"/>'
            f'<path d="{" ".join(barbs)}"/></g></svg>')


def _level(no: int, title: str, hue: str, blocks: str, thumbs: str,
           items: list[tuple[str, str, str]]) -> str:
    """One stage of the stack: plate, then its 2D panels, then a numbered legend.

    ``items`` are (label, note, dot hue). The dot hue is the model or field the
    line refers to, so the legend, the 2D panels and the 3D renders all label the
    same thing with the same colour.
    """
    def _label(text: str) -> str:
        # Paths keep their own case: SRC/DEGRADE.PY is not a file that exists.
        return f'<span class="path">{text}</span>' if ".py" in text else text

    lis = "".join(
        f'<li><span>{_label(label)}{f" <em>{note}</em>" if note else ""}</span>'
        f'<i style="background:{dot}"></i></li>'
        for label, note, dot in items)
    return f"""<div class="level">
      <div class="plate">{blocks}</div>
      <div class="thumbs">{thumbs}</div>
      <div class="legend">
        <div class="bar" style="background:{hue}">
          <span class="no">{no}</span><span class="ttl">{title}</span></div>
        <ul class="items">{lis}</ul>
      </div>
    </div>"""


def build(out: str = OUT, device: str | None = None, pool: int = 24) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p, rows, meta = stage_panels(device, pool)

    missing = [f for f, _, _ in RENDERS if not os.path.exists(f)]

    C_TRUE, C_BASE, C_OURS = LIGHT["true"], LIGHT["distortion"], LIGHT["tumor_aware"]
    C_UNC = UNCERTAINTY_RAMP[-2]
    GREY = "#6A6E73"

    def render(name: str) -> str:
        return _b64_file(name) if os.path.exists(name) else ""

    # Landscape, so the stages read left to right and the numbering counts up
    # with the flow rather than down a stack of floors.
    lv1 = _level(
        1, "Input and degradation", C_TRUE,
        _block(render("brain3d_true.png"), "The scan we start from"),
        _thumb(p["hr"], "true scan") + _thumb(p["lr"], "degraded"),
        [("Real BraTS, held-out patient", "no model has seen it", C_TRUE),
         (f"k-space truncation &times;{meta['factor']}", "resolution loss", GREY),
         ("Rician noise", f"&sigma;={meta['sigma']}, the noise magnitude MRI follows", GREY),
         ("Degraded scan is the only input", "an exact reference is kept", GREY),
         ("src/degrade.py", "", GREY)])

    lv2 = _level(
        2, "Reconstruction", C_BASE,
        (_block(render("brain3d_distortion.png"), "Distortion-optimal", small=True)
         + _block(render("brain3d_tumor_aware.png"), "Tumor-aware", small=True)),
        _thumb(p["sr_d"], "baseline") + _thumb(p["sr_t"], "ours"),
        [("Two SR U-Nets", "identical architecture, data, schedule", GREY),
         ("Distortion-optimal", "pixel error only", C_BASE),
         ("Tumor-aware", "lesion-weighted, ours", C_OURS),
         ("No GAN, no pretrained weights", "realism rewards invention", GREY),
         ("src/models.py, src/losses.py", "", GREY)])

    lv3 = _level(
        3, "Detection and safety", C_OURS,
        _block(_overlay_frame() or render("brain3d_true.png"), "All three, overlaid"),
        _thumb(p["seg_d"], "on baseline") + _thumb(p["seg_t"], "on ours"),
        [("One frozen segmenter", "reads every image", GREY),
         ("Only the image changes", "so the difference is the reconstruction", GREY),
         ("Fill = found, outline = true tumor", "a bare outline is an erasure", C_TRUE),
         ("Lesions erased, lesions fabricated", "by lesion size", C_BASE),
         ("src/metrics.py", "", GREY)])

    lv4 = _level(
        4, "Uncertainty", C_UNC,
        _block(render("brain3d_uncertainty.png"), "Where the model is unsure"),
        _thumb(p["unc"], "uncertainty") + _thumb(p["err"], "abs error"),
        [("Monte Carlo dropout", "10 stochastic passes", C_UNC),
         ("Variance across passes", "the model's own doubt", C_UNC),
         ("Is the doubt where the error is?", "uncertainty vs error AUROC", GREY),
         ("Restacked per-slice masks", "marching cubes", GREY),
         ("src/uncertainty.py, render_3d.py", "", GREY)])

    sw_a, sw_b, sw_c = (_sweep("up"), _sweep("down"), _sweep("up"))

    order = ["low-res", "distortion", "tumor-aware"]
    label = {"low-res": "low-res input", "distortion": "distortion-optimal (baseline)",
             "tumor-aware": "tumor-aware (ours)"}
    slice_rows = "".join(
        f"<tr><td>{label[k]}</td><td class=n>{rows[k]['psnr']:.1f}</td>"
        f"<td class=n>{rows[k]['ssim']:.3f}</td><td class=n>{rows[k]['dice']:.3f}</td>"
        f"<td class=n>{rows[k]['erased']}/{rows[k]['lesions']}</td>"
        f"<td class=n>{rows[k]['fabricated']}</td></tr>"
        for k in order if k in rows)

    modules = "".join(f"<tr><td><code>src/{n}</code></td><td>{d}</td>"
                      f"<td class=n>{loc}</td></tr>" for n, d, loc in code_structure())

    entries = "".join(f'<div class="entry"><b>{path}</b><p>{what}</p></div>'
                      for path, what in ENTRY_POINTS)

    note = ""
    if missing:
        note = ('<p class="small"><b>Missing 3D renders:</b> '
                + ", ".join(f"<code>{m}</code>" for m in missing)
                + ". Run <code>python main_demo.py</code> to regenerate them.</p>")

    html = f"""<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Architecture &middot; Tumor-Aware MRI Super-Resolution</title>
<style>{CSS}</style>
<div class="wrap">
  <div class="kicker">Tumor-aware MRI super-resolution</div>
  <h1>Architecture</h1>
  <p class="lede">Every panel below is this pipeline's own output on <b>one held-out
  BraTS slice</b> that no model was trained on, pushed through the real trained
  networks at build time. The models saw {meta['size']}&times;{meta['size']} slices
  degraded by k-space truncation &times;{meta['factor']} with Rician
  &sigma;={meta['sigma']}. Nothing here is an illustration.</p>

  <h2>The pipeline</h2>
  <div class="figure"><div class="fig-inner">
    <div class="stack">{lv1}{sw_a}{lv2}{sw_b}{lv3}{sw_c}{lv4}</div>
    <figcaption><b>Figure 1.</b> The tumor-aware super-resolution pipeline, drawn
    as an exploded stack: one patient, pulled apart left to right into the four
    stages that act on them. Every plate is a 3D volume this pipeline produced, and the small
    panels beside each plate are the 2D output of that same stage on one held-out
    slice. The sweeping arrows carry the direction of travel; the dotted verticals
    are alignment guides rather than flow, because the levels are one brain pulled
    apart, not four different ones. The degradation is applied on the
    fly, so every low-resolution input keeps an exact high-resolution reference and
    no paired low-field acquisition is required. The segmentation network at stage 3
    is frozen and shared, which is what makes the two reconstructions at stage 2
    comparable: the only thing that changes is the image handed to it.</figcaption>
  </div></div>
  {note}

  <h2>What that slice scored</h2>
  <p class="small">The same numbers the deck reports, for the single slice drawn
  above. It carries {meta['lesions']} lesion components. PSNR and SSIM compare the
  image to the true scan; Dice, erased and fabricated compare the frozen detector's
  output on it to the true tumor mask. One slice is an illustration, not a result:
  the aggregate figures come from 70 held-out patients.</p>
  <div class="scroll"><table>
    <tr><th>image handed to the detector</th><th class=n>PSNR</th><th class=n>SSIM</th>
      <th class=n>Dice</th><th class=n>erased</th><th class=n>fabricated</th></tr>
    {slice_rows}
  </table></div>

  <h2>Code structure</h2>
  <p class="small">Read from the modules themselves at build time: each line is the
  first line of that module's own docstring, with its length in lines.</p>
  <div class="scroll"><table>
    <tr><th>module</th><th>what it is</th><th class=n>lines</th></tr>
    {modules}
  </table></div>

  <h2>Five entry points</h2>
  <p class="small">Data, training, verification, inference, presentation. Everything
  else in the repository is a library these five call.</p>
  <div class="entries">{entries}</div>
</div>"""

    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)
    if missing:
        print("note: missing renders", missing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pool", type=int, default=24,
                    help="how many held-out slices to score when picking one")
    a = ap.parse_args()
    build(a.out, a.device, a.pool)


if __name__ == "__main__":
    main()
