"""Build ``architecture.html``: the pipeline as an academic process figure.

Every picture on the page is real output from this repository. One held-out
BraTS slice is pushed through the actual trained models, and each stage of the
pipeline is saved as its own panel: the true scan, the simulated low-field scan,
both reconstructions, what the frozen segmentation network finds in each of
them, the MC
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
PNG_OUT = "figures/architecture.png"
TMP_HTML = ".architecture_nocaption.html"

# Where to find a headless browser for the PNG export. The deck needs the figure
# as one image, and rendering it here rather than re-implementing the layout in
# main_demo.py keeps a single source for the diagram.
CHROMES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

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


def _unet_diagram(model, hue: str, label: str, passes: int = 0,
                  frozen: bool = False, inches=(1.55, 1.55), dpi=260) -> tuple[str, str]:
    """The U-Net as it is actually built, drawn square and small.

    Channel widths, depth, the residual connection, the dropout probability and
    the parameter count are read off the model rather than typed in here, so the
    drawing cannot claim an architecture the code does not have. The numbers ride
    in the HTML caption returned alongside the image: text stays crisp at any size,
    which is what lets the diagram itself be this small.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn as nn
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyArrowPatch, Rectangle

    base = model.d1.block[0].out_channels
    chans = [base, base * 2, base * 4, base * 8]
    drop = 0.0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            drop = float(m.p)
            break
    n_par = sum(q.numel() for q in model.parameters())

    # Equal scales, so a square patch draws square. dy is set so the U spans as
    # much vertically as it does horizontally and fills the square frame.
    dx, dy = 1.0, 1.9
    nodes = [(i * dx, -min(i, 6 - i) * dy) for i in range(7)]

    fig, ax = plt.subplots(figsize=inches, facecolor="white")
    side = 0.46
    for x, y in nodes:
        ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                               linewidth=0.7, edgecolor=hue,
                               facecolor=to_rgba(hue, 0.30), zorder=2))

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=4.0, linewidth=0.7,
            color="#8A8F96" if dashed else "#14161A",
            linestyle=(0, (2.2, 1.6)) if dashed else "solid",
            shrinkA=5.0, shrinkB=5.0, zorder=1))

    for i in range(6):
        arrow(nodes[i], nodes[i + 1])
    for i in (0, 1, 2):                       # the skips that make it a U
        arrow(nodes[i], nodes[6 - i], dashed=True)

    if passes:
        ax.text(6.0, 0.42, f"$\\times${passes}", ha="center", va="bottom",
                fontsize=7.0, color=hue)

    ax.set_xlim(-0.55, 6.55)
    ax.set_ylim(-3 * dy - 0.6, 0.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0.01, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    bits = [label, " &rarr; ".join(str(c) for c in chans)]
    if getattr(model, "residual", False):
        bits.append("residual")
    bits.append(f"dropout {drop:g}" if drop else "no dropout")
    if frozen:
        bits.append("frozen")
    bits.append(f"{n_par / 1e6:.1f}M params")
    return base64.b64encode(buf.getvalue()).decode(), " &middot; ".join(bits)


def _kspace_diagram(img: np.ndarray, factor: int, sigma: float, hue: str,
                    inches=(1.55, 1.55), dpi=260) -> tuple[str, str]:
    """The forward model, measured on this slice: keep the centre of k-space.

    The panel is this slice's own Fourier transform with the retained block
    outlined, so the process box is a measurement of the degradation rather than
    a picture of one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    k = np.fft.fftshift(np.fft.fft2(img))
    h, w = k.shape
    kh, kw = max(1, h // (2 * factor)), max(1, w // (2 * factor))

    fig, ax = plt.subplots(figsize=inches, facecolor="white")
    ax.imshow(np.log1p(np.abs(k)), cmap="gray")
    ax.add_patch(Rectangle((w // 2 - kw, h // 2 - kh), 2 * kw, 2 * kh,
                           fill=False, edgecolor=hue, linewidth=1.1))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_edgecolor("#8A8F96")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0.01, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    cap = (f"k-space of this slice &middot; keep the central 1/{factor} per axis "
           f"&middot; invert &middot; Rician &sigma;={sigma:g}")
    return base64.b64encode(buf.getvalue()).decode(), cap


def _mask_overlay(pred: np.ndarray, hue: str) -> np.ndarray:
    """Solid fill in one model's hue wherever it predicted tumor."""
    from matplotlib.colors import to_rgba

    fill = np.zeros((*pred.shape, 4))
    fill[...] = to_rgba(hue, alpha=0.62)
    fill[..., 3] *= (pred >= 0.5)
    return fill


def stage_panels(device: str, pool: int) -> tuple[dict, dict, dict, dict]:
    """Run one real held-out slice through the real models, panel by panel.

    Returns the panels and the process diagrams as base64 PNGs, the per-version
    metrics for that slice, and the degradation settings the models were trained
    with.
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
    # The process boxes: the forward model measured on this slice, and the two
    # networks drawn from the modules that were just loaded.
    from src.palette import LIGHT as _L
    procs = {
        "degrade": _kspace_diagram(arr(hr), factor, sigma, _L["true"]),
        "sr": _unet_diagram(sr_t, _L["distortion"], "SR U-Net, one of two"),
        "seg": _unet_diagram(seg, _L["tumor_aware"], "Segmentation U-Net",
                             frozen=True),
        "mc": _unet_diagram(sr_t, UNCERTAINTY_RAMP[-2], "Same SR U-Net",
                            passes=10),
    }
    meta = {"size": size, "factor": factor, "sigma": sigma,
            "case": i, "lesions": rows["low-res"]["lesions"]}
    return panels, procs, rows, meta


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
  .wrap{max-width:1180px; margin:0 auto; padding:1.6rem 1.3rem 2rem}
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

  /* the process box: what the stage does, drawn from the code that does it */
  /* the process box is a small square, centred under its plate; the wrapper has
     a fixed height so the numbered tabs stay on one line across all four stages */
  .procwrap{margin-top:.6rem; height:158px; position:relative; z-index:1}
  .procbox{width:112px; height:112px; margin:0 auto; border:1.6px solid #000;
    background:#fff; padding:3px; display:flex; align-items:center}
  .procbox img{display:block; width:100%; height:100%; object-fit:contain}
  .proccap{margin:.3rem 0 0; font-size:.56rem; line-height:1.3; text-align:center;
    color:#4A4F56}
  .proccap b{color:#14161A}

  /* the 2D panels, moved out of the diagram into their own strip */
  .card{background:#fff; border:1px solid var(--rule); border-radius:6px;
    padding:1rem; margin-top:.9rem; overflow-x:auto; color:#14161A}
  .strip{display:flex; gap:.6rem; min-width:760px}

  /* the legend, under the stage it describes */
  .legend{margin-top:1rem}
  .bar{display:flex; align-items:baseline; gap:.5rem; padding:.28rem .7rem; color:#fff}
  .bar .no{font-family:var(--sans); font-weight:700; font-size:1.55rem; line-height:1.2}
  .bar .ttl{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em;
    text-transform:uppercase}
  .items{list-style:none; margin:.45rem 0 0; padding:0}
  .items li{display:flex; align-items:baseline; gap:.5rem; padding:.28rem 0 .28rem .7rem;
    font-family:var(--mono); font-size:.66rem; letter-spacing:.06em; line-height:1.35;
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


def _barbs(x: float, y: float, tx: float, ty: float, head: float = 10.0,
           deg: float = 28.0) -> str:
    """The two strokes of an open V head, opening behind a tip on tangent (tx, ty)."""
    import math

    n = math.hypot(tx, ty) or 1.0
    tx, ty = tx / n, ty / n
    out = []
    for d in (deg, -deg):
        a = math.radians(d)
        bx = (-tx) * math.cos(a) - (-ty) * math.sin(a)
        by = (-tx) * math.sin(a) + (-ty) * math.cos(a)
        out.append(f'M{x:.1f},{y:.1f} L{x + bx * head:.1f},{y + by * head:.1f}')
    return " ".join(out)


def _merge(w: int = 100, h: int = 96) -> str:
    """Two branches converging into one arrow, for the stage that pools inputs.

    Both reconstructions are handed to the same frozen segmentation network, so
    the arrow
    between those stages is a merge rather than a single sweep: two bends come in
    from above and below and join a short trunk that carries the one head. Same
    stroke and same open V as ``_sweep``, so the figure reads as one hand.
    """
    y = h / 2
    join = w - 28
    branches = (f'<path d="M4,{y - 32:.0f} C34,{y - 32:.0f} 44,{y:.0f} {join},{y:.0f}"/>'
                f'<path d="M4,{y + 32:.0f} C34,{y + 32:.0f} 44,{y:.0f} {join},{y:.0f}"/>')
    trunk = f'<path d="M{join},{y:.0f} L{w - 11},{y:.0f}"/>'
    return (f'<svg class="sweep" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'aria-hidden="true"><g fill="none" stroke="#14161A" stroke-width="1.5" '
            f'stroke-linecap="round">{branches}{trunk}'
            f'<path d="{_barbs(w - 11, y, 1, 0)}"/></g></svg>')


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
    return (f'<svg class="sweep" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'aria-hidden="true"><g fill="none" stroke="#14161A" stroke-width="1.5" '
            f'stroke-linecap="round">'
            f'<path d="M{x0},{y0:.1f} C{x1},{y1:.1f} {x2},{y2:.1f} {x3},{y3:.1f}"/>'
            f'<path d="{_barbs(x3, y3, x3 - x2, y3 - y2)}"/></g></svg>')


def _proc(diagram: tuple[str, str]) -> str:
    """The process box under a plate: what this stage does, not what it output.

    Takes the (image, caption) pair the diagram builders return. The caption is
    HTML rather than baked into the PNG, so it stays sharp while the drawing above
    it can be small.
    """
    b64, cap = diagram
    return (f'<div class="procwrap"><div class="procbox">'
            f'<img src="data:image/png;base64,{b64}" alt=""></div>'
            f'<p class="proccap">{cap}</p></div>')


def _level(no: int, title: str, hue: str, blocks: str, proc: str,
           items: list[tuple[str, str, str]]) -> str:
    """One stage of the stack: plate, then how the stage works, then its legend.

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
      {proc}
      <div class="legend">
        <div class="bar" style="background:{hue}">
          <span class="no">{no}</span><span class="ttl">{title}</span></div>
        <ul class="items">{lis}</ul>
      </div>
    </div>"""


def build(out: str = OUT, device: str | None = None, pool: int = 24,
          with_caption: bool = True) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p, procs, rows, meta = stage_panels(device, pool)

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
        _proc(procs["degrade"]),
        [("Held-out patient", "", C_TRUE),
         (f"k-space &times;{meta['factor']}", f"Rician &sigma;={meta['sigma']}", GREY)])

    lv2 = _level(
        2, "Reconstruction", C_BASE,
        (_block(render("brain3d_distortion.png"), "Distortion-optimal", small=True)
         + _block(render("brain3d_tumor_aware.png"), "Tumor-aware", small=True)),
        _proc(procs["sr"]),
        [("Pixel error only", "", C_BASE),
         ("Lesion-weighted, ours", "", C_OURS)])

    lv3 = _level(
        3, "Segmentation and safety", C_OURS,
        _block(_overlay_frame() or render("brain3d_true.png"), "All three, overlaid"),
        _proc(procs["seg"]),
        [("One frozen segmentation U-Net", "", GREY),
         ("Erased or fabricated", "", C_BASE)])

    lv4 = _level(
        4, "Uncertainty", C_UNC,
        _block(render("brain3d_uncertainty.png"), "Where the model is unsure"),
        _proc(procs["mc"]),
        [("10 dropout passes", "", C_UNC),
         ("Doubt against error", "", GREY)])

    sw_a, sw_b, sw_c = (_sweep("up"), _merge(), _sweep("up"))

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

    strip = "".join(_thumb(p[k], cap) for k, cap in [
        ("hr", "true scan"), ("lr", "degraded"), ("sr_d", "baseline"),
        ("sr_t", "ours"), ("seg_d", "segmentation, baseline"),
        ("seg_t", "segmentation, ours"), ("unc", "uncertainty"),
        ("err", "abs error")])

    entries = "".join(f'<div class="entry"><b>{path}</b><p>{what}</p></div>'
                      for path, what in ENTRY_POINTS)

    note = ""
    if missing:
        note = ('<p class="small"><b>Missing 3D renders:</b> '
                + ", ".join(f"<code>{m}</code>" for m in missing)
                + ". Run <code>python main_demo.py</code> to regenerate them.</p>")

    caption = "" if not with_caption else f"""
    <figcaption><b>Figure 1.</b> One held-out patient, pulled apart left to right
    into the four stages that act on them. Each plate is a 3D volume this pipeline
    produced; the box under it is how that stage works, drawn from the code that
    does it, so the channel widths, the dropout and the truncated k-space are read
    off the modules rather than sketched. The dotted horizontals are alignment
    guides, not flow. The two arrows merging into stage 3 are the point of the
    study: both reconstructions are read by the same frozen segmentation network,
    so the only thing that changes between them is the image handed to it.
    </figcaption>"""

    # The page is the figure and nothing else. The slice strip, the metrics table,
    # the code table and the entry points were here and are still built above:
    # _thumb, code_structure and ENTRY_POINTS are one f-string away if they are
    # wanted back.
    html = f"""<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Architecture &middot; Tumor-Aware MRI Super-Resolution</title>
<style>{CSS}</style>
<div class="wrap">
  <div class="figure"><div class="fig-inner">
    <div class="stack">{lv1}{sw_a}{lv2}{sw_b}{lv3}{sw_c}{lv4}</div>{caption}
  </div></div>
  {note}
</div>"""

    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)
    if missing:
        print("note: missing renders", missing)
    return out


def export_png(html: str, out: str, width: int = 1520, height: int = 900,
               scale: int = 2, pad: int = 12) -> str:
    """Render a figure-only page to a tightly cropped PNG via headless Chrome.

    The deck wants the diagram as a single image. Screenshotting the page we
    already build means the deck and this page can never disagree about what the
    architecture is, which duplicating the markup into ``main_demo.py`` would
    eventually allow.
    """
    import shutil
    import subprocess
    import tempfile

    chrome = next((c for c in CHROMES if os.path.exists(c)), None) or \
        shutil.which("chromium") or shutil.which("google-chrome")
    if not chrome:
        raise SystemExit("no Chrome or Chromium found for --png; set one in CHROMES")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shot = os.path.join(tmp, "shot.png")
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
             f"--force-device-scale-factor={scale}",
             f"--window-size={width},{height}",
             f"--screenshot={shot}", f"file://{os.path.abspath(html)}"],
            check=True, capture_output=True)

        from PIL import Image
        im = Image.open(shot).convert("RGB")
        # crop to the white figure card: everything else on the page is the
        # background colour, so the card is the only near-white region
        px = im.load()
        w, h = im.size
        xs, ys = [], []
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                r, g, b = px[x, y]
                if r > 244 and g > 244 and b > 244:
                    xs.append(x); ys.append(y)
        if not xs:
            raise SystemExit("could not find the figure card in the screenshot")
        box = (max(0, min(xs) + pad), max(0, min(ys) + pad),
               min(w, max(xs) - pad), min(h, max(ys) - pad))
        im.crop(box).save(out)
    print("wrote", out, Image.open(out).size)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pool", type=int, default=24,
                    help="how many held-out slices to score when picking one")
    ap.add_argument("--png", metavar="PATH", nargs="?", const=PNG_OUT,
                    help="also write the figure, without its caption, as a PNG "
                         "for the deck (needs a local Chrome)")
    a = ap.parse_args()
    build(a.out, a.device, a.pool)
    if a.png:
        build(TMP_HTML, a.device, a.pool, with_caption=False)
        try:
            export_png(TMP_HTML, a.png)
        finally:
            os.remove(TMP_HTML)      # scratch, not an artifact


if __name__ == "__main__":
    main()
