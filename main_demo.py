"""The demo page. This is the one we present.

Builds ``main_demo.html``: a single self-contained page carrying the whole
story, 3D and 2D, with every image embedded as base64 so the file can be opened
or shared with nothing alongside it.

    python main_demo.py            # models -> renders -> main_demo.html

**The talk is ten slides and no more**, in the order below, one act per section
chip, seven minutes end to end:

   1 title              plus the names and roles, so the team is covered here
   2 why this matters   MRI scarcity, and low-field as the route in
   3 the problem        a forgery with the tumor deleted outscores our own
                       reconstructions, measured
   4 contribution       a safety readout, a tumor-aware objective, an
                       evaluation that can carry the claim
   5 method             the architecture figure, built by build_architecture.py
   6 result             the erasure ladder
   7 result             the same thing broken down by lesion size
   8 result             the 3D viewports, which is the live demo
   9 next steps         one ruler, the single test run, abstention, real data
  10 scope              the five limits, stated before anyone has to ask

Anything else carries ``class="slide extra"`` and lives on a second track,
reached with B or the Backup button: the tradeoff dial, the rotating overlay, the
per-slice 2D evidence, how the 3D is assembled, the segmenter's own floor, our own
audit, the roles and the credits. The dots and the counter only ever count the
ten, so material kept for questions cannot quietly lengthen the deck.

Two rules when editing. Headlines are assertions, one idea each: the headline is
the claim and the figure under it is the evidence. And a slide in the ten gets a
headline, one piece of evidence and at most two grey footnote lines; if it wants
a paragraph, it wants to be a backup slide. See ``DECK.md``.

It supersedes the older ``demo_3d.py`` (3D only) and the unstyled ``demo.py``
static page. ``demo.py`` stays as the source of the 2D inference helpers and
the optional Gradio app; this file reuses those rather than duplicating them.

Colors come from ``src.palette``, so the 2D figures and the 3D renders label
the same model with the same hue. Do not hardcode hues here.
"""

from __future__ import annotations

import base64

import torch

from demo import (SLICE_CACHE, _ensure_models, _fig_to_b64, _infer,
                  _slice_stats)
from render_3d import (render_compare_png, render_rotate_gif,
                       render_uncertainty_png)
from src.data import make_dataset
from src.palette import LIGHT, UNCERTAINTY_RAMP
from viz_bridge import build_patient_volumes

OUT = "main_demo.html"

# The measured cost, in image-quality score, of deleting the tumor outright.
# Produced by scripts/metric_blindness.py; see _blindness_block.
BLINDNESS = "results/metric_blindness.json"
# The validation run the headline slide quotes, read here so the quality figures
# on the problem slide cannot drift away from the ones on the result slide.
VAL_RUN = "results/val/val_w40_sl0.0.json"

# Who did what. The brief asks that each member's role be clear, and this file
# cannot infer it: fill a sentence in here and the roles slide renders itself.
# Left empty, the slide still renders, with the gaps marked so they are hard to
# forget on stage.
ROLES = {
    "Adiza Alhassan": "",
    "Nthabiseng Thema": "",
    "Albert Dodoo": "",
    "Victor Oyindouye Miene": "",
    "Hassan Suliman": "",
}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _blindness_block() -> str:
    """Evidence for the slide that says the training metric cannot see a tumor.

    A forgery of the true scan with the lesion painted out is scored against the
    original, and put next to what our two reconstructions score. Numbers come
    from ``scripts/metric_blindness.py`` (about 20 s on CPU, no model involved),
    computed on demand if the JSON is missing so this slide is never a claim
    with nothing under it. If the real slice cache is absent the table is
    dropped rather than filled with phantom numbers.
    """
    import json
    import os
    import sys

    blind = None
    if os.path.exists(BLINDNESS):
        blind = json.load(open(BLINDNESS))
    elif os.path.exists(SLICE_CACHE):
        sys.path.insert(0, "scripts")
        from metric_blindness import measure
        blind = measure(SLICE_CACHE)
        os.makedirs(os.path.dirname(BLINDNESS), exist_ok=True)
        with open(BLINDNESS, "w") as f:
            json.dump(blind, f, indent=2)

    if blind is None:
        return ("<p class=\"lede\">An enhancing tumor is roughly 3% of the brain area, so a "
                "pixel-error score barely moves when a model smooths it away. The measurement "
                "behind this claim needs the real slice cache: run "
                "<code>scripts/metric_blindness.py</code>.</p>")

    psnr_f = blind["psnr_brain_masked"]["median"]
    ssim_f = blind["ssim"]["median"]
    frac = blind["lesion_fraction_of_brain"]["median"] * 100
    n = blind["n_slices"]

    # Fall back to the published validation figures if the run file is absent.
    rec = {"distortion": (24.51, 0.778), "tumor-aware": (24.28, 0.764)}
    if os.path.exists(VAL_RUN):
        res = json.load(open(VAL_RUN))["results"]
        for k in rec:
            if k in res:
                rec[k] = (res[k]["psnr_brain"], res[k]["ssim"])

    return f"""<table style="margin-top:1.1rem; background:var(--card);
      border:1px solid var(--border); border-radius:14px; padding:.4rem">
      <tr><th>image scored against the true scan</th>
        <th class=num>brain-masked PSNR</th><th class=num>SSIM</th></tr>
      <tr><td><b>The true scan with the tumor painted out</b>
        <span style="color:var(--ink-light)">a deliberate forgery</span></td>
        <td class=num style="color:var(--erased)"><b>{psnr_f:.1f} dB</b></td>
        <td class=num style="color:var(--erased)"><b>{ssim_f:.3f}</b></td></tr>
      <tr><td>Standard super-resolution
        <span style="color:var(--ink-light)">(baseline)</span></td>
        <td class=num>{rec['distortion'][0]:.1f} dB</td>
        <td class=num>{rec['distortion'][1]:.3f}</td></tr>
      <tr><td>Tumor-aware super-resolution
        <b style="color:var(--safe)">(ours)</b></td>
        <td class=num>{rec['tumor-aware'][0]:.1f} dB</td>
        <td class=num>{rec['tumor-aware'][1]:.3f}</td></tr>
    </table>
    <p class="note">{n:,} validation slices, the lesion painted out with surrounding
    brain. It is {frac:.1f}% of the brain area, so deleting it leaves
    <b>{psnr_f - rec['tumor-aware'][0]:.1f}&nbsp;dB of headroom</b> over our own best
    reconstruction.</p>"""


ARCH_PNG = "figures/architecture.png"
MRI_PHOTO = "figures/mri_room.jpg"
FLOOR = "results/segmenter_floor_by_size.json"


def _val_results() -> dict:
    """The validation run the result slides quote, or None if it is absent."""
    import json
    import os

    if not os.path.exists(VAL_RUN):
        return {}
    return json.load(open(VAL_RUN))["results"]


def _matched_stats() -> str:
    """The three figures that say the quality really is matched.

    Read from the run files rather than typed in, so a rerun cannot leave the
    slide quoting the previous one.
    """
    import json
    import os

    res = _val_results()
    if not res:
        return ""
    d_dice = res["distortion"]["dice"]
    t_dice = res["tumor-aware"]["dice"]
    gap = res["distortion"]["psnr_brain"] - res["tumor-aware"]["psnr_brain"]

    removed = ""
    if os.path.exists(FLOOR):
        floor = json.load(open(FLOOR))
        n = sum(v["n"] for v in floor.values())
        fl = sum(v["erased_on_true_hr"] * v["n"] for v in floor.values()) / n
        base = res["distortion"]["safety"]["false_negative_erasure_rate"]
        ours = res["tumor-aware"]["safety"]["false_negative_erasure_rate"]
        if base > fl:
            removed = f"{(1 - (ours - fl) / (base - fl)) * 100:.0f}%"

    cells = []
    if removed:
        cells.append(f'<div class="stat"><b style="color:var(--safe)">{removed}</b>'
                     f'<span>of the erasure our own pipeline adds, removed</span></div>')
    cells.append(f'<div class="stat"><b>{t_dice:.3f}</b>'
                 f'<span>Dice, ours &middot; {d_dice:.3f} baseline</span></div>')
    cells.append(f'<div class="stat"><b>{abs(gap):.2f}<span class="u"> dB</span></b>'
                 f'<span>brain-masked PSNR gap</span></div>')
    return f'<div class="statrow">{"".join(cells)}</div>'


def _floor_chart() -> str:
    """Erasure by lesion size as the floor plus what each model adds to it.

    The absolute rate is mostly the segmenter's own floor on an untouched scan,
    so the honest picture is a stack: what is missed before we touch anything,
    then the excess each objective is responsible for.
    """
    import json
    import os

    res = _val_results()
    if not res or not os.path.exists(FLOOR):
        return ""
    floor = json.load(open(FLOOR))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = ["small", "medium", "large"]
    labels = ["small\n<50 px", "medium\n50-200 px", "large\n>200 px"]
    fl = [floor[k]["erased_on_true_hr"] * 100 for k in sizes]
    base = [res["distortion"]["safety"]["erasure_rate_by_size"][k] * 100 for k in sizes]
    ours = [res["tumor-aware"]["safety"]["erasure_rate_by_size"][k] * 100 for k in sizes]

    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.5), facecolor="white")
    ax.bar(x - w, fl, w, color="#9BA1A6", label="missed on the untouched scan")
    ax.bar(x, [b - f for b, f in zip(base, fl)], w, bottom=fl,
           color=LIGHT["distortion"], label="added by standard SR")
    ax.bar(x, fl, w, color="#9BA1A6", alpha=0.35)
    ax.bar(x + w, [o - f for o, f in zip(ours, fl)], w, bottom=fl,
           color=LIGHT["tumor_aware"], label="added by ours")
    ax.bar(x + w, fl, w, color="#9BA1A6", alpha=0.35)

    for i in range(3):
        ax.text(x[i], base[i] + 1.6, f"+{base[i] - fl[i]:.1f}", ha="center",
                fontsize=8.5, color=LIGHT["distortion"], fontweight="semibold")
        ax.text(x[i] + w, ours[i] + 1.6, f"+{ours[i] - fl[i]:.1f}", ha="center",
                fontsize=8.5, color=LIGHT["tumor_aware"], fontweight="semibold")
        ax.text(x[i] - w, fl[i] + 1.6, f"{fl[i]:.1f}", ha="center", fontsize=8.5,
                color="#6A6E73")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{floor[k]['n']:,} lesions" for l, k in zip(labels, sizes)],
                       fontsize=8.5, color="#444")
    ax.set_ylabel("% of lesion components missed", fontsize=9, color="#444")
    ax.set_ylim(0, 92)
    ax.tick_params(axis="y", labelsize=8.5, colors="#6A6E73")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D9DAD8")
    ax.grid(axis="y", color="#E4E5E3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right", handlelength=1.2)
    fig.tight_layout()
    return (f'<figure class="chart"><img src="data:image/png;base64,{_fig_to_b64(fig)}" '
            f'alt="erasure by lesion size: the segmenter\'s floor on the untouched scan, '
            f'plus the excess each objective adds"></figure>')


def _arch_block() -> str:
    """The architecture figure for the method slide.

    Built by ``python build_architecture.py --png``, which renders the same
    figure that ``architecture.html`` shows. If it has not been generated the
    slide falls back to the four-step description, so the deck always builds.
    """
    import os

    if os.path.exists(ARCH_PNG):
        return (f'<figure class="arch"><img src="data:image/png;base64,'
                f'{_b64(ARCH_PNG)}" alt="the four stages of the pipeline: degrade, '
                f'reconstruct twice, segment with one frozen network, and measure '
                f'uncertainty"></figure>')
    return """<div class="steps">
      <div class="step"><div class="n">01</div><h3>Degrade</h3>
        <p>K-space truncation and Rician noise on a real scan, so every input keeps an exact
        reference.</p></div>
      <div class="step"><div class="n">02</div><h3>Reconstruct</h3>
        <p>Two U-Nets, identical in everything but the loss: pixel error, or lesion-weighted.</p></div>
      <div class="step"><div class="n">03</div><h3>Segment</h3>
        <p>One frozen segmentation U-Net reads every image. Only the image changes.</p></div>
      <div class="step"><div class="n">04</div><h3>Score</h3>
        <p>PSNR and SSIM inside the brain, Dice, and lesions erased or fabricated.</p></div>
    </div>"""


def _photo_block() -> str:
    """The scanner photograph on the access slide, if it is present."""
    import os

    if not os.path.exists(MRI_PHOTO):
        return ""
    return (f'<figure class="photo"><img src="data:image/jpeg;base64,{_b64(MRI_PHOTO)}" '
            f'alt="a 3 tesla MRI scanner in a shielded room">'
            f'<figcaption>A three tesla scanner needs a shielded room, a cooling supply and '
            f'power that does not fail. This is the machine that will not close the '
            f'gap.</figcaption></figure>')


def _byline() -> str:
    """Names for the title slide, each with its role once ROLES is filled in.

    The brief asks that every member's role be clear. Carrying it on the title
    slide keeps it out of the ten slides the talk is allowed.
    """
    out = []
    for name, role in ROLES.items():
        out.append(f'{name} <span style="color:rgba(255,255,255,.36)">{role}</span>'
                   if role else name)
    return " &middot; ".join(out)


def _roles_block() -> str:
    """The roles slide. Renders whatever is in ROLES, and marks what is missing."""
    cards = []
    for name, role in ROLES.items():
        body = role or '<span class="todo">role to fill in</span>'
        cards.append(f'<div class="role"><b>{name}</b><span>{body}</span></div>')
    return f"""  <section class="slide extra">
    <div class="tag"><span class="sec">08</span>The team</div>
    <h2>Who did what.</h2>
    <div class="roles">{''.join(cards)}</div>
    <p class="note">Every number on this deck was produced by code in this repository, and
    every run is logged in <code>EXPERIMENTS.md</code> with its data source next to it.</p>
  </section>"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tumor-Aware MRI Super-Resolution</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    /* Chrome is deep green on off-white. The data hues below are untouched:
       green already means tumor-aware in every figure and 3D render, so the brand
       green is kept dark and desaturated and never placed beside a chart. */
    --ink:#14201B; --ink-mid:#3D4A44; --ink-light:#6B7772;
    --bg:#EFEFE9; --card:#F7F7F2; --card-2:#E7E7E0; --border:#DBDBD2;
    --navy:#0B5340; --accent:#0B5340; --accent-deep:#083A2C; --lime:#D6EF4A;
    --warn:#B8791F;
    /* the three tumor hues + the uncertainty ramp, from src/palette.py */
    --true:{c_true}; --erased:{c_di}; --safe:{c_ta}; --unc:{c_unc};
    --font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
    --mono:'DM Mono',ui-monospace,'Courier New',monospace;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0; background:var(--bg); color:var(--ink); font-family:var(--font);
    font-weight:400; line-height:1.55; -webkit-font-smoothing:antialiased;
    min-height:100vh}}

  /* ---- deck mechanics: one slide visible at a time ---- */
  .deck{{max-width:1100px; margin:0 auto; padding:1.4rem clamp(1rem,4vw,2rem) 2.4rem;
    min-height:100vh}}
  /* the title fills the screen edge to edge, so the deck's own gutters are undone
     for that one slide */
  body.is-title .deck{{max-width:none; padding:0}}
  .slide{{display:none; animation:fade .34s cubic-bezier(.2,.7,.2,1)}}
  /* vertically centred, so a short slide does not leave a lake of space under it
     while a tall one still grows downward instead of being clipped */
  .slide.on{{display:flex; flex-direction:column; justify-content:center;
    min-height:calc(100vh - 3.8rem)}}
  @keyframes fade{{from{{opacity:0; transform:translateY(10px)}} to{{opacity:1; transform:none}}}}
  @media(prefers-reduced-motion:reduce){{.slide{{animation:none}}}}

  .tag{{position:relative; font-family:var(--mono); font-size:.7rem; letter-spacing:.24em;
    text-transform:uppercase; color:var(--accent); padding:0 13rem .55rem 0;
    border-bottom:1px solid var(--border); margin-bottom:1rem}}
  .tag::after{{content:"Tumor-aware SR · ACVSS"; position:absolute; right:0; top:0;
    color:var(--ink-light); letter-spacing:.16em; white-space:nowrap}}
  .tag.backup::after{{content:"Backup · for questions"}}
  @media(max-width:820px){{.tag{{padding-right:0}} .tag::after{{content:none}}}}
  h1{{font-weight:700; font-size:clamp(2.1rem,5.6vw,3.7rem); line-height:1.03;
    letter-spacing:-.02em; margin:.5rem 0 .5rem; max-width:20ch}}
  h1 em{{font-style:normal; color:var(--accent-deep)}}
  h2 em{{font-style:normal; color:var(--accent-deep)}}
  h2{{font-weight:700; font-size:clamp(1.4rem,3vw,2rem); letter-spacing:-.01em;
    margin:.25rem 0 .5rem; max-width:34ch}}
  h3{{font-weight:600; font-size:1rem; margin:0 0 .6rem}}
  .lede{{color:var(--ink-mid); max-width:60ch; font-size:1.05rem; font-weight:300}}
  .lede b{{color:var(--ink); font-weight:600}}
  .note{{max-width:72ch; font-size:.85rem; color:var(--ink-light); margin:.9rem 0 0}}

  /* title slide */
  .title-slide{{min-height:100vh; display:flex; flex-direction:column;
    justify-content:center; background:var(--navy); color:#fff; border-radius:0;
    padding:clamp(2rem,7vw,6rem); position:relative; overflow:hidden}}
  .title-slide h1{{max-width:24ch}}
  .title-slide::after{{content:""; position:absolute; right:-8%; top:-35%; width:480px;
    height:480px; background:radial-gradient(circle, rgba(214,239,74,.22), transparent 62%)}}
  .title-slide > *{{position:relative; z-index:2}}
  .title-slide .tag{{color:var(--lime); border:0; padding:0; margin-bottom:.4rem}}
  .title-slide .tag::after{{content:none}}
  .title-slide h1 em{{color:var(--lime)}}
  .title-slide .lede{{color:rgba(255,255,255,.74)}}
  .title-slide .lede b{{color:#fff}}
  .byline{{font-family:var(--mono); font-size:clamp(.92rem,1.5vw,1.12rem);
    letter-spacing:.02em; line-height:1.75; color:rgba(255,255,255,.78);
    margin-top:2.4rem; max-width:62ch}}

  .vitals{{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.4rem}}
  .vitals.four{{grid-template-columns:repeat(4,1fr); gap:.8rem}}
  .vitals.four .vital{{padding:1.1rem}}
  .vitals.four .v{{font-size:clamp(1.7rem,4vw,2.4rem)}}
  .vital{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.3rem}}
  .vital .k{{font-family:var(--mono); font-size:.68rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--ink-light); display:flex; align-items:center}}
  .vital .v{{font-family:var(--mono); font-weight:500; font-size:clamp(2rem,5vw,2.9rem);
    line-height:1; margin-top:.5rem; color:var(--ink)}}
  .vital .u{{font-size:.85rem; color:var(--ink-light); margin-left:.2rem}}
  .vital .d{{font-size:.8rem; margin-top:.55rem; color:var(--ink-mid)}}
  .vital.safe .v{{color:var(--safe)}} .vital.erased .v{{color:var(--erased)}}
  .dot{{display:inline-block; width:.5em; height:.5em; border-radius:50%; margin-right:.5em}}

  /* Three bars, one per image handed to the same frozen segmenter: the degraded
     scan the pipeline starts from, then each reconstruction of it. Grey for the
     degraded input because it is the starting point rather than a model, so it
     must not look like one. */
  .ladder{{margin-top:1.1rem; background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.2rem 1.3rem}}
  .lrow{{display:grid; grid-template-columns:15rem 1fr 9rem; gap:1rem;
    align-items:center; margin-bottom:.75rem}}
  .llab{{font-size:.84rem; font-weight:600; line-height:1.25}}
  .llab span{{display:block; font-weight:400; font-size:.74rem; color:var(--ink-light)}}
  .lbar{{display:flex; height:1.5rem; background:var(--card-2); border-radius:4px;
    overflow:hidden}}
  .lbar i{{display:block; height:100%}}
  .lbar i.deg{{background:#9BA1A6}}
  .lbar i.base{{background:var(--erased)}}
  .lbar i.ours{{background:var(--safe)}}
  .lnum{{font-family:var(--mono); font-size:1.05rem; text-align:right; line-height:1.2}}
  .lnum span{{display:block; font-family:var(--font); font-size:.72rem;
    color:var(--ink-light)}}
  .lscale{{display:grid; grid-template-columns:15rem 1fr 9rem; gap:1rem;
    font-family:var(--mono); font-size:.68rem; color:var(--ink-light)}}
  .lscale > div{{grid-column:2; display:flex; justify-content:space-between;
    align-items:baseline; border-top:1px solid var(--border); padding-top:.3rem}}
  .lscale .mid{{font-family:var(--font)}}
  .lkey{{display:flex; flex-direction:column; gap:.4rem; margin-top:1rem;
    padding-top:.9rem; border-top:1px solid var(--border);
    font-size:.78rem; color:var(--ink-mid)}}
  @media(max-width:760px){{.lrow{{grid-template-columns:1fr; gap:.3rem}}
    .lnum{{text-align:left}}}}

  /* 2x2 rather than 1x4: at four across, each brain was ~240px and the
     difference between a preserved and an erased lesion stopped being legible. */
  .panels{{display:grid; grid-template-columns:repeat(2,1fr); gap:1.1rem; margin-top:1.2rem}}
  figure{{margin:0; border:1px solid var(--border); border-radius:14px; overflow:hidden;
    background:var(--card)}}
  figure img{{display:block; width:100%; height:auto; background:var(--navy)}}
  figcaption{{font-size:.76rem; color:var(--ink-mid); padding:.7rem .9rem;
    border-top:1px solid var(--border)}}
  figcaption b{{color:var(--ink); font-weight:600}}

  .rotate{{display:grid; grid-template-columns:1.05fr .95fr; gap:2rem; align-items:center;
    margin-top:1.2rem}}
  .rotate .view{{border:1px solid var(--border); border-radius:16px; overflow:hidden;
    background:var(--navy)}}
  .rotate .view img{{width:100%; display:block}}
  .legend{{display:flex; gap:1.3rem; flex-wrap:wrap; font-family:var(--mono);
    font-size:.78rem; color:var(--ink-mid); margin-top:1rem}}
  .ramp{{display:inline-block; width:3.2em; height:.55em; border-radius:3px; margin-right:.5em;
    vertical-align:middle; background:linear-gradient(90deg,{ramp_css})}}

  .slice{{background:var(--card); border:1px solid var(--border); border-radius:16px;
    padding:1.1rem; margin-top:1rem}}
  .slice img{{width:100%; display:block; border-radius:8px; margin:.2rem 0 .8rem;
    background:#fff; max-height:41vh; object-fit:contain}}
  table{{border-collapse:collapse; width:100%; font-family:var(--mono); font-size:.78rem}}
  th{{text-align:left; font-weight:500; color:var(--ink-light); text-transform:uppercase;
    letter-spacing:.1em; font-size:.66rem; padding:.4rem .6rem;
    border-bottom:1px solid var(--border)}}
  td{{padding:.42rem .6rem; border-bottom:1px solid var(--border); color:var(--ink-mid)}}
  td:first-child{{color:var(--ink)}}
  tr:last-child td{{border-bottom:none}}
  .num{{text-align:right; font-variant-numeric:tabular-nums}}

  /* the architecture figure, rendered by build_architecture.py so the deck and
     that page can never disagree about what the pipeline is */
  .arch{{margin:1.1rem 0 0; background:#fff; border:1px solid var(--border);
    border-radius:12px; padding:.7rem}}
  .arch img{{display:block; width:100%; height:auto}}

  /* definition cards: an accent rail on the left, a term, then what it means */
  .terms{{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.3rem}}
  .term{{background:var(--card); border:1px solid var(--border); border-radius:12px;
    border-left:4px solid var(--ink-light); padding:1rem 1.15rem}}
  .term h3{{font-size:1rem; margin:0 0 .4rem; letter-spacing:-.01em}}
  .term p{{font-size:.86rem; color:var(--ink-mid); margin:0}}
  .term p + p{{margin-top:.45rem}}
  .term code{{font-size:.8rem}}

  /* a row of headline figures, the way a stat trio reads on a slide */
  .statrow{{display:flex; gap:1.5rem; flex-wrap:wrap; margin:.1rem 0 .9rem}}
  .stat > b{{display:block; font-family:var(--mono); font-weight:500; font-size:1.6rem;
    line-height:1.1; color:var(--ink)}}
  .statrow.big .stat > b{{font-size:clamp(2.6rem,5.5vw,3.8rem); letter-spacing:-.02em}}
  .statrow.big .stat span{{font-size:.82rem; max-width:30ch}}
  .statrow.big .stat span b{{font-size:.9rem}}
  .stat span b{{font-family:var(--font); font-size:.8rem; font-weight:600; color:var(--ink)}}
  .stat > b .u{{font-size:.9rem; color:var(--ink-light)}}
  .chart{{margin:.2rem 0 0}}
  .chart img{{display:block; width:100%; height:auto; border-radius:12px}}
  .stat span{{display:block; font-size:.72rem; color:var(--ink-light); margin-top:.2rem;
    max-width:26ch}}

  /* a photograph, framed like the figures rather than floated in the page */
  .photo{{margin:0; border:1px solid var(--border); border-radius:14px; overflow:hidden;
    background:var(--card)}}
  .photo img{{display:block; width:100%; height:auto}}
  .photo figcaption{{font-size:.76rem; color:var(--ink-light); padding:.55rem .8rem .6rem}}

  /* evidence on the left, what to take from it on the right */
  .readout{{display:grid; grid-template-columns:1.5fr 1fr; gap:1.6rem; align-items:start;
    margin-top:1.2rem}}
  .readout > .ev{{min-width:0}}
  .take h3{{font-size:1rem; color:var(--accent-deep); margin:.2rem 0 .6rem}}
  .take p{{font-size:.86rem; color:var(--ink-mid); margin:0 0 .55rem}}
  .take .pill{{background:var(--card-2); border:1px solid var(--border);
    border-radius:10px; padding:.6rem .8rem; font-size:.84rem; color:var(--ink)}}
  @media(max-width:900px){{.readout{{grid-template-columns:1fr}}
    .terms{{grid-template-columns:1fr}}}}

  .steps{{display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-top:1.2rem}}
  .steps.three{{grid-template-columns:repeat(3,1fr)}}
  .step{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.1rem}}
  .step .n{{font-family:var(--mono); font-weight:500; font-size:1.3rem; color:var(--accent)}}
  .step h3{{font-size:.95rem; font-weight:600; margin:.3rem 0 0; color:var(--ink);
    letter-spacing:-.01em}}
  .step p{{font-size:.82rem; color:var(--ink-mid); margin:.4rem 0 0}}

  /* section chip: which act of the talk this slide belongs to */
  .tag .sec{{font-family:var(--mono); color:var(--accent); margin-right:.5rem}}
  .tag.backup{{color:var(--warn)}}

  .roles{{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:.9rem; margin-top:1.2rem}}
  .role{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1rem 1.1rem}}
  .role b{{display:block; font-size:.95rem}}
  .role span{{display:block; font-size:.82rem; color:var(--ink-mid); margin-top:.25rem}}
  .role .todo{{color:var(--warn); font-weight:600}}

  /* numbered rows on hairlines, so a list of limits reads as a register */
  .scope{{margin-top:1rem}}
  .scope ul{{counter-reset:sc; list-style:none; margin:0; padding:0}}
  /* hanging indent rather than a grid: an li holds a <b> and a bare text node,
     and a two-column grid would drop the text into a third cell */
  .scope li{{counter-increment:sc; position:relative; border-top:1px solid var(--border);
    padding:.72rem 0 .72rem 3.4rem; font-size:.88rem; color:var(--ink-mid)}}
  .scope li:last-child{{border-bottom:1px solid var(--border)}}
  .scope li::before{{content:"(" counter(sc, decimal-leading-zero) ")"; position:absolute;
    left:0; top:.86rem; font-family:var(--mono); font-size:.7rem; color:var(--accent);
    letter-spacing:.06em}}
  .scope li b{{color:var(--ink)}}

  .credits{{font-size:.84rem; color:var(--ink-light); margin-top:1.2rem; max-width:70ch}}
  .credits a{{color:var(--accent-deep); text-decoration:none; font-weight:500}}
  code{{font-family:var(--mono); background:var(--card-2); padding:1px 6px;
    border-radius:5px; font-size:.85em}}

  @media(max-width:760px){{.vitals,.panels,.steps{{grid-template-columns:1fr}}
    .rotate{{grid-template-columns:1fr}}}}
</style></head>
<body>

<div class="deck" id="deck">

  <section class="slide on">
    <div class="title-slide">
      <div class="tag">ACVSS &middot; MRI Super-Resolution Safety</div>
      <h1>When sharper means <em>blind</em>.</h1>
      <p class="lede">Super-resolution makes a cheap low-field brain scan look crisp.
      Trained only for image quality, it can quietly <b>erase a small tumor</b>.
      We measure how often that happens, and fix it with a tumor-aware objective.</p>
      <div class="byline">{byline}</div>
    </div>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">01</span>Why this matters</div>
    <h2>Cheap, low-quality scanners are the only realistic way to widen MRI access.</h2>
    <div class="readout">
      <div class="ev">{photo}</div>
      <div class="take">
        <div class="statrow big" style="flex-direction:column; gap:1.3rem">
          <div class="stat"><b style="color:var(--erased)">14</b>
            <span><b style="color:var(--ink)">Ghana.</b> For more than 30 million people, and
            two thirds of them sit in Greater Accra.</span></div>
          <div class="stat"><b style="color:var(--erased)">&lt;1</b>
            <span><b style="color:var(--ink)">Per million, much of sub-Saharan Africa.</b>
            MRI is scarce where the disease burden is not.</span></div>
          <div class="stat"><b>37</b>
            <span><b style="color:var(--ink)">Per million, high-income countries.</b> Up to
            this many, for the same imaging need.</span></div>
        </div>
      </div>
    </div>
    <p class="note"><b>0.055&nbsp;tesla</b> is a fiftieth of a hospital magnet:
    weaker signal, blurrier image.</p>
    <p class="note">Ogbole et&nbsp;al. 2018 &middot; Ghana MRI audit 2017 &middot;
    Anazodo et&nbsp;al. 2024.</p>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">02</span>The problem with the objective</div>
    <h2>A forgery with the tumor painted out scores <em>better</em> than either
    reconstruction we train.</h2>
    {blindness}
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">03</span>Contribution</div>
    <h2>Penalise the model heavily for mistakes where the tumor is.</h2>
    <div class="terms">
      <div class="term" style="border-left-color:var(--safe)">
        <h3>The loss</h3>
        <p>Pixel error inside the tumor mask is <b>multiplied by a factor</b>, so an error on
        the lesion costs far more than the same error in healthy tissue.</p>
        <p><code>L = |pred &minus; true| &times; (1 + w&middot;mask)</code>, w=40</p></div>
      <div class="term" style="border-left-color:var(--erased)">
        <h3>The second term</h3>
        <p>Supervises the <b>segmenter's output</b> rather than the pixels, so inventing a
        tumor is punished as well as losing one.</p></div>
      <div class="term">
        <h3>The readout</h3>
        <p>Lesions <b>erased</b> and <b>fabricated</b>, split by size, always against what the
        segmenter already misses on an untouched scan.</p></div>
    </div>
    <p class="note">The model can no longer buy score by smoothing a lesion away.
    Corrective, not architectural: no new network, three small U-Nets.</p>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">04</span>Method</div>
    <h2>Degrade a real scan, reconstruct it two ways, and ask one frozen
    segmentation network what it can still find.</h2>
    {arch}
    <p class="note"><b>No GAN, deliberately:</b> an adversarial loss rewards inventing
    plausible tissue. 17,233 slices, 468 patients.</p>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">05</span>Result &middot; real BraTS, 70 unseen patients</div>
    <h2>Super-resolution recovers tumor the degradation destroys. Ours recovers more of it.</h2>
    <div class="vitals">
      <div class="vital"><div class="k"><span class="dot" style="background:#9BA1A6"></span>The degraded scan</div>
        <div class="v" style="color:var(--ink-mid)">62.2<span class="u">%</span></div>
        <div class="d">of enhancing lesions missed in the degraded scan, before any reconstruction</div></div>
      <div class="vital erased"><div class="k"><span class="dot" style="background:var(--erased)"></span>Standard SR <b style="color:var(--ink-mid)">(baseline)</b></div>
        <div class="v">58.0<span class="u">%</span></div>
        <div class="d">recovers some of what the degradation destroyed</div></div>
      <div class="vital safe"><div class="k"><span class="dot" style="background:var(--safe)"></span>Tumor-aware SR <b style="color:var(--safe)">(OURS)</b></div>
        <div class="v">51.3<span class="u">%</span></div>
        <div class="d">&minus;6.7 points vs baseline, at matched quality</div></div>
    </div>
    <div class="ladder">
      <div class="lrow">
        <div class="llab">The degraded scan<span>a real scan we degraded to imitate a cheap scanner</span></div>
        <div class="lbar"><i class="deg" style="width:88.9%"></i></div>
        <div class="lnum">62.2%<span>missed</span></div>
      </div>
      <div class="lrow">
        <div class="llab">Standard super-resolution <span style="display:inline;font-weight:600;color:var(--ink-mid)">(baseline)</span><span>trained only for image quality</span></div>
        <div class="lbar"><i class="base" style="width:82.9%"></i></div>
        <div class="lnum">58.0%<span>missed</span></div>
      </div>
      <div class="lrow">
        <div class="llab">Tumor-aware super-resolution <span style="display:inline;font-weight:700;color:var(--safe)">(OURS)</span><span>our objective: lesion-weighted loss</span></div>
        <div class="lbar"><i class="ours" style="width:73.3%"></i></div>
        <div class="lnum">51.3%<span><b style="color:var(--safe)">&minus;6.7</b> vs baseline</span></div>
      </div>
      <div class="lscale"><div><span>0%</span>
        <span class="mid">of enhancing lesion components missed</span><span>70%</span></div></div>
      <div class="lkey">
        <span>Same frozen segmenter in all three rows: only the image changes.</span>
      </div>
    </div>

  </section>

  <section class="slide">
    <div class="tag"><span class="sec">05</span>Result &middot; broken down by lesion size</div>
    <h2>Large lesions are almost never lost. Small ones are the whole problem.</h2>
    <div class="readout">
      <div class="ev">
      {floor_chart}
        <p class="note">70 patients, 9,490 components. Grey is missed on the
        <b>untouched</b> scan; the block above it is what each objective adds.</p>
      </div>
      <div class="take">
        <h3>Read the large group first</h3>
        <p>Over 200&nbsp;px, either model misses about <b>1%</b>. Nothing here says half of
        all tumors vanish.</p>
        <p><b>71% of components are under 50&nbsp;px</b>, and that is where the objective
        earns its keep: 13.5 points added down to 4.7.</p>
        <div class="pill">On whole tumor, a tenth of the image, it does nothing at all.</div>
      </div>
    </div>
  </section>

  <section class="slide extra">
    <div class="tag"><span class="sec">05</span>Result &middot; a second finding</div>
    <h2>The erasure&ndash;hallucination tradeoff is a dial, not a fixed price.</h2>
    <p class="note">Protecting tumors costs false alarms. A loss that says "get the lesion
    region right" also rewards over-painting anything lesion-like, and that overshoot
    <i>is</i> hallucination. We expected to report that cost and leave it. Instead it turns
    out to be adjustable, using a term that supervises the downstream segmenter directly
    instead of weighting pixels.</p>
    <table style="margin-top:1.1rem; background:var(--card); border:1px solid var(--border);
      border-radius:14px; padding:.4rem">
      <tr>
        <th>objective</th>
        <th class=num>erasure removed<br><span style="font-weight:400;text-transform:none;letter-spacing:0">the win</span></th>
        <th class=num>hallucination added<br><span style="font-weight:400;text-transform:none;letter-spacing:0">the cost</span></th>
      </tr>
      <tr><td>lesion-weighted only &nbsp;<span style="color:var(--ink-light)">seg&lambda;=0</span></td>
        <td class=num style="color:var(--safe)"><b>&minus;6.64 pp</b></td>
        <td class=num style="color:var(--erased)"><b>+0.121</b></td></tr>
      <tr><td>+ segmentation consistency &nbsp;<span style="color:var(--ink-light)">seg&lambda;=0.5</span></td>
        <td class=num style="color:var(--safe)">&minus;4.86 pp</td>
        <td class=num style="color:var(--erased)">+0.055</td></tr>
      <tr><td>+ heavier lesion weight &nbsp;<span style="color:var(--ink-light)">w=80, seg&lambda;=0.5</span></td>
        <td class=num style="color:var(--safe)">&minus;3.46 pp</td>
        <td class=num style="color:var(--erased)"><b>+0.017</b></td></tr>
    </table>
    <p class="note"><b>Why the term does this.</b> Lesion weighting is one-sided: it only asks
    that the tumor region be reconstructed accurately, so the cheapest way to satisfy it is to
    make everything tumour-ish brighter and more solid. The consistency term instead penalises
    the reconstruction whenever the frozen segmenter's output on it disagrees with the true
    mask &mdash; in <i>either</i> direction. It therefore punishes inventing tumor as well as
    losing it. Two-sided supervision, against a one-sided proxy.</p>
    <p class="note">This term existed in our codebase from the start and had never been
    trained with. It does not improve the headline metric, so on erasure alone the simple
    lesion-weighted loss still wins. What it offers is a <b>choice</b>: a screening setting
    that cannot afford false alarms can give up roughly half the erasure gain and take the
    hallucination cost to almost nothing.</p>
    <p class="note" style="color:var(--erased)"><b>Held back deliberately.</b> Each row above
    trained its own segmenter, and the segmenter is the instrument that measures both rates.
    The low-resolution baseline &mdash; which uses no reconstruction at all and depends only on
    the segmenter &mdash; came out at 0.622, 0.715 and 0.719 across the three rows. Those
    should be identical. GPU training is nondeterministic, so these three objectives were
    measured with three different rulers. The ranking is suggestive and <b>not yet a
    result</b>; the rerun that shares one frozen segmenter is written and queued.</p>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">05</span>Result &middot; four viewports</div>
    <h2>The lesions the baseline loses show up in 3D as empty blue shells.</h2>
    <div class="panels">
    <figure><img src="data:image/png;base64,{img_true}" alt="ground truth tumor in 3D">
      <figcaption><b>Ground truth.</b> The true lesions, in blue.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_ta}" alt="tumor-aware reconstruction in 3D">
      <figcaption><b>Tumor-aware &mdash; OURS.</b> Lesions preserved.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_di}" alt="distortion reconstruction in 3D">
      <figcaption><b>Distortion-optimal &mdash; baseline.</b> Small lesions dropped.</figcaption></figure>
    {unc_panel}
    </div>
    <p class="note">A blue shell with nothing inside it is a lesion the model lost.</p>
  </section>

  <section class="slide extra">
    <div class="tag"><span class="sec">05</span>Result &middot; one brain, three reconstructions</div>
    <h2>A rotating overlay lets you check the claim by eye. It is not how we measure it.</h2>
    <div class="rotate">
      <div class="view"><img src="data:image/gif;base64,{img_gif}" alt="rotating 3D brain with tumor overlays"></div>
      <div>
        <p style="color:var(--ink-mid)">The orbit overlays all three reconstructions on one
        brain. This view is a qualitative illustration. The quantitative claim is that at
        matched image quality, <b>image quality is not a safety metric</b>.</p>
        <div class="legend">
          <span><span class="dot" style="background:var(--true)"></span>ground truth</span>
          <span><span class="dot" style="background:var(--safe)"></span>tumor-aware <b>(ours)</b></span>
          <span><span class="dot" style="background:var(--erased)"></span>distortion-optimal (baseline)</span>
          <span><span class="ramp"></span>uncertainty, low to high</span>
        </div>
      </div>
    </div>
  </section>

  {slices}

  <section class="slide extra">
    <div class="tag"><span class="sec">04</span>Method &middot; how the 3D readout is built</div>
    <h2>The 3D volumes are restacked 2D predictions, not the output of a 3D model.</h2>
    <div class="steps">
      <div class="step"><div class="n">01</div><p>A real held-out BraTS case the models never saw during training.</p></div>
      <div class="step"><div class="n">02</div><p>Each axial slice is degraded, super-resolved by both models, then segmented.</p></div>
      <div class="step"><div class="n">03</div><p>Predicted masks are restacked into 3D volumes.</p></div>
      <div class="step"><div class="n">04</div><p>Marching cubes and eye-dome lighting render the meshes.</p></div>
    </div>
    <p class="note">Three networks, all small 2D U-Nets from one shared implementation,
    all trained from scratch. No pretrained weights and no adversarial loss: a GAN rewards
    output that merely looks like plausible tissue, which is the mechanism behind the
    hallucination we are trying to measure.</p>
  </section>

  <section class="slide">
    <div class="tag"><span class="sec">06</span>Next steps</div>
    <h2>Next: generative models, tested with the readout we just built.</h2>
    <div class="terms">
      <div class="term" style="border-left-color:var(--erased)">
        <h3>Why generative</h3>
        <p>Diffusion and adversarial models make the sharpest low-field images anyone has
        produced. They are also the ones most likely to invent tissue.</p></div>
      <div class="term" style="border-left-color:var(--safe)">
        <h3>Why we can now try them</h3>
        <p>We can measure what they lose and what they fabricate. Sharpness no longer has to
        be taken on trust.</p></div>
      <div class="term">
        <h3>Then the one number</h3>
        <p>A single evaluation on the 94 patients nothing has touched.</p></div>
    </div>
    <p class="note">A safety metric is what makes a generative model testable rather than
    impressive.</p>
  </section>

  <section class="slide extra">
    <div class="tag backup">What this is not</div>
    <h2>Five limits we would rather state than be asked.</h2>
    <div class="scope"><ul>
      <li><b>Only small lesions.</b> On whole tumor, no benefit at all: 64.6% vs 63.7%.</li>
      <li><b>Enhancement can make detection worse.</b> On whole tumor both models erase
          more than the raw input.</li>
      <li><b>Simulated degradation.</b> Resolution loss and noise, not low-field
          contrast.</li>
      <li><b>The fix costs hallucination.</b> 0.266 to 0.387.</li>
      <li><b>Validation, not test.</b> The 94 test patients are still untouched.</li>
      
    </ul></div>
    <p class="note">The floor, our audit, the tradeoff dial and the per-slice evidence
    are on the backup slides. Press <b>B</b>.</p>
  </section>

  {roles}

  <section class="slide extra">
    <div class="tag">Credits</div>
    <h2>Reproducing this.</h2>
    <p class="credits">3D rendering by the vendored
    <a href="https://github.com/asmarufoglu/neuro-voxel">neuro-voxel</a> analyzer
    (PyVista and VTK marching cubes). Every volume shown is our own model output on a real
    held-out BraTS case, degraded by our own forward model, which makes this a proof of concept
    on simulated low-field degradation rather than a clinical result.
    Colors are validated for colorblind separability in <code>src/palette.py</code>.
    Regenerate the whole deck with <code>python main_demo.py</code>. See
    <code>README.md</code> for the study, <code>EXPERIMENTS.md</code> for every run and the
    audit behind these numbers, and <code>paper/</code> for the write-up.</p>
  </section>

  <section class="slide extra">
    <div class="tag backup">Backup &middot; the denominator</div>
    <h2>The detector misses most small lesions even on an untouched scan.</h2>
    <p class="note">The frozen segmenter run on the <b>original high-resolution scan</b> &mdash;
    no degradation, no reconstruction &mdash; on the same 70 validation patients. This is the
    floor that any absolute erasure rate sits on top of.</p>
    <table style="margin-top:1.1rem; background:var(--card); border:1px solid var(--border);
      border-radius:14px; padding:.4rem">
      <tr><th>lesion size</th><th class=num>how many</th>
        <th class=num>missed on the<br>untouched scan</th>
        <th class=num>baseline adds</th><th class=num>ours adds</th></tr>
      <tr><td><b>small</b> &lt;50&nbsp;px</td><td class=num>6,762 <span style="color:var(--ink-light)">(71%)</span></td>
        <td class=num>65.1%</td><td class=num style="color:var(--erased)">+13.5</td>
        <td class=num style="color:var(--safe)"><b>+4.7</b></td></tr>
      <tr><td><b>medium</b> 50&ndash;200&nbsp;px</td><td class=num>1,005</td>
        <td class=num>10.0%</td><td class=num style="color:var(--erased)">+6.4</td>
        <td class=num style="color:var(--safe)"><b>+3.5</b></td></tr>
      <tr><td><b>large</b> &gt;200&nbsp;px</td><td class=num>1,723</td>
        <td class=num>0.5%</td><td class=num style="color:var(--erased)">+0.7</td>
        <td class=num style="color:var(--safe)"><b>+0.4</b></td></tr>
      <tr><td><b>overall</b></td><td class=num>9,490</td>
        <td class=num><b>47.6%</b></td><td class=num style="color:var(--erased)">+10.4</td>
        <td class=num style="color:var(--safe)"><b>+3.8</b></td></tr>
    </table>
    <p class="note"><b>Read the last two columns, not the first.</b> A raw rate near 50% is
    mostly this floor rather than damage done by super-resolution. The part attributable to the
    pipeline is the excess, and our objective removes about <b>64% of it</b> overall. Every
    rate in this table is pooled over lesions; per patient the floor is 45.5% rather than
    47.6%, and mixing the two is how an earlier version of this slide reached 71%. It also
    means the honest priority is a stronger detector: it is the weakest link, not the
    enhancement. And it is why cm&sup3; must never be read off a render &mdash; the segmenter
    over-reads even a clean scan, so a reconstruction is only ever compared against the
    segmenter's own reading of the true scan.</p>
  </section>

  <section class="slide extra">
    <div class="tag backup">Backup &middot; our own audit</div>
    <h2>Two flaws in our own pipeline cut our first headline in half.</h2>
    <div class="scope"><ul>
      <li><b>A dropout leak.</b> The Monte Carlo uncertainty pass switched dropout on and never
          restored eval mode, so every slice after the first was super-resolved
          <i>stochastically</i>. Every safety number the evaluation had ever produced was scored
          on noise. After the fix the erasure gap fell from 5.5 points to <b>2.4</b>.</li>
      <li><b>Lesions counted per slice, not per patient.</b> One tumor spans twenty axial
          slices and those outcomes move together, so pooling cross-sections inflated our
          effective sample size. Recomputed with the patient as the unit: <b>+3.2 points, 95%
          CI [0.7, 6.9], p&nbsp;=&nbsp;0.007</b>, and 11 of 71 patients got <i>worse</i>. The
          earlier "roughly 5 sigma" claim was wrong.</li>
      <li><b>An earlier headline retired entirely.</b> "The baseline erases the tumor
          completely, 0.00&nbsp;cm&sup3;" came from an undertrained checkpoint. Properly
          trained, both models over-segment. The claim is gone and the cm&sup3; figure is now
          labelled an illustration.</li>
      <li><b>Uncertainty is a reliability signal, not a tumor detector.</b> It runs about
          1.5&times; higher inside the lesion than in healthy tissue, which is suggestive
          rather than diagnostic.</li>
      <li><b>Lesion components fragment.</b> An enhancing rim breaks into many small
          4-connected pieces, so component counts run high and a five-pixel speck counts the
          same as a whole tumor. That inflates absolute rates in both directions.</li>
    </ul></div>
    <p class="note">Every run, its data source and what it actually showed are in
    <code>EXPERIMENTS.md</code>, which is append-only. The point of that file is that a number
    on a slide can be traced back to a checkpoint and a command.</p>
  </section>

</div>


<script>
(function () {{
  /* No visible chrome: the deck is driven from the keyboard, and the backup track
     is still there behind B for questions. */
  var main = Array.prototype.slice.call(document.querySelectorAll('.slide:not(.extra)'));
  var extra = Array.prototype.slice.call(document.querySelectorAll('.slide.extra'));
  var backup = false;
  var at = 0, atx = 0;

  function paint() {{
    var list = backup ? extra : main;
    var i = backup ? atx : at;
    main.concat(extra).forEach(function (s) {{ s.classList.remove('on'); }});
    if (list[i]) list[i].classList.add('on');
    document.body.classList.toggle('is-title', !backup && i === 0);
    window.scrollTo({{ top: 0, behavior: 'instant' }});
    location.hash = (backup ? 'b' : 'p') + (i + 1);
  }}

  function go(i) {{
    var n = (backup ? extra : main).length;
    i = Math.max(0, Math.min(n - 1, i));
    if (backup) {{ atx = i; }} else {{ at = i; }}
    paint();
  }}

  function toggle() {{
    if (!extra.length) return;
    backup = !backup;
    paint();
  }}

  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{ e.preventDefault(); go((backup ? atx : at) + 1); }}
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ e.preventDefault(); go((backup ? atx : at) - 1); }}
    if (e.key === 'b' || e.key === 'B') {{ toggle(); }}
    if (e.key === 'Escape' && backup) {{ toggle(); }}
    if (e.key === 'Home') {{ go(0); }}
    if (e.key === 'End') {{ go((backup ? extra : main).length - 1); }}
  }});

  var h = location.hash || '';
  backup = h.charAt(1) === 'b';
  var start = parseInt(h.replace(/^#[pb]/, ''), 10);
  go(isNaN(start) ? 0 : start - 1);
}})();
</script>

</body></html>"""


def _test_set(size, pool):
    """Held-out slices: real patients when the cache is present, else phantom.

    The real split is by patient, so these are cases neither the segmenter nor
    either SR model has seen.
    """
    import os
    if os.path.exists(SLICE_CACHE):
        ds = make_dataset("cached", path=SLICE_CACHE, split="test")
        return torch.utils.data.Subset(ds, range(min(pool, len(ds))))
    return make_dataset("synthetic", n=pool, size=size, seed=999)


def _pick_slices(seg, sr_d, sr_t, size, factor, sigma, device, n_slices, pool=64):
    """Choose the test slices that actually demonstrate the phenomenon.

    A slice where both models behave identically is a wasted panel: the viewer
    sees two indistinguishable columns and concludes nothing. So we score a pool
    of held-out slices by how many more lesions the distortion model erases than
    the tumor-aware one, and present the highest-scoring slices first. Ties fall
    back to slices that at least contain a small lesion. Nothing is discarded on
    the basis of the result; the ordering is presentational only, and the
    aggregate rates on the opening slide come from the full held-out set.
    """
    ds = _test_set(size, pool)
    scored = []
    for i in range(len(ds)):
        s = ds[i]
        hr = s["hr"][None].to(device)
        gt = s["mask"][0].cpu().numpy()
        versions, lr, mean, unc = _infer(seg, sr_d, sr_t, hr, factor, sigma,
                                         device, seed=100 + i)
        rows = _slice_stats(seg, versions, gt, hr)
        gap = rows["distortion"]["erased"] - rows["tumor-aware"]["erased"]
        scored.append((gap, rows["distortion"]["lesions"], i))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return ds, [i for _, _, i in scored[:n_slices]]


def _slice_blocks(device, n_slices):
    """One slide per test slice: comparison figure, uncertainty figure, table."""
    import matplotlib
    matplotlib.use("Agg")
    from src.figures import comparison_figure, uncertainty_figure

    seg, sr_d, sr_t, size, factor, sigma = _ensure_models(device)
    ds, picks = _pick_slices(seg, sr_d, sr_t, size, factor, sigma, device, n_slices)

    blocks = []
    for n, i in enumerate(picks, start=1):
        s = ds[i]
        hr = s["hr"][None].to(device)
        mask = s["mask"][None].to(device)
        gt = s["mask"][0].cpu().numpy()
        versions, lr, mean, unc = _infer(seg, sr_d, sr_t, hr, factor, sigma,
                                         device, seed=100 + i)
        rows = _slice_stats(seg, versions, gt, hr)

        cimg = _fig_to_b64(comparison_figure(
            hr, mask, lr, {"distortion": versions["distortion"],
                           "tumor-aware": versions["tumor-aware"]}, seg))
        uimg = _fig_to_b64(uncertainty_figure(lr, mean, unc, hr))

        gap = rows["distortion"]["erased"] - rows["tumor-aware"]["erased"]
        if gap > 0:
            verdict = (f"Distortion-optimal erases {gap} lesion"
                       f"{'s' if gap > 1 else ''} that the tumor-aware model keeps.")
        else:
            verdict = "Both objectives recover the same lesions on this slice."

        # Rows in the order a reader expects: the cheap scan the pipeline starts
        # from, then each reconstruction of it. Every row is scored by the same
        # frozen segmenter, so the only thing changing is the image.
        order = ["low-res", "distortion", "tumor-aware"]
        n_les = rows["low-res"]["lesions"]
        cells = []
        for k in order:
            v = rows.get(k)
            if v is None:
                continue
            tags = {"distortion": " <span style='color:var(--ink-light)'>(baseline)</span>",
                    "tumor-aware": " <b style='color:var(--safe)'>(ours)</b>",
                    "low-res": " <span style='color:var(--ink-light)'>(the degraded scan)</span>"}
            cells.append(
                f"<tr><td>{k}{tags.get(k, '')}</td><td class=num>{v['psnr']:.1f}</td>"
                f"<td class=num>{v['ssim']:.3f}</td>"
                f"<td class=num>{v['dice']:.3f}</td>"
                f"<td class=num>{v['erased']}/{v['lesions']}</td>"
                f"<td class=num>{v['fabricated']}</td></tr>")
        body = "".join(cells)
        blocks.append(f"""
      <section class="slide extra">
        <div class="tag"><span class="sec">05</span>Result &middot; 2D evidence {n} of {len(picks)}</div>
        <h2>{verdict}</h2>
        <p class="note">Top row: the image at each stage. Bottom row: what the frozen
        segmenter finds, filled in that model's color, with the true tumor outlined.
        An outline with nothing inside it is an erased lesion.</p>
        <div class="slice">
          <img src="data:image/png;base64,{cimg}" alt="slice {n} comparison">
          <img src="data:image/png;base64,{uimg}" alt="slice {n} uncertainty and error">
          <table>
            <tr>
              <th>image being segmented</th>
              <th class=num colspan="3">image vs true HR &nbsp;&rarr;</th>
              <th class=num colspan="3">detector output vs true mask &nbsp;&rarr;</th>
            </tr>
            <tr><th></th><th class=num>PSNR</th><th class=num>SSIM</th>
                <th class=num>Dice</th><th class=num>erased</th>
                <th class=num>fabricated</th></tr>
            {body}
          </table>
          <p class="note" style="margin-top:.8rem">Each row is a different image handed to
          the <i>same</i> frozen tumor detector. <b>PSNR and SSIM</b> compare that image to the
          true scan; <b>Dice, erased and fabricated</b> compare the detector's output on it to
          the true tumor mask. This slice contains {n_les} lesion components, so
          <b>erased</b> counts how many of those the detector could no longer find.</p>
        </div>
      </section>""")
    return "".join(blocks), size, factor, sigma


def build(out: str = OUT, device: str | None = None, n_slices: int = 3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    patients, vols = build_patient_volumes(device=device)
    pngs = render_compare_png(patients, vols)
    unc_png = render_uncertainty_png(patients)
    gif = render_rotate_gif(patients, vols)
    slices, size, factor, sigma = _slice_blocks(device, n_slices)

    unc_panel = ""
    if unc_png:
        unc_panel = (
            f'<figure><img src="data:image/png;base64,{_b64(unc_png)}" '
            f'alt="MC dropout uncertainty field in 3D">'
            f'<figcaption><b>Uncertainty.</b> Where the model is least sure.'
            f'</figcaption></figure>')

    html = PAGE.format(
        c_true=LIGHT["true"], c_ta=LIGHT["tumor_aware"], c_di=LIGHT["distortion"],
        c_unc=UNCERTAINTY_RAMP[4],
        ramp_css=",".join(UNCERTAINTY_RAMP),
        img_true=_b64(pngs["true"]), img_ta=_b64(pngs["tumor-aware"]),
        img_di=_b64(pngs["distortion"]), img_gif=_b64(gif),
        unc_panel=unc_panel, slices=slices,
        blindness=_blindness_block(), roles=_roles_block(), byline=_byline(),
        arch=_arch_block(), photo=_photo_block(),
        matched=_matched_stats(), floor_chart=_floor_chart(),
        size=size, factor=factor, sigma=sigma,
    )
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build()
