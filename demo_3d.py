"""3D demo page: renders this project's SR output as an interactive-looking
brain/tumor scene (via the vendored neuro-voxel analyzer) and writes a
self-contained ``demo_3d.html`` for judges.

    python demo_3d.py            # bridge -> render -> demo_3d.html

The page embeds the three 3D panels (ground truth / tumor-aware / distortion)
and a rotating GIF as base64, so the single HTML file is shareable with no
assets alongside it -- same philosophy as demo.html.
"""

from __future__ import annotations

import base64

import torch

from render_3d import render_compare_png, render_rotate_gif
from viz_bridge import build_patient_volumes

OUT = "demo_3d.html"


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime(path: str) -> str:
    return "image/gif" if path.endswith(".gif") else "image/png"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tumor-Aware SR: 3D Safety Readout</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --ink:#1A1A1A; --ink-mid:#444; --ink-light:#6A6E73;
    --bg:#DEDFDD; --card:#F1F1F0; --card-2:#E4E5E3; --border:#E2E3E1;
    --navy:#16213E; --accent:#3B82F6; --accent-deep:#2A4A7F;
    --good:#2E7D5B; --warn:#C0862B; --low:#B4553B;
    --font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
    --mono:'DM Mono',ui-monospace,'Courier New',monospace;
  }}
  *{{box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{margin:0; background:var(--bg); color:var(--ink); font-family:var(--font);
    font-weight:400; line-height:1.55; -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1080px; margin:0 auto; padding:0 clamp(1rem,4vw,2rem) 4rem}}

  .tag{{font-family:var(--mono); font-size:.7rem; letter-spacing:.24em; text-transform:uppercase;
    color:var(--accent)}}

  /* navy hero band -- echoes the RISE dashboard sidebar */
  .hero{{background:var(--navy); color:#fff; border-radius:0 0 20px 20px;
    position:relative; overflow:hidden}}
  .hero-inner{{max-width:1080px; margin:0 auto; position:relative; z-index:2;
    padding:clamp(2.4rem,5vw,3.8rem) clamp(1rem,4vw,2rem) clamp(3.8rem,6vw,5.2rem)}}
  .hero::after{{content:""; position:absolute; right:-6%; top:-40%; width:460px; height:460px;
    background:radial-gradient(circle, rgba(59,130,246,.28), transparent 62%); z-index:1}}
  .hero .tag{{color:#8fb4ff}}
  .hero h1{{font-weight:700; font-size:clamp(2.2rem,6vw,4rem); line-height:1.02;
    letter-spacing:-.02em; margin:.5rem 0 .4rem; max-width:20ch}}
  .hero h1 em{{font-style:normal; color:#7fa8ff}}
  .lede{{color:rgba(255,255,255,.72); max-width:62ch; font-size:1.02rem; font-weight:300}}
  .lede b{{color:#fff; font-weight:600}}

  /* vitals readout -- floats up over the hero */
  .vitals{{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;
    margin:-2.6rem 0 1.4rem; position:relative; z-index:3}}
  .vital{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.3rem 1.3rem; box-shadow:0 12px 30px -18px rgba(22,33,62,.5)}}
  .vital .k{{font-family:var(--mono); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
    color:var(--ink-light); display:flex; align-items:center}}
  .vital .v{{font-family:var(--mono); font-weight:500; font-size:clamp(2rem,5vw,2.9rem);
    line-height:1; margin-top:.5rem; color:var(--ink)}}
  .vital .u{{font-size:.85rem; color:var(--ink-light); margin-left:.2rem}}
  .vital .d{{font-size:.8rem; margin-top:.55rem; color:var(--ink-mid)}}
  .vital.true .v{{color:var(--accent-deep)}} .vital.safe .v{{color:var(--good)}}
  .vital.erased .v{{color:var(--low)}}
  .dot{{display:inline-block; width:.5em; height:.5em; border-radius:50%; margin-right:.5em}}

  section{{margin-top:2.6rem}}
  .note{{max-width:70ch; font-size:.86rem; color:var(--ink-light); margin:1.1rem 0 0}}
  h2{{font-weight:700; font-size:clamp(1.5rem,3.2vw,2.1rem); letter-spacing:-.01em; margin:.2rem 0 .5rem}}

  /* 3D viewport panels */
  .panels{{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem}}
  figure{{margin:0; border:1px solid var(--border); border-radius:14px; overflow:hidden;
    background:var(--card)}}
  figure img{{display:block; width:100%; height:auto; background:var(--navy)}}
  figcaption{{font-size:.76rem; color:var(--ink-mid); padding:.7rem .9rem;
    border-top:1px solid var(--border)}}
  figcaption b{{color:var(--ink); font-weight:600}}

  .rotate{{display:grid; grid-template-columns:1.05fr .95fr; gap:2rem; align-items:center}}
  .rotate .view{{border:1px solid var(--border); border-radius:16px; overflow:hidden; background:var(--navy)}}
  .rotate .view img{{width:100%; display:block}}
  .legend{{display:flex; gap:1.3rem; flex-wrap:wrap; font-family:var(--mono); font-size:.78rem;
    color:var(--ink-mid); margin-top:1rem}}

  .steps{{display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-top:1rem}}
  .step{{background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem}}
  .step .n{{font-family:var(--mono); font-weight:500; font-size:1.3rem; color:var(--accent)}}
  .step p{{font-size:.82rem; color:var(--ink-mid); margin:.4rem 0 0}}

  footer{{margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border);
    font-size:.78rem; color:var(--ink-light)}}
  footer a{{color:var(--accent-deep); text-decoration:none; font-weight:500}}
  code{{font-family:var(--mono); background:var(--card-2); padding:1px 6px; border-radius:5px; font-size:.85em}}

  .rise{{opacity:0; transform:translateY(14px); animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards}}
  @keyframes rise{{to{{opacity:1; transform:none}}}}
  {stagger}
  @media(max-width:760px){{.vitals,.panels,.steps{{grid-template-columns:1fr}} .rotate{{grid-template-columns:1fr}}}}
  @media(prefers-reduced-motion:reduce){{.rise{{animation:none;opacity:1;transform:none}}}}
</style></head>
<body>

  <header class="hero">
   <div class="hero-inner">
    <div class="tag rise" style="--i:0">ACVSS &middot; MRI Super-Resolution Safety</div>
    <h1 class="rise" style="--i:1">When sharper means <em>blind</em>.</h1>
    <p class="lede rise" style="--i:2">Super-resolution makes a cheap low-field brain scan look crisp.
    When it is trained only for image quality, it can quietly <b>erase the tumor</b>.
    The panels below show our own pipeline output, rebuilt in 3D with the
    <b>neuro-voxel</b> volume renderer. The brain and the segmenter stay fixed.
    Only the super-resolution objective changes.</p>
   </div>
  </header>

  <div class="wrap">

  <section class="vitals rise" style="--i:3; margin-top:-2.6rem" aria-label="tumor volume readout">
    <div class="vital true"><div class="k"><span class="dot" style="background:var(--accent)"></span>Ground truth</div>
      <div class="v">{true_v}<span class="u">cm&sup3;</span></div>
      <div class="d">the tumor actually present</div></div>
    <div class="vital safe"><div class="k"><span class="dot" style="background:var(--good)"></span>Tumor-aware SR</div>
      <div class="v">{ta_v}<span class="u">cm&sup3;</span></div>
      <div class="d" style="color:var(--good)">{ta_pct}% preserved &middot; lesion kept</div></div>
    <div class="vital erased"><div class="k"><span class="dot" style="background:var(--low)"></span>Distortion-optimal SR</div>
      <div class="v">{di_v}<span class="u">cm&sup3;</span></div>
      <div class="d" style="color:var(--low)">{di_pct}% preserved &middot; lesion erased</div></div>
  </section>

  <p class="note rise" style="--i:4">Ground truth is the true high-resolution scan (true HR).
  We degrade it to imitate a cheap low-field scanner, then measure how much tumor each
  model recovers from that degraded input.</p>

  <section class="panels rise" style="--i:4">
    <figure><img src="data:image/png;base64,{img_true}" alt="ground truth tumor in 3D">
      <figcaption><b>Ground truth.</b> Four lesions, {true_v} cm&sup3;.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_ta}" alt="tumor-aware reconstruction in 3D">
      <figcaption><b>Tumor-aware SR.</b> The lesion survives at {ta_v} cm&sup3;.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_di}" alt="distortion reconstruction in 3D">
      <figcaption><b>Distortion-optimal SR.</b> The lesion is gone at {di_v} cm&sup3;.</figcaption></figure>
  </section>

  <section class="rotate rise" style="--i:5">
    <div class="view"><img src="data:image/gif;base64,{img_gif}" alt="rotating 3D brain with tumor overlays"></div>
    <div>
      <h2>One brain, three verdicts.</h2>
      <p style="color:var(--ink-mid)">The orbit overlays all three results. It shows the
      true tumor shell, the volume the tumor-aware model keeps, and the empty space the
      distortion model leaves. At matched image quality, <b>image quality is not a safety
      metric</b>.</p>
      <div class="legend">
        <span><span class="dot" style="background:var(--accent)"></span>true</span>
        <span><span class="dot" style="background:var(--good)"></span>tumor-aware</span>
        <span><span class="dot" style="background:var(--low)"></span>distortion</span>
      </div>
    </div>
  </section>

  <section class="rise" style="--i:6">
    <div class="tag">How the 3D is built</div>
    <h2>From 2D model to 3D truth.</h2>
    <div class="steps">
      <div class="step"><div class="n">01</div><p>A coherent 3D brain phantom with tumors of varied size.</p></div>
      <div class="step"><div class="n">02</div><p>Each axial slice is degraded, super-resolved by both models, then segmented.</p></div>
      <div class="step"><div class="n">03</div><p>Predicted masks are restacked into 3D volumes.</p></div>
      <div class="step"><div class="n">04</div><p>Marching cubes and eye-dome lighting render the meshes.</p></div>
    </div>
  </section>

  <footer>
    3D rendering by the vendored <a href="https://github.com/asmarufoglu/neuro-voxel">neuro-voxel</a>
    analyzer (PyVista and VTK marching cubes). The volumes shown are our own model output on a
    synthetic phantom, which makes this a proof of concept. Regenerate with <code>python demo_3d.py</code>.
    See <code>README.md</code> for the full safety study.
  </footer>

  </div>
</body></html>"""


def build(out: str = OUT, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patients, vols = build_patient_volumes(device=device)
    pngs = render_compare_png(patients, vols)
    gif = render_rotate_gif(patients, vols)

    true_v = vols["true"]
    stagger = "".join(f".rise[style*='--i:{i}']{{animation-delay:{i*0.09:.2f}s}}"
                      for i in range(7))
    html = PAGE.format(
        stagger=stagger,
        true_v=f"{vols['true']:.2f}", ta_v=f"{vols['tumor-aware']:.2f}",
        di_v=f"{vols['distortion']:.2f}",
        ta_pct=f"{100*vols['tumor-aware']/true_v:.0f}" if true_v else "0",
        di_pct=f"{100*vols['distortion']/true_v:.0f}" if true_v else "0",
        img_true=_b64(pngs["true"]), img_ta=_b64(pngs["tumor-aware"]),
        img_di=_b64(pngs["distortion"]), img_gif=_b64(gif),
    )
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build()
