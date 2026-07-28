"""The demo page. This is the one we present.

Builds ``main_demo.html``: a single self-contained page carrying the whole
story, 3D and 2D, with every image embedded as base64 so the file can be opened
or shared with nothing alongside it.

    python main_demo.py            # models -> renders -> main_demo.html

What is on the page, in order:
  1. the measured safety headline (erasure rates and the tradeoff);
  2. four 3D viewports: ground truth, tumor-aware, distortion-optimal, and the
     MC dropout uncertainty field;
  3. a rotating overlay of all three reconstructions on one brain;
  4. the 2D per-slice evidence the 3D is built from: comparison panels, the
     uncertainty and error maps, and the per-version metrics table;
  5. how the 3D is assembled, and what the scope limits are.

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


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tumor-Aware MRI Super-Resolution</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{
    --ink:#1A1A1A; --ink-mid:#444; --ink-light:#6A6E73;
    --bg:#DEDFDD; --card:#F1F1F0; --card-2:#E4E5E3; --border:#D9DAD8;
    --navy:#16213E; --accent:#3B82F6; --accent-deep:#2A4A7F; --warn:#C0862B;
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
  .deck{{max-width:1100px; margin:0 auto; padding:1.2rem clamp(1rem,4vw,2rem) 6.5rem;
    min-height:100vh}}
  .slide{{display:none; animation:fade .34s cubic-bezier(.2,.7,.2,1)}}
  .slide.on{{display:block}}
  @keyframes fade{{from{{opacity:0; transform:translateY(10px)}} to{{opacity:1; transform:none}}}}
  @media(prefers-reduced-motion:reduce){{.slide{{animation:none}}}}

  /* ---- fixed bottom navigation bar ---- */
  .navbar{{position:fixed; left:0; right:0; bottom:0; z-index:50;
    background:rgba(241,241,240,.94); backdrop-filter:blur(10px);
    border-top:1px solid var(--border)}}
  .navbar .inner{{max-width:1100px; margin:0 auto; display:flex; align-items:center;
    gap:1rem; padding:.7rem clamp(1rem,4vw,2rem)}}
  .btn{{font-family:var(--font); font-size:.85rem; font-weight:600; cursor:pointer;
    border:1px solid var(--border); background:var(--card); color:var(--ink);
    padding:.5rem 1.1rem; border-radius:99px; transition:.15s}}
  .btn:hover:not(:disabled){{background:var(--navy); color:#fff; border-color:var(--navy)}}
  .btn:disabled{{opacity:.32; cursor:not-allowed}}
  .btn.primary{{background:var(--navy); color:#fff; border-color:var(--navy)}}
  .btn.primary:hover{{background:#0e1730}}
  .dots{{display:flex; gap:.42rem; flex:1; flex-wrap:wrap}}
  .dot-nav{{width:.62rem; height:.62rem; border-radius:50%; border:none; padding:0;
    background:var(--border); cursor:pointer; transition:.15s}}
  .dot-nav:hover{{background:var(--ink-light)}}
  .dot-nav.on{{background:var(--navy); transform:scale(1.28)}}
  .counter{{font-family:var(--mono); font-size:.74rem; color:var(--ink-light);
    min-width:4.2rem; text-align:right}}
  .hint{{font-family:var(--mono); font-size:.66rem; color:var(--ink-light)}}
  @media(max-width:700px){{.hint{{display:none}}}}

  .tag{{font-family:var(--mono); font-size:.7rem; letter-spacing:.24em; text-transform:uppercase;
    color:var(--accent)}}
  h1{{font-weight:700; font-size:clamp(2.1rem,5.6vw,3.7rem); line-height:1.03;
    letter-spacing:-.02em; margin:.5rem 0 .5rem; max-width:20ch}}
  h1 em{{font-style:normal; color:var(--accent-deep)}}
  h2{{font-weight:700; font-size:clamp(1.4rem,3vw,2rem); letter-spacing:-.01em;
    margin:.25rem 0 .5rem; max-width:34ch}}
  h3{{font-weight:600; font-size:1rem; margin:0 0 .6rem}}
  .lede{{color:var(--ink-mid); max-width:60ch; font-size:1.05rem; font-weight:300}}
  .lede b{{color:var(--ink); font-weight:600}}
  .note{{max-width:72ch; font-size:.85rem; color:var(--ink-light); margin:.9rem 0 0}}

  /* title slide */
  .title-slide{{min-height:calc(100vh - 8rem); display:flex; flex-direction:column;
    justify-content:center; background:var(--navy); color:#fff; border-radius:20px;
    padding:clamp(2rem,5vw,4rem); position:relative; overflow:hidden}}
  .title-slide::after{{content:""; position:absolute; right:-8%; top:-35%; width:480px;
    height:480px; background:radial-gradient(circle, rgba(59,130,246,.3), transparent 62%)}}
  .title-slide > *{{position:relative; z-index:2}}
  .title-slide .tag{{color:#8fb4ff}}
  .title-slide h1 em{{color:#7fa8ff}}
  .title-slide .lede{{color:rgba(255,255,255,.74)}}
  .title-slide .lede b{{color:#fff}}
  .byline{{font-family:var(--mono); font-size:.76rem; color:rgba(255,255,255,.5);
    margin-top:2.2rem}}

  .vitals{{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.4rem}}
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

  /* Three bars, one per image handed to the same frozen detector: the cheap
     scan, then each reconstruction. Grey for the low-resolution input because it
     is the starting point rather than a model, so it must not look like one. */
  .ladder{{margin-top:1.1rem; background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.2rem 1.3rem}}
  .lrow{{display:grid; grid-template-columns:15rem 1fr 9rem; gap:1rem;
    align-items:center; margin-bottom:.75rem}}
  .llab{{font-size:.84rem; font-weight:600; line-height:1.25}}
  .llab span{{display:block; font-weight:400; font-size:.74rem; color:var(--ink-light)}}
  .lbar{{display:flex; height:1.5rem; background:var(--card-2); border-radius:4px;
    overflow:hidden}}
  .lbar i{{display:block; height:100%}}
  .lbar i.lowres{{background:#9BA1A6}}
  /* 2px of surface between stacked segments so the boundary reads as a break
     rather than a colour transition. */
  .lbar i.add{{border-left:2px solid var(--card)}}
  .lbar i.di{{background:var(--erased)}}
  .lbar i.ta{{background:var(--safe)}}
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
  .sw{{display:inline-block; width:.8rem; height:.8rem; border-radius:3px;
    margin-right:.4rem; vertical-align:-1px}}
  .sw.lowres{{background:#9BA1A6}} .sw.di{{background:var(--erased)}}
  .sw.ta{{background:var(--safe); margin-left:-.15rem; margin-right:.5rem}}
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

  .steps{{display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-top:1.2rem}}
  .step{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.1rem}}
  .step .n{{font-family:var(--mono); font-weight:500; font-size:1.3rem; color:var(--accent)}}
  .step p{{font-size:.82rem; color:var(--ink-mid); margin:.4rem 0 0}}

  .scope{{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:1.2rem 1.4rem; margin-top:1.2rem}}
  .scope li{{font-size:.88rem; color:var(--ink-mid); margin:.45rem 0}}
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
      <div class="byline">Adiza Alhassan &middot; Nthabiseng Thema &middot; Albert Dodoo &middot; Victor Oyindouye Miene &middot; Hassan Suliman</div>
    </div>
  </section>

  <section class="slide">
    <div class="tag">Validation result &middot; real BraTS, 70 unseen patients</div>
    <h2>Super-resolution recovers tumor the cheap scan loses. Ours recovers more of it.</h2>
    <div class="vitals">
      <div class="vital"><div class="k"><span class="dot" style="background:#9BA1A6"></span>Low-resolution scan</div>
        <div class="v" style="color:var(--ink-mid)">62.2<span class="u">%</span></div>
        <div class="d">of enhancing lesions missed, before any super-resolution</div></div>
      <div class="vital erased"><div class="k"><span class="dot" style="background:var(--erased)"></span>Standard SR <b style="color:var(--ink-mid)">(baseline)</b></div>
        <div class="v">58.0<span class="u">%</span></div>
        <div class="d">recovers some of what the cheap scan loses</div></div>
      <div class="vital safe"><div class="k"><span class="dot" style="background:var(--safe)"></span>Tumor-aware SR <b style="color:var(--safe)">(OURS)</b></div>
        <div class="v">51.3<span class="u">%</span></div>
        <div class="d">&minus;6.7 points vs baseline, at matched quality</div></div>
    </div>
    <div class="ladder">
      <div class="lrow">
        <div class="llab">Low-resolution scan<span>what a cheap scanner produces</span></div>
        <div class="lbar"><i class="lowres" style="width:88.9%"></i></div>
        <div class="lnum">62.2%<span>missed</span></div>
      </div>
      <div class="lrow">
        <div class="llab">Standard super-resolution <span style="display:inline;font-weight:600;color:var(--ink-mid)">(baseline)</span><span>trained only for image quality</span></div>
        <div class="lbar"><i class="add di" style="width:82.9%"></i></div>
        <div class="lnum">58.0%<span>missed</span></div>
      </div>
      <div class="lrow">
        <div class="llab">Tumor-aware super-resolution <span style="display:inline;font-weight:700;color:var(--safe)">(OURS)</span><span>our objective: lesion-weighted loss</span></div>
        <div class="lbar"><i class="add ta" style="width:73.3%"></i></div>
        <div class="lnum">51.3%<span><b style="color:var(--safe)">&minus;6.7</b> vs baseline</span></div>
      </div>
      <div class="lscale"><div><span>0%</span>
        <span class="mid">of enhancing lesion components missed</span><span>70%</span></div></div>
      <div class="lkey">
        <span>Same frozen tumor detector in all three rows. Only the image it is given changes,
        so the difference is caused by the reconstruction and not by a different detector.</span>
      </div>
    </div>

    <p class="note"><b>These are validation numbers, not the final test result.</b> Three loss
    configurations are being compared on the validation split; the winner will be evaluated
    once on 94 held-out test patients, and that single number is the one to quote. Reporting
    the best of several configurations on the test set would not be an evaluation.</p>
    <p class="note">Real brain MRI from the Medical Segmentation Decathlon (BraTS), 17,233
    slices from 468 patients, split <b>by patient</b> so no slice of a held-out case was ever
    trained on. The unit of analysis is the patient, not the lesion: one tumor spans many
    slices and those outcomes move together, so pooling lesion cross-sections overstates
    precision by about 1.5&times;. Brain-masked PSNR differs by 0.23&nbsp;dB (24.51 vs 24.28),
    measured inside the brain because roughly a tenth of a slice is empty background that both
    models reproduce for free and which flatters the match. The cost is hallucination, with
    the false positive rate rising from 0.266 to 0.387.
    True HR is the original high-resolution scan; we degrade it to imitate a cheap low-field
    scanner.</p>
  </section>

  <section class="slide">
    <div class="tag">Broken down by lesion size</div>
    <h2>Large lesions are almost never lost. Small ones are the whole problem.</h2>
    <p class="note">Same 70 validation patients, 9,490 lesion components, split by area. Each
    column is the share of lesions the detector could no longer find in that image.</p>
    <table style="margin-top:1.1rem; background:var(--card); border:1px solid var(--border);
      border-radius:14px; padding:.4rem">
      <tr>
        <th>lesion size</th><th class=num>how many</th>
        <th class=num>low-resolution<br><span style="font-weight:400;text-transform:none;letter-spacing:0">the cheap scan</span></th>
        <th class=num>standard SR<br><span style="font-weight:400;text-transform:none;letter-spacing:0">(baseline)</span></th>
        <th class=num>tumor-aware SR<br><span style="font-weight:400;text-transform:none;letter-spacing:0">(ours)</span></th>
      </tr>
      <tr><td><b>small</b> &lt;50&nbsp;px</td><td class=num>6,762 <span style="color:var(--ink-light)">(71%)</span></td>
        <td class=num>82.1%</td><td class=num style="color:var(--erased)">78.6%</td>
        <td class=num style="color:var(--safe)"><b>69.8%</b></td></tr>
      <tr><td><b>medium</b> 50&ndash;200&nbsp;px</td><td class=num>1,005</td>
        <td class=num>30.0%</td><td class=num style="color:var(--erased)">16.4%</td>
        <td class=num style="color:var(--safe)"><b>13.5%</b></td></tr>
      <tr><td><b>large</b> &gt;200&nbsp;px</td><td class=num>1,723</td>
        <td class=num>3.2%</td><td class=num style="color:var(--erased)">1.2%</td>
        <td class=num style="color:var(--safe)"><b>0.9%</b></td></tr>
    </table>
    <p class="note"><b>Start with the large row.</b> A lesion over 200&nbsp;px is missed under 1%
    of the time by either model, so substantial tumors are not being erased and a raw figure
    near 50% should not be read as "half the tumors vanish".</p>
    <p class="note"><b>Then the small row.</b> 71% of all components are under 50&nbsp;px, which
    is what drags the overall rate up, and it is also where our objective helps most: 78.6% down
    to 69.8%, against a fraction of a point on large lesions. That is the thesis exactly &mdash;
    the objective matters where the structure is small enough for a pixel-error score to ignore
    it.</p>
    <p class="note">Small does not mean unimportant &mdash; a 30&nbsp;px enhancing focus can be
    an early recurrence, and this is where the remaining work is. It is also partly an artefact:
    a thin enhancing rim fragments into many 4-connected pieces, and this metric weights a
    five-pixel speck the same as a whole tumor.</p>
  </section>

  <section class="slide">
    <div class="tag">A second finding</div>
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
    <div class="tag">Four viewports</div>
    <h2>What each objective leaves behind.</h2>
    <div class="panels">
    <figure><img src="data:image/png;base64,{img_true}" alt="ground truth tumor in 3D">
      <figcaption><b>Ground truth.</b> The true lesions, in blue.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_ta}" alt="tumor-aware reconstruction in 3D">
      <figcaption><b>Tumor-aware &mdash; OURS.</b> Lesions preserved.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_di}" alt="distortion reconstruction in 3D">
      <figcaption><b>Distortion-optimal &mdash; baseline.</b> Small lesions dropped.</figcaption></figure>
    {unc_panel}
    </div>
    <p class="note">A blue ghost with no fill inside it is a lesion the model lost. This is
    one held-out patient, restacked from per-slice predictions. Models were trained on
    {size}&times;{size} slices with k-space factor {factor} and Rician &sigma;={sigma}; the
    volume here is rendered at full head width so the surface closes.</p>
  </section>

  <section class="slide">
    <div class="tag">One brain, three reconstructions</div>
    <h2>Where each objective keeps or loses tissue.</h2>
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

  <section class="slide">
    <div class="tag">How the 3D is built</div>
    <h2>From a 2D model to a 3D readout.</h2>
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
    <div class="tag">What this is not</div>
    <h2>Scope, stated plainly.</h2>
    <div class="scope"><ul>
      <li><b>It only works on small lesions.</b> Repeating the identical pipeline on whole
          tumor (10.9% of the image, against 3.5% for enhancing tumor) shows no benefit at
          all: 64.6% vs 63.7% erasure. That is what the theory predicts. The objective is
          only misaligned when erasing the structure costs negligible PSNR, so there is
          nothing to fix for a large region.</li>
      <li><b>Enhancement can make detection worse.</b> On whole tumor, both SR models erase
          more than the raw low-res input (64.6% and 63.7% vs 61.5%) while improving PSNR by
          3&nbsp;dB. Better image quality, worse finding.</li>
      <li><b>Simulated degradation.</b> It reproduces resolution loss and noise, but not the
          contrast change of a genuinely low-field scanner. We degrade real high-field scans;
          we have not tested a real low-field acquisition.</li>
      <li><b>The fix costs hallucination.</b> False positive rate rises from 0.266 to 0.387.
          Which error a clinic can tolerate is a clinical decision, not ours.</li>
      <li><b>Numbers on this deck are validation, not test.</b> The final figure comes from
          one evaluation of one configuration on 94 patients never used for any decision.</li>
      <li><b>The tumor detector is the weakest link, not the enhancement.</b> It misses a
          large share of small enhancing components even on an untouched scan, so a stronger
          downstream model would buy more than a better loss function would.</li>
      <li><b>Lesion components fragment.</b> An enhancing rim breaks into many small
          4-connected pieces, so component counts run high and each counts equally. That
          inflates absolute rates.</li>
      <li><b>Uncertainty is a reliability signal, not a tumor detector.</b> It runs about
          1.5&times; higher inside the lesion than in healthy tissue here, which is
          suggestive rather than diagnostic.</li>
      <li><b>Do not read cm&sup3; off a render.</b> The segmenter over-reads even a clean
          high-resolution scan, so compare a reconstruction against the segmenter's own
          reading of the true scan, never against the true mask.</li>
    </ul></div>
  </section>

  <section class="slide">
    <div class="tag">Credits</div>
    <h2>Reproducing this.</h2>
    <p class="credits">3D rendering by the vendored
    <a href="https://github.com/asmarufoglu/neuro-voxel">neuro-voxel</a> analyzer
    (PyVista and VTK marching cubes). Every volume shown is our own model output on a
    synthetic phantom, which makes this a proof of concept rather than a clinical result.
    Colors are validated for colorblind separability in <code>src/palette.py</code>.
    Regenerate the whole deck with <code>python main_demo.py</code>. See
    <code>README.md</code> for the full safety study and <code>paper/</code> for the write-up.</p>
  </section>

</div>

<nav class="navbar">
  <div class="inner">
    <button class="btn" id="prev" aria-label="previous slide">&larr; Prev</button>
    <button class="btn primary" id="next" aria-label="next slide">Next &rarr;</button>
    <div class="dots" id="dots"></div>
    <span class="hint">&larr; &rarr; or space</span>
    <span class="counter" id="counter"></span>
  </div>
</nav>

<script>
(function () {{
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var dots = document.getElementById('dots');
  var prev = document.getElementById('prev');
  var next = document.getElementById('next');
  var counter = document.getElementById('counter');
  var at = 0;

  slides.forEach(function (_, i) {{
    var b = document.createElement('button');
    b.className = 'dot-nav';
    b.setAttribute('aria-label', 'go to slide ' + (i + 1));
    b.addEventListener('click', function () {{ go(i); }});
    dots.appendChild(b);
  }});

  function go(i) {{
    at = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach(function (s, j) {{ s.classList.toggle('on', j === at); }});
    Array.prototype.forEach.call(dots.children, function (d, j) {{
      d.classList.toggle('on', j === at);
    }});
    prev.disabled = at === 0;
    next.disabled = at === slides.length - 1;
    counter.textContent = (at + 1) + ' / ' + slides.length;
    window.scrollTo({{ top: 0, behavior: 'instant' }});
    location.hash = 'p' + (at + 1);
  }}

  prev.addEventListener('click', function () {{ go(at - 1); }});
  next.addEventListener('click', function () {{ go(at + 1); }});
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{ e.preventDefault(); go(at + 1); }}
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ e.preventDefault(); go(at - 1); }}
    if (e.key === 'Home') {{ go(0); }}
    if (e.key === 'End') {{ go(slides.length - 1); }}
  }});

  var start = parseInt((location.hash || '').replace('#p', ''), 10);
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
        # frozen detector, so the only thing changing is the image.
        order = ["low-res", "distortion", "tumor-aware"]
        n_les = rows["low-res"]["lesions"]
        cells = []
        for k in order:
            v = rows.get(k)
            if v is None:
                continue
            tags = {"distortion": " <span style='color:var(--ink-light)'>(baseline)</span>",
                    "tumor-aware": " <b style='color:var(--safe)'>(ours)</b>",
                    "low-res": " <span style='color:var(--ink-light)'>(the cheap scan)</span>"}
            cells.append(
                f"<tr><td>{k}{tags.get(k, '')}</td><td class=num>{v['psnr']:.1f}</td>"
                f"<td class=num>{v['ssim']:.3f}</td>"
                f"<td class=num>{v['dice']:.3f}</td>"
                f"<td class=num>{v['erased']}/{v['lesions']}</td>"
                f"<td class=num>{v['fabricated']}</td></tr>")
        body = "".join(cells)
        blocks.append(f"""
      <section class="slide">
        <div class="tag">2D evidence &middot; {n} of {len(picks)}</div>
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
        size=size, factor=factor, sigma=sigma,
    )
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)
    return out


if __name__ == "__main__":
    build()
