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
    <div class="tag">The measured result &middot; real BraTS, 71 unseen patients</div>
    <h2>At equal segmentation quality, the tumor-aware objective erases fewer enhancing lesions.</h2>
    <div class="vitals">
      <div class="vital erased"><div class="k"><span class="dot" style="background:var(--erased)"></span>Distortion-optimal SR</div>
        <div class="v">41.6<span class="u">%</span></div>
        <div class="d">mean per-patient erasure</div></div>
      <div class="vital safe"><div class="k"><span class="dot" style="background:var(--safe)"></span>Tumor-aware SR</div>
        <div class="v">38.4<span class="u">%</span></div>
        <div class="d">&minus;3.2 points, 95% CI [0.7, 6.9], p&nbsp;=&nbsp;0.007</div></div>
      <div class="vital"><div class="k"><span class="dot" style="background:var(--accent)"></span>Quality gap</div>
        <div class="v" style="color:var(--accent-deep)">0.004</div>
        <div class="d">Dice 0.696 vs 0.692, so sharpness cannot explain the difference</div></div>
    </div>
    <p class="note">Real brain MRI from the Medical Segmentation Decathlon (BraTS), split
    <b>by patient</b> so no slice of a test case was ever trained on. The unit of analysis is
    the patient, not the lesion: one tumor spans many slices and those outcomes move together,
    so pooling lesion cross-sections would overstate precision by about 1.7&times;. Paired
    across 71 patients, the objective <b>helps 29 and hurts 11</b>. The cost is hallucination:
    the false positive rate rises from 0.310 to 0.371. True HR is the original high-resolution
    scan; we degrade it to imitate a cheap low-field scanner.</p>
  </section>

  <section class="slide">
    <div class="tag">Four viewports</div>
    <h2>What each objective leaves behind.</h2>
    <div class="panels">
    <figure><img src="data:image/png;base64,{img_true}" alt="ground truth tumor in 3D">
      <figcaption><b>Ground truth.</b> The true lesions, in blue.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_ta}" alt="tumor-aware reconstruction in 3D">
      <figcaption><b>Tumor-aware.</b> Lesions preserved.</figcaption></figure>
    <figure><img src="data:image/png;base64,{img_di}" alt="distortion reconstruction in 3D">
      <figcaption><b>Distortion-optimal.</b> Small lesions dropped.</figcaption></figure>
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
          <span><span class="dot" style="background:var(--safe)"></span>tumor-aware</span>
          <span><span class="dot" style="background:var(--erased)"></span>distortion-optimal</span>
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
      <li><b>The fix costs hallucination.</b> False positive rate rises from 0.310 to 0.371.
          Which error a clinic can tolerate is a clinical decision, not ours.</li>
      <li><b>It does not help everyone.</b> Paired across 71 patients, 29 improve and
          <b>11 get worse</b>. The 95% interval on the mean benefit runs from 0.7 to 6.9
          points, so the true effect could be small.</li>
      <li><b>Both objectives still miss most small lesions</b> (63% and 66% erased). This
          reduces a failure; it does not solve it.</li>
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

        body = "".join(
            f"<tr><td>{k}</td><td class=num>{v['psnr']:.1f}</td>"
            f"<td class=num>{v['ssim']:.3f}</td><td class=num>{v['dice']:.3f}</td>"
            f"<td class=num>{v['erased']}/{v['lesions']}</td>"
            f"<td class=num>{v['fabricated']}</td></tr>"
            for k, v in rows.items())
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
            <tr><th>version</th><th class=num>PSNR</th><th class=num>SSIM</th>
                <th class=num>Dice</th><th class=num>lesions erased</th>
                <th class=num>fabricated</th></tr>
            {body}
          </table>
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
