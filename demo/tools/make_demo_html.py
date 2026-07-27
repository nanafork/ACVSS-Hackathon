import json
import os

ROOT = os.environ.get(
    "TRUSTMRI_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

P = json.load(open(os.path.join(ROOT, "demo_payload.json")))
DATA = json.dumps(P)

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image quality is not a safety metric</title>
<style>
  :root{
    --bg:#0d1117; --card:#161b22; --line:#272e38;
    --ink:#e6edf3; --mut:#8b949e; --dim:#6e7681;
    --good:#3fb950; --bad:#f85149; --accent:#58a6ff; --warn:#d29922;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
  h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
  .sub{color:var(--mut);margin:0 0 28px;font-size:15px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
  .sliderbox{margin-bottom:24px}
  .sliderbox label{display:flex;justify-content:space-between;align-items:baseline;
    font-size:13px;color:var(--mut);margin-bottom:10px}
  .sliderbox b{color:var(--ink);font-size:15px;font-variant-numeric:tabular-nums}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .ticks{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;margin-top:4px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
  .col h2{font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 4px}
  .col .tag{color:var(--mut);font-size:12px;margin:0 0 12px;min-height:32px}
  .col img{width:100%;image-rendering:pixelated;border-radius:8px;display:block;
    border:1px solid var(--line);background:#000}
  .row{display:flex;justify-content:space-between;padding:7px 0;
    border-bottom:1px solid var(--line);font-size:13px}
  .row:last-child{border-bottom:0}
  .row span{color:var(--mut)}
  .row b{font-variant-numeric:tabular-nums}
  .verdict{margin-top:12px;padding:11px 13px;border-radius:8px;font-size:13px;font-weight:600}
  .v-safe{background:rgba(63,185,80,.12);color:var(--good);border:1px solid rgba(63,185,80,.3)}
  .v-bad{background:rgba(248,81,73,.12);color:var(--bad);border:1px solid rgba(248,81,73,.32)}
  .meter{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:6px}
  .meter i{display:block;height:100%;background:var(--accent);transition:width .18s}
  .legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin-top:14px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
    vertical-align:middle}
  .note{margin-top:26px;padding:16px 18px;border-left:3px solid var(--warn);
    background:rgba(210,153,34,.07);color:var(--mut);font-size:13px;border-radius:0 8px 8px 0}
  .note b{color:var(--ink)}
  .foot{margin-top:22px;color:var(--dim);font-size:12px}
  .kicker{display:inline-block;background:rgba(88,166,255,.12);color:var(--accent);
    border:1px solid rgba(88,166,255,.3);border-radius:20px;padding:3px 11px;
    font-size:11px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px}
</style></head><body><div class="wrap">

<div class="kicker">TrustMRI &middot; tumor-aware super-resolution</div>
<h1>The scan gets sharper. The score goes up. The tumor is gone.</h1>
<p class="sub">Drag the slider to degrade the acquisition, as a cheaper or faster
scanner would. Both models then super-resolve it. Watch the image-quality
numbers &mdash; and watch the lesions.</p>

<div class="panel sliderbox">
  <label><span>Acquisition degradation &mdash; k-space truncation factor</span>
    <b id="flabel">&times;4</b></label>
  <input type="range" id="slider" min="0" max="5" value="2" step="1">
  <div class="ticks" id="ticks"></div>
</div>

<div class="grid">
  <div class="panel col">
    <h2>What was acquired</h2>
    <p class="tag">The low-field scan. Blurry, noisy, honest.</p>
    <img id="img-lr">
  </div>
  <div class="panel col">
    <h2 style="color:var(--bad)">Distortion-optimal</h2>
    <p class="tag">Standard training: plain pixel L1, the objective the field
      optimises.</p>
    <img id="img-d">
    <div class="row"><span>PSNR</span><b id="d-psnr"></b></div>
    <div class="row"><span>SSIM</span><b id="d-ssim"></b></div>
    <div class="row"><span>Lesions found</span><b id="d-found"></b></div>
    <div class="row"><span>Model confidence</span><b id="d-conf"></b></div>
    <div class="meter"><i id="d-meter"></i></div>
    <div class="verdict" id="d-verdict"></div>
  </div>
  <div class="panel col">
    <h2 style="color:var(--good)">Tumor-aware</h2>
    <p class="tag">Same network, same data. Three lines changed: errors inside
      the lesion cost more.</p>
    <img id="img-t">
    <div class="row"><span>PSNR</span><b id="t-psnr"></b></div>
    <div class="row"><span>SSIM</span><b id="t-ssim"></b></div>
    <div class="row"><span>Lesions found</span><b id="t-found"></b></div>
    <div class="row"><span>Model confidence</span><b id="t-conf"></b></div>
    <div class="meter"><i id="t-meter"></i></div>
    <div class="verdict" id="t-verdict"></div>
  </div>
</div>

<div class="legend">
  <span><i style="background:#5cd9fd"></i>true lesion outline</span>
  <span><i style="background:#f55947"></i>what the segmenter found</span>
</div>

<div class="note">
  <b>How the flag works &mdash; and what it costs.</b>
  The verdict uses no ground truth. It runs the super-resolution network several
  times with dropout active, segments every output, and measures how much the
  segmenter disagrees with itself. The counter-intuitive part is the sign:
  <b>when a lesion is erased the segmenter gets more confident, not less.</b> It
  is confidently wrong, so unusually low disagreement is the danger signal.
  Slice-level AUROC <b>0.782</b> on held-out synthetic data.
  Stratified by lesion size it is 0.853 (medium) and 0.821 (large) &mdash; but
  <b>0.571, essentially chance, on the smallest lesions</b>, which is exactly
  where erasure matters most. That gap is the open problem, not a solved one.
</div>

<p class="foot">
  Data: <b>synthetic phantom slices</b>, not BraTS and not clinical data. Single
  checkpoint (seed 0), one slice chosen because it shows the effect clearly;
  the effect is real but modest on average &mdash; across 3 seeds the tumor-aware
  objective cut small-lesion erasure by 0.064 at equal PSNR, and roughly doubled
  false positives. Confidence is the MC-dropout segmentation spread, rescaled to
  0&ndash;100 across the frames shown here.
</p>

</div><script>
const P = __DATA__;
const F = P.frames, N = P.n_true;
const ticks = document.getElementById('ticks');
ticks.innerHTML = F.map(f => '<span>&times;' + f.factor + '</span>').join('');

function paint(i){
  const f = F[i];
  document.getElementById('flabel').textContent = '\\u00d7' + f.factor;
  document.getElementById('img-lr').src = 'data:image/png;base64,' + f.lr;
  for (const [k, p] of [['distortion','d'], ['tumor_aware','t']]){
    const m = f[k];
    document.getElementById('img-' + p).src = 'data:image/png;base64,' + m.img;
    document.getElementById(p + '-psnr').textContent = m.psnr.toFixed(2) + ' dB';
    document.getElementById(p + '-ssim').textContent = m.ssim.toFixed(3);
    const lost = N - m.found;
    const el = document.getElementById(p + '-found');
    el.textContent = m.found + ' of ' + N;
    el.style.color = lost > 0 ? 'var(--bad)' : 'var(--good)';
    document.getElementById(p + '-conf').textContent = m.confidence.toFixed(0) + '%';
    document.getElementById(p + '-meter').style.width = m.confidence + '%';
    const v = document.getElementById(p + '-verdict');
    if (lost > 0 && m.confidence >= 85){
      v.className = 'verdict v-bad';
      v.textContent = '\\u26a0 FLAG \\u2014 ' + lost + ' lesion' + (lost>1?'s':'') +
        ' erased, and the model is confident. Confidently wrong.';
    } else if (lost > 0){
      v.className = 'verdict v-bad';
      v.textContent = '\\u26a0 ' + lost + ' lesion' + (lost>1?'s':'') + ' erased.';
    } else {
      v.className = 'verdict v-safe';
      v.textContent = '\\u2713 All lesions preserved.';
    }
  }
}
const sl = document.getElementById('slider');
sl.addEventListener('input', e => paint(+e.target.value));
paint(+sl.value);
</script></body></html>
"""

OUT = os.path.join(ROOT, "demo_safety.html")
open(OUT, "w").write(HTML.replace("__DATA__", DATA))
print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
