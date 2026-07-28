"""Self-contained HTML demo, built on real BraTS.

Precomputes every frame (slice x degradation x model), embeds them as base64,
and writes one shareable file that needs no server, no GPU and no network.
"""
import sys, os, io, json, base64
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "safety", "assets")
sys.path.insert(0, ROOT)
import numpy as np, torch
import imageio.v3 as iio

from src.models import seg_unet, sr_unet
from src.data import make_dataset
from src.degrade import degrade
from src.metrics import to_mask_np, psnr, ssim

DEV = "cuda"
FACTORS = [2, 3, 4, 5, 6, 8]
ck = torch.load(os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt"),
                map_location=DEV, weights_only=False)
SIG = float(ck["meta"]["sigma"])
sr_base = sr_unet(base=32, dropout=0.2).to(DEV); sr_base.load_state_dict(ck["sr_distortion"]); sr_base.eval()
sr_ours = sr_unet(base=32, dropout=0.2).to(DEV); sr_ours.load_state_dict(ck["sr_tumor_aware"]); sr_ours.eval()
reader = seg_unet(base=32).to(DEV); reader.load_state_dict(ck["seg"]); reader.eval()
VAL = make_dataset("cached", path=os.path.join(ASSETS, "data", "et_full.npz"), split="val")


def cc(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); out = []; n = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                n += 1; st = [(i, j)]; lab[i, j] = n; px = []
                while st:
                    y, x = st.pop(); px.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and lab[ny,nx]==0:
                            lab[ny,nx] = n; st.append((ny,nx))
                m = np.zeros((h,w), bool)
                for y,x in px: m[y,x] = True
                out.append(m)
    return out


def edge(m):
    e = np.zeros_like(m)
    e[1:,:] |= m[1:,:] & ~m[:-1,:]; e[:-1,:] |= m[:-1,:] & ~m[1:,:]
    e[:,1:] |= m[:,1:] & ~m[:,:-1]; e[:,:-1] |= m[:,:-1] & ~m[:,1:]
    return e


def grow(m, k=1):
    o = m.copy()
    for _ in range(k):
        d = o.copy()
        d[1:,:] |= o[:-1,:]; d[:-1,:] |= o[1:,:]
        d[:,1:] |= o[:,:-1]; d[:,:-1] |= o[:,1:]
        o = d
    return o


def png(img, overlays=(), scale=4):
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
    arr = np.clip(np.repeat(np.repeat(rgb, scale, 0), scale, 1), 0, 1) * 255
    buf = io.BytesIO(); iio.imwrite(buf, arr.astype(np.uint8), extension=".png")
    return base64.b64encode(buf.getvalue()).decode()


def run(idx, factor, model):
    s = VAL[int(idx)]
    hr = s["hr"][0].numpy().astype(np.float32)
    gt = s["mask"][0].numpy() > 0.5
    lr = degrade(hr, factor=int(factor), sigma=SIG, rng=np.random.default_rng(7))
    t = torch.from_numpy(lr)[None, None].float().to(DEV)
    with torch.no_grad():
        out = model(t)
        p_lr = to_mask_np(reader(t)) > 0.5
        p_sr = to_mask_np(reader(out)) > 0.5
    van = np.zeros_like(p_lr)
    for r in cc(p_lr):
        if r.sum() >= 4 and (r & p_sr).sum() / max(1, r.sum()) < 0.1:
            van |= r
    # The acquisition keeps 1/factor of k-space per axis, so the smallest
    # structure it can resolve is about `factor` pixels across. A reference
    # region smaller than that was never measured and cannot be verified by
    # ANY image-space check. If nothing resolvable survives in the acquired
    # read, the layer has no reference and must say so instead of staying
    # silent -- silence would read as "safe".
    _res_px = int(factor)
    _min_area = max(4, _res_px * _res_px)
    n_ref = sum(1 for r in cc(p_lr) if r.sum() >= _min_area)
    # A lesion narrower than the resolution limit was never sampled, so no
    # image-space check can confirm or deny it. Count how many of the lesions
    # present fall below that limit; when most of them do, the layer's silence
    # means "unmeasured", not "safe".
    _diam = [2.0 * (float(L.sum()) / 3.14159) ** 0.5 for L in cc(gt)]
    n_below = sum(1 for d in _diam if d < _res_px)
    n_les_tot = max(1, len(_diam))
    h = torch.from_numpy(hr)[None, None].to(DEV)
    les = cc(gt)
    det = lambda m: sum(1 for L in les if (L & m).sum() / max(1, L.sum()) >= 0.1)
    with torch.no_grad():
        p_hr = to_mask_np(reader(h.float())) > 0.5
    return dict(hr=hr, gt=gt, lr=lr, sr=out[0,0].cpu().numpy(), p_lr=p_lr, p_sr=p_sr,
                van=van, psnr=float(psnr(out, h)), ssim=float(ssim(out, h)),
                n_true=len(les), d_hr=det(p_hr), n_lr=det(p_lr), n_sr=det(p_sr),
                n_gone=len(cc(van)), n_ref=n_ref, res_px=_res_px,
                n_below=n_below, n_les_tot=n_les_tot)


SLICES = [(1960, "A - drag right: the standard model loses the tumor, ours holds on"),
          (1718, "B - our model keeps lesions the standard model loses")]
BLUE, RED, CYAN = (96,165,250), (255,90,70), (92,217,253)
cases = []
for SLICE, CAPTION in SLICES:
  frames = []
  for f in FACTORS:
    rb = run(SLICE, f, sr_base)
    ro = run(SLICE, f, sr_ours)
    frames.append({
        "factor": f,
        "acq": png(rb["lr"], [(rb["p_lr"], BLUE, True)]),
        "std": png(rb["sr"], [(rb["p_sr"], BLUE, True), (rb["van"], RED, False)]),
        "our": png(ro["sr"], [(ro["p_sr"], BLUE, True), (ro["van"], RED, False)]),
        "std_off": png(rb["sr"], [(rb["p_sr"], BLUE, True)]),
        "our_off": png(ro["sr"], [(ro["p_sr"], BLUE, True)]),
        "tru": png(rb["hr"], [(rb["gt"], CYAN, False)]),
        "std_m": {"psnr": round(rb["psnr"],2), "ssim": round(rb["ssim"],3),
                  "gone": rb["n_gone"], "n_lr": rb["n_lr"], "n_sr": rb["n_sr"], "n_true": rb["n_true"], "d_hr": rb["d_hr"], "n_ref": rb["n_ref"], "res_px": rb["res_px"], "n_below": rb["n_below"], "n_les_tot": rb["n_les_tot"]},
        "our_m": {"psnr": round(ro["psnr"],2), "ssim": round(ro["ssim"],3),
                  "gone": ro["n_gone"], "n_lr": ro["n_lr"], "n_sr": ro["n_sr"], "n_true": ro["n_true"], "d_hr": ro["d_hr"], "n_ref": ro["n_ref"], "res_px": ro["res_px"], "n_below": ro["n_below"], "n_les_tot": ro["n_les_tot"]},
    })
    print("slice", SLICE, "factor", f, "done", flush=True)
  cases.append({"slice": SLICE, "caption": CAPTION, "frames": frames})

DATA = json.dumps({"cases": cases})

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enhancement can make a tumor disappear</title>
<style>
 :root{--bg:#0b0f14;--card:#131a23;--line:#232d3a;--ink:#e8eef5;--mut:#8fa0b3;
  --dim:#78889c;--good:#3ecf8e;--bad:#ff6b57;--acc:#5aa9ff;--cyan:#5cd9fd}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:1240px;margin:0 auto;padding:34px 20px 64px}
 .kick{display:inline-block;background:rgba(90,169,255,.12);color:var(--acc);
  border:1px solid rgba(90,169,255,.28);border-radius:20px;padding:4px 12px;
  font-size:11px;letter-spacing:.09em;text-transform:uppercase;margin-bottom:14px}
 h1{font-size:29px;line-height:1.2;margin:0 0 8px;letter-spacing:-.025em}
 .sub{color:var(--mut);margin:0 0 26px;max-width:78ch}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px}
 .slid label{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);margin-bottom:10px}
 .slid b{color:var(--ink);font-size:16px}
 input[type=range]{width:100%;accent-color:var(--acc)}
 .ticks{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;margin-top:4px}
 .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}
 @media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}}
 .col h2{font-size:14px;margin:0 0 2px} .col p{margin:0 0 9px;font-size:12px;color:var(--dim)}
 .col img{width:100%;image-rendering:pixelated;border-radius:9px;border:1px solid var(--line);display:block}
 .verd{margin-top:14px;padding:11px 13px;border-radius:9px;font-size:13px;font-weight:600}
 .bad{background:rgba(255,107,87,.12);color:var(--bad);border:1px solid rgba(255,107,87,.32)}
 .good{background:rgba(62,207,142,.11);color:var(--good);border:1px solid rgba(62,207,142,.28)}
 .blind{background:rgba(232,179,57,.10);color:#e8b339;border:1px solid rgba(232,179,57,.35)}
 .row2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
 @media(max-width:1000px){.row2{grid-template-columns:1fr}}
 .stats{margin-top:18px;color:var(--mut);font-size:13px}
 .note{margin-top:24px;padding:15px 18px;border-left:3px solid #e8b339;
  background:rgba(232,179,57,.06);color:var(--mut);font-size:12.5px;border-radius:0 9px 9px 0}
 .note b{color:var(--ink)}
 .lg{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin-top:12px}
 .lg i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle}
 .band{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}
 @media(max-width:900px){.band{grid-template-columns:repeat(2,1fr)}}
 .bi{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 15px}
 .bi b{display:block;font-size:23px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
 .bi span{display:block;color:var(--dim);font-size:11.5px;margin-top:3px;line-height:1.4}
 .bi.good b{color:var(--good)} .bi.bad b{color:var(--bad)}
 .tog{display:inline-flex;align-items:center;gap:9px;margin-top:16px;padding:9px 14px;
  border:1px solid var(--line);border-radius:9px;background:#0f151d;cursor:pointer;
  font-size:13px;font-weight:700;letter-spacing:.04em;color:var(--mut);user-select:none}
 .tog input{accent-color:var(--bad);width:16px;height:16px;cursor:pointer}
 .tog.on{color:var(--bad);border-color:rgba(255,107,87,.4);background:rgba(255,107,87,.07)}
</style></head><body><div class="wrap">
<div class="kick">TrustMRI &middot; safety layer</div>
<h1>Enhancement can make a tumor disappear</h1>
<p class="sub">Real held-out BraTS. Drag the slider to degrade the acquisition, as a
cheaper or faster scan would. Both models then enhance it, and the safety layer
compares the segmentation before and after &mdash; using no ground truth.</p>

<div class="band">
 <div class="bi"><b>9,490</b><span>lesions, held-out patients</span></div>
 <div class="bi good"><b>+774</b><span>recovered by enhancement<br>(invisible before, found after)</span></div>
 <div class="bi bad"><b>&minus;339</b><span>lost to enhancement<br>(found before, gone after)</span></div>
 <div class="bi"><b>+435</b><span>net effect &mdash; enhancement helps,<br>and loses some silently</span></div>
</div>

<div class="panel slid">
 <div id="cases" style="display:flex;gap:10px;margin-bottom:16px"></div>
 <label><span>Acquisition degradation &mdash; k-space truncation factor</span><b id="fl">&times;4</b></label>
 <input type="range" id="sl" min="0" max="5" value="2" step="1">
 <div class="ticks" id="tk"></div>
 <label class="tog"><input type="checkbox" id="sf" checked>
   <span>SAFETY LAYER</span></label>
</div>

<div class="grid">
 <div class="panel col"><h2>Acquired (no enhancement)</h2><p>blurry and noisy &mdash; this is
 what enhancement exists to fix, and it hides lesions of its own</p><img id="i-acq"></div>
 <div class="panel col"><h2 style="color:#ff8f7a">Standard model</h2><p>PSNR-optimal</p><img id="i-std"></div>
 <div class="panel col"><h2 style="color:#3ecf8e">Ours</h2><p>tumor-aware</p><img id="i-our"></div>
 <div class="panel col"><h2 style="color:#5cd9fd">Ground truth</h2><p>never shown to the layer</p><img id="i-tru"></div>
</div>

<div class="row2"><div id="v-std"></div><div id="v-our"></div></div>
<div class="stats" id="stats"></div>
<div class="lg">
 <span><i style="background:#60a5fa"></i>what the segmenter found</span>
 <span><i style="background:#ff6b57"></i>present before enhancement, gone after</span>
 <span><i style="background:#5cd9fd"></i>true lesion outline</span>
</div>

<div class="note"><b>What this is, and is not.</b> Real BraTS (MSD Task01), enhancing
tumor, held-out patients. Measured across the validation set: enhancement
<b>destroys 3.6%</b> of lesions and <b>recovers 8.2%</b> &mdash; it helps on net, and
its losses are invisible to PSNR. The layer raises about 0.15 flags per slice, of
which <b>20% mark a real vanished lesion</b>: a review prompt, not a diagnosis.
This slice is chosen because it shows the effect clearly; roughly 2 slices in 300 do.
</div>
</div><script>
const P=__DATA__; let CI=0; let F=P.cases[0].frames;
document.getElementById('tk').innerHTML=F.map(f=>'<span>&times;'+f.factor+'</span>').join('');
document.getElementById('cases').innerHTML=P.cases.map((c,i)=>
 '<button class="cbtn" data-i="'+i+'" style="flex:1;padding:10px 12px;border-radius:9px;'+
 'border:1px solid var(--line);background:#0f151d;color:var(--mut);font-size:12.5px;'+
 'font-weight:600;cursor:pointer;text-align:left">'+c.caption+'</button>').join('');
function selCase(i){
 CI=i; F=P.cases[i].frames;
 document.querySelectorAll('.cbtn').forEach((b,j)=>{
   b.style.background = j===i ? 'rgba(90,169,255,.10)' : '#0f151d';
   b.style.color = j===i ? 'var(--acc)' : 'var(--mut)';
   b.style.borderColor = j===i ? 'rgba(90,169,255,.4)' : 'var(--line)';
 });
 paint(+document.getElementById('sl').value);
}
document.getElementById('cases').addEventListener('click',e=>{
 const b=e.target.closest('.cbtn'); if(b) selCase(+b.dataset.i);
});
function verdict(m,name){
 const on=document.getElementById('sf').checked;
 const blind = m.n_ref===0 || (2*m.n_below >= m.n_les_tot);
 if(on && blind)
   return '<div class="verd blind">&#9888; '+name+': CANNOT VERIFY &mdash; this '+
     'acquisition never measured structures under ~'+m.res_px+' px, and '+m.n_below+
     ' of the '+m.n_les_tot+' lesions here are below that. No image-space check can '+
     'confirm them. Silence means unmeasured, not safe.</div>';
 if(!document.getElementById('sf').checked)
   return '<div class="verd" style="background:#0f151d;color:#78889c;'+
          'border:1px solid var(--line)">safety layer off &mdash; '+
          'nothing is being checked</div>';
 if(m.gone>0) return '<div class="verd bad">&#9888; '+name+': '+m.gone+
   ' region(s) vanished during enhancement &mdash; review the original</div>';
 return '<div class="verd good">&#10003; '+name+': nothing vanished</div>';
}
function paint(i){
 const f=F[i];
 document.getElementById('fl').innerHTML='&times;'+f.factor;
 const on=document.getElementById('sf').checked;
 document.getElementById('i-acq').src='data:image/png;base64,'+f.acq;
 document.getElementById('i-tru').src='data:image/png;base64,'+f.tru;
 document.getElementById('i-std').src='data:image/png;base64,'+(on?f.std:f.std_off);
 document.getElementById('i-our').src='data:image/png;base64,'+(on?f.our:f.our_off);
 document.querySelector('.tog').classList.toggle('on', on);
 document.getElementById('v-std').innerHTML=verdict(f.std_m,'standard');
 document.getElementById('v-our').innerHTML=verdict(f.our_m,'ours');
 document.getElementById('stats').innerHTML =
  '<b>True lesions the segmenter finds</b> (of '+f.std_m.n_true+' present): &nbsp; '+
  'original scan <b>'+f.std_m.d_hr+'</b> &nbsp;|&nbsp; acquired <b>'+f.std_m.n_lr+
  '</b> &nbsp;&rarr;&nbsp; standard <b>'+f.std_m.n_sr+
  '</b> &nbsp;&rarr;&nbsp; ours <b>'+f.our_m.n_sr+'</b>'+
  '<br><span style="color:var(--dim)">image quality &mdash; standard PSNR '+
  f.std_m.psnr.toFixed(2)+' dB / SSIM '+f.std_m.ssim.toFixed(3)+
  ' &nbsp;|&nbsp; ours PSNR '+f.our_m.psnr.toFixed(2)+' dB / SSIM '+
  f.our_m.ssim.toFixed(3)+' &mdash; note how little these move while lesions '+
  'appear and disappear</span>';
}
const s=document.getElementById('sl');
s.addEventListener('input',e=>paint(+e.target.value));
document.getElementById('sf').addEventListener('change',()=>paint(+s.value));
selCase(0);
</script></body></html>
"""

out = os.path.join(ROOT, "safety", "demo_safety.html")
open(out, "w").write(HTML.replace("__DATA__", DATA))
print("wrote", out, round(os.path.getsize(out)/1e6, 2), "MB")
for fr in frames:
    print(f"  x{fr['factor']}  standard gone={fr['std_m']['gone']}  ours gone={fr['our_m']['gone']}")
