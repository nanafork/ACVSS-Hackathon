"""Risk-controlled, model-agnostic erasure monitor.

At inference the monitor sees only the acquired scan, the enhanced image, and a
downstream segmenter it does not own. It never sees ground truth.

  1. PROPOSE   candidate lesion sites = connected components of p(acquired) > tau_lo.
               tau_lo is deliberately BELOW the operating threshold so sites the
               deployed segmenter would not report still get examined.
  2. SCORE     evidence destroyed, in log-odds:
                   delta = mean logit p(acquired) - mean logit p(enhanced)
               over the site. Positive delta = the enhancement removed evidence.
  3. CALIBRATE conformal threshold on held-out CASES: pick the largest delta
               threshold whose empirical miss rate on true erasures is <= alpha.
               Finite-sample corrected, distribution-free.
  4. FLAG      sites with delta >= threshold.

Ground truth is used ONLY to label erasures for calibration and scoring, exactly
as any supervised monitor is calibrated. Cases are split so calibration and
evaluation never share a patient.
"""
import sys, json
import os
ROOT = os.environ.get("TRUSTMRI_ROOT", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "safety", "assets")
CACHE = os.path.join(ASSETS, "data", "et_full.npz")
CKPT = os.path.join(ASSETS, "shared", "sh_w40_sl0.0.pt")
sys.path.insert(0, ROOT)

import numpy as np

TAU_LO = 0.15          # proposal threshold (below the 0.5 operating point)
OP = 0.5               # the deployed segmenter's operating threshold
ALPHA = 0.10           # target miss rate on true erasures


def components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); out = []; nxt = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                nxt += 1; stack = [(i, j)]; lab[i, j] = nxt; pix = []
                while stack:
                    y, x = stack.pop(); pix.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and lab[ny,nx]==0:
                            lab[ny,nx] = nxt; stack.append((ny,nx))
                m = np.zeros((h,w), bool)
                for y,x in pix: m[y,x] = True
                out.append(m)
    return out


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def size_bin(n):
    return "small" if n < 50 else ("medium" if n <= 200 else "large")


# ---- recover the case id of every cached val slice (needed for patient-level split)
cache = np.load(CACHE)
case_id_all, hr_all, mask_all = cache["case_id"], cache["hr"], cache["mask"]
cases = np.unique(case_id_all)
n_test = max(1, int(round(len(cases) * 0.2)))
n_val = max(1, int(round(len(cases) * 0.15)))
order = np.argsort([hash((int(c), 20260727)) for c in cases])
val_cases = set(cases[order[n_test:n_test + n_val]].tolist())
val_idx = np.array([int(c) in val_cases for c in case_id_all])
val_case_of_slice = case_id_all[val_idx]
val_masks = mask_all[val_idx]
keep = np.array([m.any() for m in (val_masks > 0)])          # watchdog_data skipped empty
case_of = val_case_of_slice[keep]

d = np.load(os.path.join(ASSETS, "val_cache.npz"))
x, gt_all, pred_all = d["x"], d["gt"], d["pred"]
assert len(x) == len(case_of), f"slice/case mismatch {len(x)} vs {len(case_of)}"
print(f"val: {len(x)} slices, {len(set(case_of.tolist()))} cases", flush=True)

# ---- build the site table
sites = []
for i in range(len(x)):
    p_lr = x[i, 3].astype(np.float32)
    p_sr = x[i, 2].astype(np.float32)
    gt = gt_all[i] > 0
    lr_hit = p_lr > OP
    sr_hit = p_sr > OP

    # true damage lesions on this slice: visible in the acquired scan, gone after SR
    damaged = []
    for m in components(gt):
        in_lr = (m & lr_hit).sum() / max(1, m.sum()) >= 0.1
        in_sr = (m & sr_hit).sum() / max(1, m.sum()) >= 0.1
        if in_lr and not in_sr:
            damaged.append(m)

    for site in components(p_lr > TAU_LO):
        if site.sum() < 4:
            continue
        delta = float(logit(p_lr[site]).mean() - logit(p_sr[site]).mean())
        overlaps_damage = any((site & m).any() for m in damaged)
        gt_overlap = (site & gt).any()
        sites.append({"case": int(case_of[i]), "slice": i, "delta": delta,
                      "area": int(site.sum()), "damage": bool(overlaps_damage),
                      "on_tumor": bool(gt_overlap),
                      "bin": size_bin(int(site.sum()))})
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(x)} slices, {len(sites)} sites", flush=True)

D = np.array([s["delta"] for s in sites])
Y = np.array([s["damage"] for s in sites])
CASE = np.array([s["case"] for s in sites])
print(f"\nsites: {len(sites)}   true-damage sites: {int(Y.sum())} "
      f"({100*Y.mean():.2f}%)", flush=True)

# ---- conformal calibration on half the cases, evaluation on the other half
ucases = np.unique(CASE)
half = len(ucases) // 2
cal_cases = set(ucases[:half].tolist())
cal = np.array([c in cal_cases for c in CASE])
ev = ~cal

pos_cal = D[cal & Y]
n = len(pos_cal)
# conformal quantile with finite-sample correction: the alpha-quantile of the
# calibration positives, adjusted by (n+1) so coverage holds in expectation.
q = int(np.floor(ALPHA * (n + 1))) - 1
q = max(0, min(q, n - 1))
thr = float(np.sort(pos_cal)[q])
print(f"calibration: {n} damage sites over {len(cal_cases)} cases  ->  "
      f"threshold delta >= {thr:.3f} for <= {100*ALPHA:.0f}% miss rate", flush=True)

fired = D[ev] >= thr
yev = Y[ev]
miss = 1 - (fired & yev).sum() / max(1, yev.sum())
fa = (fired & ~yev).sum() / max(1, (~yev).sum())
n_slices_ev = len(set(np.array([s["slice"] for s in sites])[ev].tolist()))
print(f"\n=== evaluation half ({len(ucases)-half} unseen cases) ===")
print(f"  true damage sites            {int(yev.sum())}")
print(f"  miss rate (target <= {100*ALPHA:.0f}%)     {100*miss:.1f}%")
print(f"  false-alarm rate on non-damage sites  {100*fa:.1f}%")
print(f"  flags raised per slice       {fired.sum()/max(1,n_slices_ev):.2f}")

# how many false alarms sit on a real tumor (a cautious flag) vs on nothing
ONT = np.array([s["on_tumor"] for s in sites])[ev]
fp = fired & ~yev
print(f"  of the false alarms, {100*(fp & ONT).sum()/max(1,fp.sum()):.0f}% land on a real "
      f"tumor (partial suppression), {100*(fp & ~ONT).sum()/max(1,fp.sum()):.0f}% on healthy tissue")

# ---- ROC, and the binary-rule baseline for comparison
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    if y.all() or not y.any(): return float("nan")
    o = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); st = cum - cnt
    r = ((st + cum + 1) / 2.0)[inv]
    return float((r[y].sum() - y.sum()*(y.sum()+1)/2) / (y.sum()*(~y).sum()))

print(f"\n  AUROC of the log-odds score (evaluation half): {auroc(D[ev], yev):.3f}")
json.dump({"threshold": thr, "alpha": ALPHA, "miss": float(miss), "fa": float(fa),
           "auroc": auroc(D[ev], yev), "n_sites": len(sites),
           "n_damage": int(Y.sum())},
          open(os.path.join(ROOT, "safety", "results", "out.json"), "w"), indent=2)
print("wrote /marimo/watchdog/monitor.json")
