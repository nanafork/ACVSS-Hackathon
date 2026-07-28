"""Bridge: turn this project's 2D super-resolution output into a 3D scene.

The SR/segmentation models work per 2D slice. To visualize their behaviour in
3D (the neuro-voxel way), we:

  1. build a *coherent* 3D synthetic phantom -- an ellipsoid "brain" with a few
     spherical "tumors" of varying size, styled like ``SyntheticSliceDataset``
     so the trained models behave meaningfully;
  2. degrade each axial slice (k-space truncation + Rician noise), super-resolve
     it with both the distortion-optimal and tumor-aware models, and segment
     every version;
  3. restack the per-slice results into 3D volumes and wrap them as
     ``viz.PatientVolume`` objects the analyzer/renderer can consume.

The headline: at similar image quality, the distortion model's reconstruction
makes the segmenter *erase* small tumors, so its predicted tumor volume (cm3)
is lower than the tumor-aware model's -- visible as missing blobs in 3D.

No download or GPU required; run ``python viz_bridge.py`` for a quick summary.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from src.checkpoint import load_models
from src.degrade import degrade
from src.metrics import to_mask_np
from src.uncertainty import mc_predict
from viz import PatientVolume, VolumeAnalyzer

CKPT = os.environ.get("ACVSS_CKPT", "checkpoints/demo_et_v2.pt")

# A real held-out case, if one has been fetched. When present the 3D scene shows
# actual anatomy from a patient the models never saw, instead of the phantom.
REAL_VOLUME = os.environ.get("ACVSS_VOLUME", "data/real_case.npz")


def load_real_volume(path: str = ""):
    """Return (volume, mask) for a real case, or None if none is available.

    The array is (D, H, W) in [0, 1] with a binary lesion mask, matching what
    make_phantom_3d produces, so the rest of the pipeline does not care which
    one it is given.
    """
    path = path or REAL_VOLUME
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["vol"].astype(np.float32), d["mask"].astype(np.uint8)

# Tumor placement as (fz, fy, fx, radius_voxels): a spread of sizes so lesion
# erasure is visible -- the small ones are the ones a distortion model drops.
TUMOR_SPECS = [
    (0.42, 0.44, 0.56, 8.0),
    (0.58, 0.56, 0.44, 5.0),
    (0.50, 0.62, 0.60, 3.0),
    (0.55, 0.40, 0.46, 2.0),
]


def make_phantom_3d(size: int = 96, depth: int = 96, seed: int = 7):
    """Return (volume, mask): coherent 3D brain phantom + binary tumor labels.

    Styled to match ``SyntheticSliceDataset``: brain tissue ~0.55, bright tumor
    core ~0.95, values in [0, 1]. Arrays are (D, H, W).
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:depth, 0:size, 0:size].astype(np.float32)
    cz, cy, cx = depth / 2, size / 2, size / 2

    vol = np.zeros((depth, size, size), dtype=np.float32)
    mask = np.zeros((depth, size, size), dtype=np.uint8)

    # Brain ellipsoid.
    rz, ry, rx = depth * 0.40, size * 0.42, size * 0.34
    brain = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2
             + ((xx - cx) / rx) ** 2) <= 1.0
    vol[brain] = 0.55

    # A few soft anatomy-like blobs for texture (never as bright as a tumor).
    for _ in range(6):
        bz = rng.uniform(0.30, 0.70) * depth
        by = rng.uniform(0.30, 0.70) * size
        bx = rng.uniform(0.30, 0.70) * size
        br = rng.uniform(0.06, 0.12) * size
        val = rng.uniform(0.10, 0.25) * (1 if rng.random() > 0.5 else -1)
        blob = ((zz - bz) ** 2 + (yy - by) ** 2 + (xx - bx) ** 2) <= br ** 2
        vol[blob & brain] += val

    # Spherical tumors of varied size.
    for fz, fy, fx, r in TUMOR_SPECS:
        tz, ty, tx = fz * depth, fy * size, fx * size
        sphere = ((zz - tz) ** 2 + (yy - ty) ** 2 + (xx - tx) ** 2) <= r ** 2
        sphere &= brain
        vol[sphere] = 0.95
        mask[sphere] = 1

    return np.clip(vol, 0.0, 1.0), mask


def run_pipeline_3d(vol, gt_mask, seg, sr_d, sr_t, factor, sigma,
                    device="cpu", seed=0, mc_passes=10):
    """Run the real models slice-by-slice and restack into 3D volumes.

    Returns a dict of (D,H,W) arrays: the degraded input, the tumor-aware and
    distortion SR image volumes, predicted tumor-mask volumes for true-HR,
    tumor-aware and distortion inputs, and the tumor-aware MC dropout
    uncertainty volume.
    """
    depth = vol.shape[0]
    out = {
        "lr": np.zeros_like(vol),
        "sr_tumor_aware": np.zeros_like(vol),
        "sr_distortion": np.zeros_like(vol),
        "pred_true": np.zeros_like(gt_mask),
        "pred_tumor_aware": np.zeros_like(gt_mask),
        "pred_distortion": np.zeros_like(gt_mask),
        "unc_tumor_aware": np.zeros_like(vol),
    }
    rng = np.random.default_rng(seed)
    for z in range(depth):
        hr_np = vol[z]
        hr = torch.from_numpy(hr_np)[None, None].to(device)
        lr_np = degrade(hr_np, factor=factor, sigma=sigma, rng=rng)
        lr = torch.from_numpy(lr_np)[None, None].float().to(device)
        with torch.no_grad():
            srd = sr_d(lr)
            srt = sr_t(lr)
            out["pred_true"][z] = to_mask_np(seg(hr))
            out["pred_distortion"][z] = to_mask_np(seg(srd))
            out["pred_tumor_aware"][z] = to_mask_np(seg(srt))
        out["lr"][z] = lr_np
        out["sr_distortion"][z] = srd[0, 0].cpu().numpy()
        out["sr_tumor_aware"][z] = srt[0, 0].cpu().numpy()

    # Uncertainty runs in its own pass, after every deterministic prediction is
    # already stored. mc_predict leaves dropout switched on, so interleaving it
    # with the loop above would make the reconstructions themselves stochastic.
    for z in range(depth):
        lr = torch.from_numpy(out["lr"][z])[None, None].float().to(device)
        with torch.no_grad():
            _, unc = mc_predict(sr_t, lr, passes=mc_passes)
        out["unc_tumor_aware"][z] = unc[0, 0].cpu().numpy()
    sr_t.eval()  # restore deterministic inference for any later caller
    return out


def build_patient_volumes(spacing=(1.0, 1.0, 1.0), device="cpu", seed=7):
    """End-to-end: phantom -> real models -> PatientVolume objects + cm3 stats.

    Returns (patients, volumes_cm3) where ``patients`` maps a version name to a
    ``PatientVolume`` (brain image + that version's tumor mask) and
    ``volumes_cm3`` maps the same names to tumor volume in cm3.
    """
    seg, sr_d, sr_t, meta = load_models(CKPT, device=device)
    size = int(meta.get("size", 96))
    factor = int(meta.get("factor", 4))
    sigma = float(meta.get("sigma", 0.03))

    real = load_real_volume()
    if real is not None:
        vol, gt_mask = real
        print(f"3D scene: REAL held-out case, volume {vol.shape}")
    else:
        vol, gt_mask = make_phantom_3d(size=size, depth=size, seed=seed)
        print(f"3D scene: synthetic phantom (no {REAL_VOLUME}), volume {vol.shape}")

    # Surround the volume with empty space before anything is meshed. A real
    # case is a crop out of a larger head, so tissue runs right up to the array
    # boundary; marching cubes then leaves the surface open there and renders
    # the cut as a large flat sheet across the scene. A few voxels of zeros give
    # the isosurface somewhere to close.
    pad = 4
    vol = np.pad(vol, pad, mode="constant")
    gt_mask = np.pad(gt_mask, pad, mode="constant")
    res = run_pipeline_3d(vol, gt_mask, seg, sr_d, sr_t, factor, sigma,
                          device=device, seed=seed)

    # Brain image for the "glass brain": the tumor-aware reconstruction (our
    # model's own output). Mask varies per version.
    brain_img = res["sr_tumor_aware"]
    specs = {
        "true": gt_mask,
        "tumor-aware": res["pred_tumor_aware"],
        "distortion": res["pred_distortion"],
    }
    analyzer = VolumeAnalyzer()
    patients, volumes_cm3 = {}, {}
    for name, m in specs.items():
        pv = PatientVolume(
            id=f"phantom-{name}",
            # The uncertainty volume rides along as a second "modality" so the
            # renderer can reach it without changing this function's signature.
            modalities={"t1": brain_img, "uncertainty": res["unc_tumor_aware"]},
            mask=m.astype(np.uint8),
            affine=np.eye(4),
            spacing=spacing,
        )
        patients[name] = pv
        volumes_cm3[name] = analyzer.calculate_volume(pv, label_idx=1)
    return patients, volumes_cm3


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients, vols = build_patient_volumes(device=device)
    print("\nTumor volume by version (cm3):")
    true_v = vols["true"]
    for name in ["true", "tumor-aware", "distortion"]:
        v = vols[name]
        pct = 100.0 * v / true_v if true_v else float("nan")
        print(f"  {name:12s} {v:7.2f} cm3   ({pct:5.1f}% of true)")
    erased = vols["true"] - vols["distortion"]
    saved = vols["tumor-aware"] - vols["distortion"]
    print(f"\nDistortion model erases {erased:.2f} cm3 of tumor vs ground truth.")
    print(f"Tumor-aware model recovers {saved:.2f} cm3 the distortion model loses.")
