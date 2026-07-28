"""End-to-end evaluation: compare distortion-optimal vs tumor-aware SR.

For each test slice we:
  1. degrade the HR scan (k-space truncation + Rician noise)  -> LR baseline,
  2. super-resolve with each SR model,
  3. segment every version with the frozen segmenter,
  4. score image quality (PSNR/SSIM vs HR), segmentation Dice, and the safety
     rates (lesion erasure, hallucination), and
  5. estimate MC-dropout uncertainty and test whether it predicts the SR error.

The headline comparison the proposal promises is the returned dict: at similar
PSNR/SSIM, does the tumor-aware model have a lower False Negative Erasure Rate?
"""

from __future__ import annotations

import numpy as np
import torch

from .degrade import degrade
from .metrics import (aggregate_safety, dice, hallucination_stats,
                      lesion_records, psnr, ssim, to_mask_np)
from .uncertainty import mc_predict, uncertainty_error_auroc


def _seg_mask(segmenter, img: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = segmenter(img)
    return to_mask_np(logits, is_logit=True)


def evaluate_pipeline(sr_models: dict, segmenter, dataset, factor: int = 4,
                      sigma: float = 0.02, device: str = "cpu",
                      mc_passes: int = 10, err_thresh: float = 0.1,
                      size_edges=(50, 200), seed: int = 7) -> dict:
    """Run the full comparison over a dataset.

    Args:
        sr_models: {name: model}. Include the special name "lowres" is NOT
            needed; the degraded input is always evaluated as the baseline.
        segmenter: frozen segmentation model.
        dataset: yields {'hr','mask'}.
    Returns:
        dict keyed by model name (plus "lowres" baseline) with mean image
        quality, Dice, safety aggregate, and mean uncertainty-error AUROC.
    """
    segmenter.to(device).eval()
    for m in sr_models.values():
        m.to(device).eval()

    # Accumulators. "truehr" is the segmenter's own detection floor: what it
    # misses when handed the original scan with no degradation and no
    # reconstruction at all. Without that row an erasure rate has no denominator
    # -- on real BraTS enhancing tumor the floor is around 45%, so quoting a
    # model at 48% as though it erased 48% of lesions blames the enhancement for
    # the segmenter's blindness. The number that means something is the excess
    # over this floor.
    names = ["truehr", "lowres"] + list(sr_models.keys())
    acc = {n: {"psnr": [], "psnr_brain": [], "ssim": [], "dice": [],
               "records": [], "halluc": [], "auroc": []} for n in names}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        hr = sample["hr"][None].to(device)      # (1,1,H,W)
        gt = sample["mask"][0].cpu().numpy()    # (H,W)
        rng = np.random.default_rng(seed + idx)
        lr_np = degrade(hr[0, 0].cpu().numpy(), factor=factor, sigma=sigma, rng=rng)
        lr = torch.from_numpy(lr_np)[None, None].to(device)

        versions = {"truehr": hr, "lowres": lr}
        for name, model in sr_models.items():
            with torch.no_grad():
                versions[name] = model(lr)

        # Brain mask from the reference scan: anything above the background floor.
        brain = (hr > 0.05)
        for name, img in versions.items():
            acc[name]["psnr"].append(psnr(img, hr))
            acc[name]["psnr_brain"].append(psnr(img, hr, mask=brain))
            acc[name]["ssim"].append(ssim(img, hr))
            pred_mask = _seg_mask(segmenter, img)
            acc[name]["dice"].append(dice(pred_mask, gt))
            acc[name]["records"].extend(lesion_records(gt, pred_mask))
            acc[name]["halluc"].append(hallucination_stats(gt, pred_mask))

        # Uncertainty only for the SR models (dropout networks).
        for name, model in sr_models.items():
            mean, unc = mc_predict(model, lr, passes=mc_passes)
            err = (mean - hr).abs().squeeze().cpu().numpy() > err_thresh
            auroc = uncertainty_error_auroc(unc.squeeze().cpu().numpy(), err)
            if not np.isnan(auroc):
                acc[name]["auroc"].append(auroc)
            # mc_predict switches dropout back on and does not undo it. Without
            # this, every slice after the first was super-resolved stochastically
            # at the top of the loop, so the reconstruction being scored was not
            # the model's actual deterministic output. This silently corrupted
            # every safety number this function has ever produced.
            model.eval()

    # Reduce.
    results = {}
    for name in names:
        a = acc[name]
        results[name] = {
            # PSNR/SSIM of the reference against itself are degenerate, so they
            # are reported as NaN rather than as a perfect score. A table showing
            # "99.0 dB" invites reading the floor row as a model result.
            "psnr": float("nan") if name == "truehr" else float(np.mean(a["psnr"])),
            "psnr_brain": (float("nan") if name == "truehr"
                           else float(np.nanmean(a["psnr_brain"]))),
            "ssim": float("nan") if name == "truehr" else float(np.mean(a["ssim"])),
            "dice": float(np.mean(a["dice"])),
            "safety": aggregate_safety(a["records"], a["halluc"], edges=size_edges),
            "uncertainty_error_auroc": (float(np.mean(a["auroc"]))
                                        if a["auroc"] else float("nan")),
        }

    # Erasure attributable to the pipeline, i.e. above the segmenter's floor.
    floor = results["truehr"]["safety"]["false_negative_erasure_rate"]
    for name, r in results.items():
        rate = r["safety"]["false_negative_erasure_rate"]
        r["safety"]["segmenter_floor"] = floor
        r["safety"]["erasure_above_floor"] = (None if rate is None or floor is None
                                              else rate - floor)
    return results


def format_results(results: dict) -> str:
    """Pretty one-screen summary table."""
    lines = []
    hdr = f"{'model':<14}{'PSNR':>7}{'SSIM':>7}{'Dice':>7}{'Erase':>7}" \
          f"{'Er.sm':>7}{'FPdet':>7}{'U-AUC':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, r in results.items():
        s = r["safety"]
        er = s["false_negative_erasure_rate"]
        er_sm = s["erasure_rate_by_size"].get("small")
        fp = s["false_positive_detection_rate"]
        au = r["uncertainty_error_auroc"]
        def f(x):
            return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
        lines.append(
            f"{name:<14}{r['psnr']:>7.2f}{r['ssim']:>7.3f}{r['dice']:>7.3f}"
            f"{f(er):>7}{f(er_sm):>7}{f(fp):>7}{f(au):>7}")
    lines.append("")
    lines.append("Erase = False Negative Erasure Rate (lower is safer); "
                 "Er.sm = erasure rate on small lesions;")
    lines.append("FPdet = False Positive Detection Rate (hallucination); "
                 "U-AUC = uncertainty predicts error (0.5=chance).")
    return "\n".join(lines)
