"""Metrics: image quality, segmentation overlap, and the safety rates.

Image quality:  psnr, ssim  (self-contained torch; no skimage dependency).
Segmentation:   dice.
Safety (the contribution):
  * False Negative Erasure Rate  -- fraction of real lesions the downstream
    segmenter misses on the super-resolved image (pseudo-normalization).
  * False Positive Detection Rate -- fraction of predicted lesions that are
    fabricated, i.e. overlap no true lesion (hallucination).
Both are computed per connected-component lesion and can be stratified by
lesion size, which is the point: small lesions are where erasure bites.

All safety functions take numpy 2D arrays in {0,1}; helpers convert torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Image quality
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB). Higher is better."""
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = g[:, None] @ g[None, :]
    return w[None, None]  # (1,1,size,size)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         win: int = 11) -> float:
    """Structural similarity (mean over the image). Expects (B,1,H,W) or (1,H,W)."""
    if pred.dim() == 3:
        pred, target = pred[None], target[None]
    w = _gaussian_window(win).to(pred.dtype)
    pad = win // 2
    mu_x = F.conv2d(pred, w, padding=pad)
    mu_y = F.conv2d(target, w, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x = F.conv2d(pred ** 2, w, padding=pad) - mu_x2
    sig_y = F.conv2d(target ** 2, w, padding=pad) - mu_y2
    sig_xy = F.conv2d(pred * target, w, padding=pad) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return s.mean().item()


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Dice overlap between two binary masks."""
    p = (pred_mask > 0.5).astype(np.float32)
    g = (gt_mask > 0.5).astype(np.float32)
    inter = (p * g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


# --------------------------------------------------------------------------- #
# Connected components (4-connectivity flood fill, no scipy dependency)
# --------------------------------------------------------------------------- #
def label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected foreground regions. Returns (labels, count)."""
    b = (binary > 0.5)
    labels = np.zeros(b.shape, dtype=np.int32)
    cur = 0
    h, w = b.shape
    for i in range(h):
        for j in range(w):
            if b[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and b[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels, cur


# --------------------------------------------------------------------------- #
# Safety rates (the contribution)
# --------------------------------------------------------------------------- #
def lesion_records(gt_mask: np.ndarray, pred_mask: np.ndarray,
                   detect_thresh: float = 0.1) -> list[dict]:
    """Per-lesion detection status.

    For every connected lesion in the ground truth, record its size (pixels)
    and whether the predicted mask detects it (overlap fraction of the lesion
    covered by prediction >= detect_thresh).

    Returns a list of {"size": int, "detected": bool}.
    """
    gt_lab, n = label_components(gt_mask)
    pred = (pred_mask > 0.5)
    out = []
    for k in range(1, n + 1):
        region = gt_lab == k
        size = int(region.sum())
        covered = float((region & pred).sum()) / max(1, size)
        out.append({"size": size, "detected": covered >= detect_thresh})
    return out


def hallucination_stats(gt_mask: np.ndarray, pred_mask: np.ndarray,
                        overlap_thresh: float = 0.1) -> dict:
    """Fabricated (hallucinated) lesion stats.

    A predicted connected component is "fabricated" if it overlaps no true
    lesion (overlap fraction of the component < overlap_thresh with GT).

    Returns per-image counts plus the fabricated-pixel fraction.
    """
    pred_lab, m = label_components(pred_mask)
    gt = (gt_mask > 0.5)
    fabricated = 0
    for k in range(1, m + 1):
        comp = pred_lab == k
        overlap = float((comp & gt).sum()) / max(1, int(comp.sum()))
        if overlap < overlap_thresh:
            fabricated += 1
    pred_pixels = int((pred_mask > 0.5).sum())
    fab_pixels = int(((pred_mask > 0.5) & (~gt)).sum())
    return {
        "pred_components": m,
        "fabricated_components": fabricated,
        "fabricated_pixel_fraction": (fab_pixels / pred_pixels) if pred_pixels else 0.0,
    }


def size_bin(size: int, edges=(50, 200)) -> str:
    """Bucket a lesion size (pixels) into small / medium / large."""
    if size < edges[0]:
        return "small"
    if size < edges[1]:
        return "medium"
    return "large"


def aggregate_safety(records: list[dict], halluc: list[dict],
                     edges=(50, 200)) -> dict:
    """Aggregate per-image lesion records into rates, overall and by size bin.

    Args:
        records: concatenated lesion_records() across the test set.
        halluc:  list of hallucination_stats() dicts, one per image.
    Returns a dict with the False Negative Erasure Rate (overall + per bin) and
    the False Positive Detection Rate (fabricated components / predicted
    components), plus counts.
    """
    bins = {"small": [], "medium": [], "large": []}
    for r in records:
        bins[size_bin(r["size"], edges)].append(r)

    def erasure_rate(rs):
        if not rs:
            return None
        erased = sum(0 if r["detected"] else 1 for r in rs)
        return erased / len(rs)

    total_pred = sum(h["pred_components"] for h in halluc)
    total_fab = sum(h["fabricated_components"] for h in halluc)

    return {
        "n_lesions": len(records),
        "false_negative_erasure_rate": erasure_rate(records),
        "erasure_rate_by_size": {k: erasure_rate(v) for k, v in bins.items()},
        "lesions_by_size": {k: len(v) for k, v in bins.items()},
        "false_positive_detection_rate": (total_fab / total_pred) if total_pred else 0.0,
        "fabricated_components": total_fab,
        "predicted_components": total_pred,
    }


# --------------------------------------------------------------------------- #
# torch helpers
# --------------------------------------------------------------------------- #
def to_mask_np(logits_or_prob: torch.Tensor, is_logit: bool = True,
               thresh: float = 0.5) -> np.ndarray:
    """(1,1,H,W) or (1,H,W) logits/prob -> (H,W) binary numpy mask."""
    x = logits_or_prob.detach()
    if is_logit:
        x = torch.sigmoid(x)
    x = x.squeeze().cpu().numpy()
    return (x > thresh).astype(np.float32)
