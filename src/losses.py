"""Super-resolution losses: distortion-optimal vs tumor-aware.

The whole contribution hinges on the difference between these two objectives:

  * distortion_loss  -- plain pixel L1. This is the distortion-optimal
    baseline; it regresses toward the mean and barely "notices" a small tumor,
    because the lesion is a negligible fraction of the pixels.

  * tumor_aware_loss -- the same pixel L1, but errors inside the tumor mask are
    up-weighted by ``weight``. Optionally adds a segmentation-consistency term
    so the super-resolved image is penalized when a frozen segmenter disagrees
    with the ground-truth mask. This is the fix we propose.

Both take (pred, target, mask) so the training loop can swap objectives with a
single flag while holding everything else identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distortion_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """Plain pixel L1 (distortion-optimal). ``mask`` is ignored, kept for a
    common signature with tumor_aware_loss."""
    return F.l1_loss(pred, target)


def tumor_aware_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, weight: float = 20.0) -> torch.Tensor:
    """Lesion-weighted pixel L1.

    Per-pixel L1 is multiplied by (1 + weight * mask), so an error inside the
    tumor costs (1 + weight) times as much as the same error in healthy tissue.
    This directly opposes the pseudo-normalization failure, where the model
    would otherwise smooth a small lesion away for a negligible PSNR cost.

    Args:
        pred, target: (B, 1, H, W) in [0, 1].
        mask: (B, 1, H, W) tumor mask in {0, 1}.
        weight: extra weight applied inside the tumor.
    """
    per_pixel = (pred - target).abs()
    w = 1.0 + weight * mask
    return (per_pixel * w).sum() / w.sum()


def seg_consistency_loss(pred: torch.Tensor, mask: torch.Tensor,
                         segmenter: torch.nn.Module) -> torch.Tensor:
    """Optional term: penalize the SR output when a frozen segmenter's
    prediction on it disagrees with the ground-truth tumor mask.

    The segmenter is used in eval mode with gradients flowing back into ``pred``
    only (its own parameters are frozen by the caller). Encourages SR outputs
    that keep the tumor segmentable.
    """
    logits = segmenter(pred)
    return F.binary_cross_entropy_with_logits(logits, mask)


def make_sr_loss(kind: str, weight: float = 20.0,
                 segmenter: torch.nn.Module | None = None,
                 seg_lambda: float = 0.5):
    """Return a callable loss(pred, target, mask) for the chosen objective.

    kind:
        'distortion'  -> pixel L1 baseline.
        'tumor_aware' -> lesion-weighted L1, plus optional seg-consistency
                         when a ``segmenter`` is provided.
    """
    if kind == "distortion":
        return lambda pred, target, mask: distortion_loss(pred, target, mask)

    if kind == "tumor_aware":
        def _loss(pred, target, mask):
            loss = tumor_aware_loss(pred, target, mask, weight=weight)
            if segmenter is not None:
                loss = loss + seg_lambda * seg_consistency_loss(pred, mask, segmenter)
            return loss
        return _loss

    raise ValueError(f"unknown SR loss kind {kind!r}")
