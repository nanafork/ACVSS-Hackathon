"""Figures for the demo: comparison panel and uncertainty map.

Kept out of the notebook so the plotting is reusable and testable. matplotlib
is imported lazily so importing the package never requires it.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics import to_mask_np


def _np(img: torch.Tensor) -> np.ndarray:
    return img.detach().squeeze().cpu().numpy()


def comparison_figure(hr, mask, lr, sr_outputs: dict, segmenter, save_path=None):
    """Grid: top row images, bottom row predicted tumor masks over each image.

    Columns: low-resolution, each SR output, then the true HR image.
    """
    import matplotlib.pyplot as plt

    segmenter.eval()
    cols = [("low-res", lr)] + list(sr_outputs.items()) + [("true HR", hr)]
    n = len(cols)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    gt = _np(mask)
    for j, (name, img) in enumerate(cols):
        im = _np(img)
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(name)
        with torch.no_grad():
            pred = to_mask_np(segmenter(img if img.dim() == 4 else img[None]))
        ax[1, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        ax[1, j].imshow(np.ma.masked_where(pred < 0.5, pred), cmap="autumn", alpha=0.6)
        ax[1, j].contour(gt, levels=[0.5], colors="cyan", linewidths=0.8)
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([])
    ax[1, 0].set_ylabel("pred mask (cyan = true)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def uncertainty_figure(lr, mean, unc, hr, save_path=None):
    """Show LR input, SR mean, uncertainty (std), and |error| vs HR."""
    import matplotlib.pyplot as plt

    err = (_np(mean) - _np(hr))
    panels = [("low-res", _np(lr), "gray"),
              ("SR mean", _np(mean), "gray"),
              ("uncertainty", _np(unc), "viridis"),
              ("|error| vs HR", np.abs(err), "magma")]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (name, im, cmap) in zip(ax, panels):
        m = a.imshow(im, cmap=cmap)
        a.set_title(name); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(m, ax=a, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
