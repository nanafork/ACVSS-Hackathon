"""Figures for the demo: comparison panel and uncertainty map.

Kept out of the notebook so the plotting is reusable and testable. matplotlib
is imported lazily so importing the package never requires it.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics import to_mask_np
from .palette import FIG, error_cmap, uncertainty_cmap


def _np(img: torch.Tensor) -> np.ndarray:
    return img.detach().squeeze().cpu().numpy()


# Which hue a predicted mask gets. Keyed on the column name so the 2D panels and
# the 3D renders label the same model with the same color.
_PRED_COLOR = {
    "low-res": FIG["low_res"],
    "distortion": FIG["distortion"],
    "tumor-aware": FIG["tumor_aware"],
    "true HR": FIG["true"],
}


def comparison_figure(hr, mask, lr, sr_outputs: dict, segmenter, save_path=None):
    """Grid: top row images, bottom row predicted tumor masks over each image.

    Columns: low-resolution, each SR output, then the true HR image.
    """
    import matplotlib.pyplot as plt

    segmenter.eval()
    cols = [("low-res", lr)] + list(sr_outputs.items()) + [("true HR", hr)]
    n = len(cols)
    fig, ax = plt.subplots(2, n, figsize=(2.7 * n, 5.6), facecolor="white")
    gt = _np(mask)
    for j, (name, img) in enumerate(cols):
        im = _np(img)
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        # Title in the model's own color, so the column, its fill below, and
        # the 3D panel elsewhere on the page all agree on which model is which.
        ax[0, j].set_title(name, color=_PRED_COLOR.get(name, FIG["low_res"]),
                           fontsize=11, fontweight="semibold", pad=7)
        with torch.no_grad():
            pred = to_mask_np(segmenter(img if img.dim() == 4 else img[None]))
        ax[1, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        # Solid fill in that model's own hue, with the true outline on top, so
        # erasure reads as a missing fill inside an outline that is still there.
        from matplotlib.colors import to_rgba
        fill = np.zeros((*pred.shape, 4))
        fill[...] = to_rgba(_PRED_COLOR.get(name, FIG["low_res"]), alpha=0.62)
        fill[..., 3] *= (pred >= 0.5)
        ax[1, j].imshow(fill)
        ax[1, j].contour(gt, levels=[0.5], colors=[FIG["true"]], linewidths=1.3)
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([])
            for spine in a.spines.values():
                spine.set_edgecolor("#E2E3E1")
    ax[0, 0].set_ylabel("image", fontsize=9, color="#6A6E73")
    ax[1, 0].set_ylabel("segmenter output\n(outline = true tumor)", fontsize=9,
                        color="#6A6E73")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def uncertainty_figure(lr, mean, unc, hr, save_path=None):
    """Show LR input, SR mean, uncertainty (std), and |error| vs HR."""
    import matplotlib.pyplot as plt

    err = (_np(mean) - _np(hr))
    # The uncertainty and error fields are heavy-tailed: a handful of voxels sit
    # far above the bulk, so scaling to the maximum flattens the whole panel to
    # one flat tint. Clip the color scale at a high percentile instead.
    def _hi(a):
        return max(float(np.percentile(a, 99.0)), 1e-6)

    u, e = _np(unc), np.abs(err)
    panels = [("low-res", _np(lr), "gray", None),
              ("SR mean", _np(mean), "gray", None),
              ("uncertainty", u, uncertainty_cmap(on_dark=False), _hi(u)),
              ("|error| vs HR", e, error_cmap(), _hi(e))]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), facecolor="white")
    for a, (name, im, cmap, vmax) in zip(ax, panels):
        m = a.imshow(im, cmap=cmap, vmin=0 if vmax else None, vmax=vmax)
        a.set_title(name, fontsize=11, color="#1A1A1A")
        a.set_xticks([]); a.set_yticks([])
        cb = fig.colorbar(m, ax=a, fraction=0.046, pad=0.04,
                          extend="max" if vmax else "neither")
        cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
