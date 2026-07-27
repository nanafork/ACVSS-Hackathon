"""Offscreen 3D rendering of the SR safety result, the neuro-voxel way.

Uses the vendored ``viz`` analyzer (PyVista marching cubes + eye-dome lighting)
to turn the bridge's ``PatientVolume`` objects into:

  * ``brain3d_compare.png`` -- one panel per version: a glass brain with the
    ground-truth tumor ghosted in and that version's predicted tumor solid;
  * ``brain3d_uncertainty.png`` -- the tumor-aware MC dropout uncertainty field;
  * ``brain3d_rotate.gif`` -- one orbiting scene with all three tumors overlaid.

Colors come from ``src.palette`` and are validated there rather than chosen by
eye. Do not hardcode hues in this file.

Everything renders offscreen (no window), so it works on a headless machine and
for a screen-free screen recording.
"""

from __future__ import annotations

import io

import numpy as np

from src.palette import (BG_DARK, BRAIN_NEUTRAL, BRAIN_OPACITY, DARK,
                         uncertainty_cmap)
from viz import VolumeAnalyzer

BG_COLOR = BG_DARK
BRAIN_COLOR = BRAIN_NEUTRAL
TRUE_COLOR = DARK["true"]
TA_COLOR = DARK["tumor_aware"]
DI_COLOR = DARK["distortion"]


def _meshes(patients):
    """Marching-cubes meshes shared by the PNG and the GIF."""
    az = VolumeAnalyzer()
    return {
        "brain": az.get_brain_mesh(patients["tumor-aware"], modality="t1", iso=0.12),
        "true": az.get_mesh_from_mask(patients["true"]),
        "tumor-aware": az.get_mesh_from_mask(patients["tumor-aware"]),
        "distortion": az.get_mesh_from_mask(patients["distortion"]),
    }


def _add_brain(p, brain):
    if brain is not None and brain.n_points:
        p.add_mesh(brain, color=BRAIN_COLOR, opacity=BRAIN_OPACITY,
                   smooth_shading=True)


def _add_tumor(p, mesh, color, opacity=1.0, style="surface"):
    if mesh is not None and mesh.n_points:
        p.add_mesh(mesh, color=color, opacity=opacity, style=style,
                   smooth_shading=True)
        return True
    return False


def _render_one(m, pred_mesh, pred_color, out, size=(580, 560)):
    """Render a single glass-brain scene with one predicted tumor + true ghost.

    No text is baked into the image. Labels live in the HTML so they use the
    page fonts. Single-view offscreen rendering is used (reliable), rather than
    PyVista subplots, which can leave a panel blank when captured headless.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background(BG_COLOR)
    _add_brain(p, m["brain"])
    # Faint "ghost" of the true tumor so the viewer sees where it should be.
    _add_tumor(p, m["true"], TRUE_COLOR, opacity=0.16)
    _add_tumor(p, pred_mesh, pred_color, opacity=1.0)
    try:
        p.enable_eye_dome_lighting()
    except Exception:
        pass
    p.camera_position = "iso"
    p.camera.zoom(1.5)  # fill the frame so the panel is not mostly empty margin
    p.screenshot(out)
    p.close()
    return out


def render_compare_png(patients, vols, prefix="brain3d"):
    """Three single-view PNGs: ground truth, tumor-aware, distortion."""
    m = _meshes(patients)
    outs = {
        "true": _render_one(m, m["true"], TRUE_COLOR, f"{prefix}_true.png"),
        "tumor-aware": _render_one(m, m["tumor-aware"], TA_COLOR,
                                   f"{prefix}_tumor_aware.png"),
        "distortion": _render_one(m, m["distortion"], DI_COLOR,
                                  f"{prefix}_distortion.png"),
    }
    return outs


def render_uncertainty_png(patients, out="brain3d_uncertainty.png",
                           size=(580, 560), floor=0.35, pct=99.5, iso=0.12):
    """Volume-render the tumor-aware MC dropout uncertainty inside the brain.

    The scalar field is normalized per case, so the image shows *where* the
    model is least sure rather than an absolute noise level. Normalization uses
    a high percentile rather than the maximum: a handful of voxels sit roughly
    60x above the 99th percentile, and dividing by that peak crushes the entire
    field to near zero, leaving a nearly black panel. Values above the
    percentile are clipped to the top of the ramp.

    Everything below ``floor`` is made fully transparent, otherwise the whole
    brain glows faintly and the hot spots stop reading.

    Uses the sequential violet ramp, ordered so low uncertainty sinks into the
    background and the uncertain regions are the bright ones.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    pv_case = patients["tumor-aware"]
    unc = pv_case.modalities.get("uncertainty")
    if unc is None:
        return None

    # Restrict to tissue. About 60% of the raw uncertainty mass sits in the
    # empty background, where the network's dropout variance is real but
    # meaningless, and it swamps the render. We are asking where the model is
    # unsure *about the brain*, so the air is masked out at the same isovalue
    # the brain mesh uses.
    unc = np.where(pv_case.modalities["t1"] > iso, unc, 0.0)

    peak = float(np.percentile(unc[unc > 0], pct)) if (unc > 0).any() else 0.0
    if peak <= 0:
        return None
    field = np.clip(unc / peak, 0.0, 1.0).astype(np.float32)

    # Axis convention must match the vendored analyzer, which maps array axis 0
    # to VTK x (its brain mesh comes out with the array's z-extent on x). So the
    # grid keeps the array's own axis order and the cell data varies axis 0
    # fastest, which is Fortran order. Getting this wrong renders the field
    # transposed against the brain it is supposed to sit inside.
    grid = pv.ImageData(dimensions=tuple(d + 1 for d in field.shape))
    grid.cell_data["uncertainty"] = field.ravel(order="F")

    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background(BG_COLOR)
    # A fainter shell than the other panels: at the usual opacity the surface
    # occludes the very interior field this panel exists to show.
    brain = _meshes(patients)["brain"]
    if brain is not None and brain.n_points:
        p.add_mesh(brain, color=BRAIN_COLOR, opacity=0.06, smooth_shading=True)
    # Opacity climbs from fully transparent at the floor to solid at the peak.
    ramp = np.clip((np.linspace(0.0, 1.0, 16) - floor) / (1.0 - floor), 0.0, 1.0)
    p.add_volume(grid, scalars="uncertainty", cmap=uncertainty_cmap(on_dark=True),
                 opacity=(ramp * 255).astype(np.uint8), show_scalar_bar=False)
    p.camera_position = "iso"
    p.camera.zoom(1.5)
    p.screenshot(out)
    p.close()
    return out


def render_rotate_gif(patients, vols, out="brain3d_rotate.gif",
                      size=(680, 680), n_frames=36, step=10):
    import imageio.v2 as imageio
    import pyvista as pv
    pv.OFF_SCREEN = True

    m = _meshes(patients)
    p = pv.Plotter(off_screen=True, window_size=size, border=False)
    p.set_background(BG_COLOR)
    _add_brain(p, m["brain"])
    _add_tumor(p, m["true"], TRUE_COLOR, opacity=0.18)
    _add_tumor(p, m["tumor-aware"], TA_COLOR, opacity=1.0)
    _add_tumor(p, m["distortion"], DI_COLOR, opacity=1.0)
    try:
        p.enable_eye_dome_lighting()
    except Exception:
        pass
    p.camera_position = "iso"
    p.camera.zoom(1.3)  # fill the frame while leaving room for the orbit
    p.show(auto_close=False)

    frames = []
    for _ in range(n_frames):
        p.camera.azimuth += step
        p.render()
        frames.append(p.screenshot(return_img=True))
    p.close()
    imageio.mimsave(out, frames, duration=0.08, loop=0)
    return out


if __name__ == "__main__":
    import torch

    from viz_bridge import build_patient_volumes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients, vols = build_patient_volumes(device=device)
    pngs = render_compare_png(patients, vols)
    unc = render_uncertainty_png(patients)
    gif = render_rotate_gif(patients, vols)
    print("wrote", list(pngs.values()), unc, "and", gif)
