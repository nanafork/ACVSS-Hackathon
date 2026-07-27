"""Offscreen 3D rendering of the SR safety result, the neuro-voxel way.

Uses the vendored ``viz`` analyzer (PyVista marching cubes + eye-dome lighting)
to turn the bridge's ``PatientVolume`` objects into:

  * ``brain3d_compare.png`` -- side by side: distortion-optimal vs tumor-aware,
    each a glass brain with the ground-truth tumor (cyan) and that model's
    predicted tumor (red = distortion, green = tumor-aware);
  * ``brain3d_rotate.gif`` -- one orbiting scene with all three tumors overlaid.

Everything renders offscreen (no window), so it works on a headless machine and
for a screen-free screen recording.
"""

from __future__ import annotations

import io

import numpy as np

from viz import VolumeAnalyzer

# RISE palette, brightened for a deep-viewport background (framed by the light
# RISE card on the page). Page dots use the darker base hues; these are the
# on-dark variants of the same blue / green / red.
BG_COLOR = "#0C1220"         # deep near-black navy viewport
BRAIN_COLOR = "#D8DBE4"      # light steel glass brain
TRUE_COLOR = "#5B9DF9"       # accent blue -- ground-truth tumor
TA_COLOR = "#35C089"         # good green  -- tumor-aware prediction (safe)
DI_COLOR = "#E8694A"         # low red     -- distortion prediction (erased)


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
        p.add_mesh(brain, color=BRAIN_COLOR, opacity=0.17, smooth_shading=True)


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
    gif = render_rotate_gif(patients, vols)
    print("wrote", list(pngs.values()), "and", gif)
