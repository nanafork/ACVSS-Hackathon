"""VolumeAnalyzer: tumor volume (cm3) and marching-cubes surface meshes.

Vendored from neuro-voxel (src/core/analyzer.py). PyVista is imported lazily so
the rest of the project (and ``import viz``) works even when PyVista is absent;
only the mesh methods need it. The volume calculation is pure NumPy.
"""

from __future__ import annotations

import numpy as np

from .structure import PatientVolume


class VolumeAnalyzer:
    def calculate_volume(self, patient: PatientVolume, label_idx: int = 1) -> float:
        """Volume of a given mask label in cm3 = voxel_count * voxel_volume / 1000."""
        if patient.mask is None:
            return 0.0
        voxel_count = int(np.sum(patient.mask == label_idx))
        one_voxel_vol = float(np.prod(patient.spacing))  # mm^3
        return voxel_count * one_voxel_vol / 1000.0

    def get_mesh_from_mask(self, patient: PatientVolume, label_idx: int = 1):
        """Marching-cubes surface for one mask label, smoothed. Needs PyVista."""
        if patient.mask is None:
            return None
        import pyvista as pv

        binary_mask = np.where(patient.mask == label_idx, 1.0, 0.0)
        if binary_mask.sum() == 0:
            return None
        grid = pv.wrap(binary_mask)
        grid.spacing = patient.spacing
        try:
            mesh = grid.contour(isosurfaces=[0.5])
            if mesh.n_points == 0:
                return None
            return mesh.smooth(n_iter=100)
        except Exception as e:  # pragma: no cover - defensive, matches upstream
            print(f"Mesh could not be created (label {label_idx} may be missing): {e}")
            return None

    def get_brain_mesh(self, patient: PatientVolume, modality: str = "t1",
                       iso: float = 0.12):
        """'Glass brain' surface from an image modality. Needs PyVista.

        ``iso`` is an intensity threshold in the modality's own units. Images in
        this project are normalized to [0, 1], so a small positive value picks
        the outer brain surface (background is 0).
        """
        if modality not in patient.modalities:
            return None
        import pyvista as pv

        data = np.ascontiguousarray(patient.modalities[modality], dtype=np.float32)
        grid = pv.wrap(data)
        grid.spacing = patient.spacing
        try:
            mesh = grid.contour(isosurfaces=[iso])
            if mesh.n_points == 0:
                return None
            return mesh.smooth(n_iter=50)
        except Exception as e:  # pragma: no cover
            print(f"Error generating brain surface: {e}")
            return None
