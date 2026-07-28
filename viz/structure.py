"""PatientVolume: the 3D data container consumed by the analyzer/renderer.

Vendored from neuro-voxel (src/core/structure.py), with the ``__repr__``
indentation bug fixed so it is actually a method of the dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class PatientVolume:
    """Stores all MRI data and metadata for one patient / case.

    Arrays are indexed (D, H, W). ``spacing`` is the physical voxel size in mm
    along each axis, used to report tumor volume in cm3.
    """

    id: str                                      # case identifier
    modalities: Dict[str, np.ndarray]            # e.g. {"t1": (D,H,W) float array}
    mask: Optional[np.ndarray]                   # (D,H,W) integer label volume
    affine: Optional[np.ndarray] = None          # 4x4 voxel->world matrix
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __repr__(self) -> str:
        mods = list(self.modalities.keys())
        has_mask = "Yes" if self.mask is not None else "No"
        shape = self.modalities[mods[0]].shape if mods else "()"
        return (f"<PatientVolume id={self.id} modalities={mods} "
                f"mask={has_mask} shape={shape}>")
