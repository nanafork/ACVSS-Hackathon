"""3D visualization bridge.

Vendors the reusable core of the neuro-voxel project
(https://github.com/asmarufoglu/neuro-voxel) -- the ``PatientVolume`` data
structure and the ``VolumeAnalyzer`` (tumor volume in cm3 + marching-cubes
surface mesh) -- so this super-resolution safety study can render its own
model output as an interactive 3D brain/tumor scene.

Bugs fixed relative to upstream:
  * brats_loader mask glob ``*_seg.nii`` -> ``*_seg.nii*`` (also matched .nii.gz)
    -- not vendored here (we feed volumes directly), noted for the record.
  * structure.py ``__repr__`` was defined at module scope (wrong indent) so it
    never attached to the dataclass; fixed to be a real method.
"""

from .structure import PatientVolume
from .analyzer import VolumeAnalyzer

__all__ = ["PatientVolume", "VolumeAnalyzer"]
