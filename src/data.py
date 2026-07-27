"""Data: 2D slice pairs for the tumor-aware super-resolution pipeline.

Two sources:
  * BraTSSliceDataset  -- reads BraTS .nii.gz volumes (real data, on Kaggle).
    nibabel is imported lazily so the rest of the pipeline works without it.
  * SyntheticSliceDataset -- procedurally generated brain-like slices with
    tumor masks, so the whole pipeline can be trained and tested locally with
    no download. Useful for CI / smoke tests.

Every sample is a dict with:
    hr   : (1, H, W) float32 high-resolution image in [0, 1]  (the answer key)
    mask : (1, H, W) float32 tumor mask in {0, 1}
The degraded low-resolution input is produced on the fly by src.degrade so we
always keep the ground-truth image and mask.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _norm01(x: np.ndarray) -> np.ndarray:
    """Scale an image to [0, 1] robustly (99th percentile to limit outliers)."""
    x = x.astype(np.float32)
    lo = float(x.min())
    hi = float(np.percentile(x, 99.0))
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def center_crop(x: np.ndarray, size: int) -> np.ndarray:
    """Center-crop (or pad) a 2D array to (size, size)."""
    h, w = x.shape
    out = np.zeros((size, size), dtype=x.dtype)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    dh, dw = max(0, (size - h) // 2), max(0, (size - w) // 2)
    ch, cw = min(size, h), min(size, w)
    out[dh:dh + ch, dw:dw + cw] = x[sh:sh + ch, sw:sw + cw]
    return out


class SyntheticSliceDataset(Dataset):
    """Procedural brain-like slices with tumors. No external data needed.

    Each slice is a smooth elliptical "brain" with a few bright/dark blobs for
    texture and one or more circular "tumors". Tumor radius is sampled to span
    a range of sizes so lesion-size stratification is meaningful.
    """

    def __init__(self, n: int = 64, size: int = 128, seed: int = 0,
                 tumor_frac: float = 0.9):
        self.n = n
        self.size = size
        self.seed = seed
        self.tumor_frac = tumor_frac

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 100003 + idx)
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cy, cx = s / 2, s / 2
        img = np.zeros((s, s), dtype=np.float32)

        # Skull/brain ellipse.
        ry, rx = s * 0.42, s * 0.34
        brain = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        img[brain] = 0.55

        # A few anatomy-like blobs for texture.
        for _ in range(rng.integers(4, 8)):
            by, bx = rng.uniform(0.25, 0.75, size=2) * s
            br = rng.uniform(0.05, 0.13) * s
            val = rng.uniform(0.15, 0.35) * (1 if rng.random() > 0.5 else -1)
            blob = ((yy - by) ** 2 + (xx - bx) ** 2) <= br ** 2
            img[blob & brain] += val

        mask = np.zeros((s, s), dtype=np.float32)
        if rng.random() < self.tumor_frac:
            for _ in range(rng.integers(1, 3)):
                ty, tx = rng.uniform(0.3, 0.7, size=2) * s
                # Radius spans small..large so lesion-size analysis is possible.
                tr = rng.uniform(0.02, 0.12) * s
                tum = ((yy - ty) ** 2 + (xx - tx) ** 2) <= tr ** 2
                tum &= brain
                img[tum] = 0.95  # enhancing-like bright core
                mask[tum] = 1.0

        img = np.clip(img, 0.0, 1.0)
        hr = torch.from_numpy(img)[None]
        m = torch.from_numpy(mask)[None]
        return {"hr": hr, "mask": m}


class BraTSSliceDataset(Dataset):
    """2D axial slices from BraTS .nii.gz volumes.

    Expects the standard BraTS layout, one folder per case containing files
    that end with the modality tag, e.g. ``*_t1c.nii.gz`` (2023+) or
    ``*_t1ce.nii.gz`` (<=2021) and a ``*_seg.nii.gz`` label volume.

    We take a single modality (default the contrast-enhanced T1) and the middle
    axial slices, where the anatomy and tumor are clearest. Tumor labels are
    binarized (any positive label -> tumor); the WT/TC/ET split lives in
    metrics.py for evaluation.
    """

    MOD_TAGS = {
        "t1c": ["t1c", "t1ce"],
        "t1": ["t1n", "t1"],
        "t2": ["t2w", "t2"],
        "flair": ["t2f", "flair"],
    }

    def __init__(self, root: str, modality: str = "t1c", size: int = 128,
                 slices_per_case: int = 12, min_tumor_pixels: int = 0):
        self.root = root
        self.size = size
        self.tags = self.MOD_TAGS[modality]
        self.index: list[tuple[str, str, int]] = []  # (img_path, seg_path, z)

        case_dirs = sorted(
            d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
        )
        for d in case_dirs:
            img_path = self._find(d, self.tags)
            seg_path = self._find(d, ["seg"])
            if img_path is None or seg_path is None:
                continue
            z0, z1 = self._middle_slice_range(seg_path, slices_per_case)
            import nibabel as nib  # lazy: only needed for real data
            seg = nib.load(seg_path).get_fdata()
            for z in range(z0, z1):
                if min_tumor_pixels and (seg[:, :, z] > 0).sum() < min_tumor_pixels:
                    continue
                self.index.append((img_path, seg_path, z))
        if not self.index:
            raise RuntimeError(f"No BraTS slices found under {root!r}")

    @staticmethod
    def _find(case_dir: str, tags: list[str]) -> str | None:
        for tag in tags:
            hits = glob.glob(os.path.join(case_dir, f"*{tag}.nii*"))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _middle_slice_range(seg_path: str, k: int) -> tuple[int, int]:
        import nibabel as nib
        depth = nib.load(seg_path).shape[2]
        mid = depth // 2
        half = k // 2
        return max(0, mid - half), min(depth, mid + half)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        import nibabel as nib
        img_path, seg_path, z = self.index[idx]
        img = nib.load(img_path).get_fdata()[:, :, z]
        seg = nib.load(seg_path).get_fdata()[:, :, z]
        img = center_crop(_norm01(np.asarray(img)), self.size)
        mask = center_crop((np.asarray(seg) > 0).astype(np.float32), self.size)
        hr = torch.from_numpy(img.astype(np.float32))[None]
        m = torch.from_numpy(mask.astype(np.float32))[None]
        return {"hr": hr, "mask": m}


class CachedSliceDataset(Dataset):
    """Pre-extracted real slices from an ``.npz`` written by prepare_msd.py.

    Holds ``hr`` (float16), ``mask`` (uint8) and ``case_id`` (int32) in memory.
    A few thousand 128x128 slices is well under a gigabyte, and serving them
    from RAM keeps the GPU fed instead of waiting on nibabel.

    ``split`` divides the data **by case, never by slice**. Adjacent axial
    slices of one patient are nearly duplicates, so a random slice-level split
    would put near-copies of a training image in the test set and report an
    optimistic score. Cases are partitioned on a hash of the case id, so the
    same case always lands on the same side regardless of load order.
    """

    def __init__(self, path: str, split: str = "all", test_frac: float = 0.2):
        d = np.load(path)
        hr, mask, case_id = d["hr"], d["mask"], d["case_id"]

        if split != "all":
            cases = np.unique(case_id)
            n_test = max(1, int(round(len(cases) * test_frac)))
            # Deterministic case-level partition, independent of ordering.
            order = np.argsort([hash((int(c), 20260727)) for c in cases])
            test_cases = set(cases[order[:n_test]].tolist())
            want_test = split == "test"
            keep = np.array([(int(c) in test_cases) == want_test for c in case_id])
            hr, mask, case_id = hr[keep], mask[keep], case_id[keep]

        self.hr = hr
        self.mask = mask
        self.case_id = case_id

    def __len__(self) -> int:
        return len(self.hr)

    def n_cases(self) -> int:
        return len(np.unique(self.case_id))

    def __getitem__(self, idx: int):
        hr = torch.from_numpy(self.hr[idx].astype(np.float32))[None]
        m = torch.from_numpy(self.mask[idx].astype(np.float32))[None]
        return {"hr": hr, "mask": m}


def make_dataset(kind: str = "synthetic", **kwargs) -> Dataset:
    """Factory: 'synthetic' (local), 'brats' (standard BraTS layout), or
    'cached' (real slices pre-extracted by scripts/prepare_msd.py)."""
    if kind == "synthetic":
        return SyntheticSliceDataset(**kwargs)
    if kind == "brats":
        return BraTSSliceDataset(**kwargs)
    if kind == "cached":
        return CachedSliceDataset(**kwargs)
    raise ValueError(f"unknown dataset kind {kind!r}")
