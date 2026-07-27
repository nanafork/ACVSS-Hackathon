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
    """2D axial slices from BraTS .nii.gz volumes, pre-extracted into memory.

    Expects the standard BraTS layout, one folder per case containing files
    that end with the modality tag, e.g. ``*_t1c.nii.gz`` (2023+) or
    ``*_t1ce.nii.gz`` (<=2021) and a ``*_seg.nii.gz`` label volume.

    We take a single modality (default the contrast-enhanced T1) and a handful
    of axial slices per case. Tumor labels are binarized (any positive label
    -> tumor).

    Why slices are pre-extracted: a BraTS volume decompresses to ~70 MB, and
    .nii.gz is gzip, so pulling one 2D slice costs a full sequential decode of
    the volume. Reading per ``__getitem__`` would re-decode every volume once
    per slice per epoch, which dominates training time by orders of magnitude.
    Instead every volume is decoded exactly once here, the wanted slices are
    cropped to (size, size) and kept as arrays. A 128px, 12-slices-per-case,
    300-case set is ~240 MB, which fits comfortably in RAM.

    Args:
        root: directory containing one folder per case.
        modality: which MRI contrast to read (see MOD_TAGS).
        size: output side length; slices are center-cropped/padded to it.
        slices_per_case: how many axial slices to keep per case.
        min_tumor_pixels: drop slices with fewer tumor pixels than this.
        max_cases: only read the first N case folders (for quick runs).
        select: 'middle' takes the central slices; 'tumor' takes the slices
            with the largest tumor area, which is usually what a lesion study
            wants.
        normalize: 'slice' scales each slice to [0,1] independently (the
            historical behaviour); 'volume' scales each case by its own
            volume-wide statistics, which keeps intensities comparable across
            slices of the same scan.
        cache_path: optional .npz path. Written after the first build and
            reused on later runs, so a Colab session that disconnects does not
            pay the extraction cost twice.
    """

    MOD_TAGS = {
        "t1c": ["t1c", "t1ce"],
        "t1": ["t1n", "t1"],
        "t2": ["t2w", "t2"],
        "flair": ["t2f", "flair"],
    }

    def __init__(self, root: str, modality: str = "t1c", size: int = 128,
                 slices_per_case: int = 12, min_tumor_pixels: int = 0,
                 max_cases: int | None = None, select: str = "middle",
                 normalize: str = "slice", cache_path: str | None = None,
                 log=print):
        self.root = root
        self.size = size
        self.tags = self.MOD_TAGS[modality]

        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path, log)
            return

        case_dirs = sorted(
            d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
        )
        if max_cases is not None:
            case_dirs = case_dirs[:max_cases]
        if not case_dirs:
            raise RuntimeError(f"No case folders found under {root!r}")

        import nibabel as nib  # lazy: only needed for real data

        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        case_ids: list[int] = []
        skipped = 0

        for ci, d in enumerate(case_dirs):
            img_path = self._find(d, self.tags)
            seg_path = self._find(d, ["seg"])
            if img_path is None or seg_path is None:
                skipped += 1
                continue

            # One decode per volume, float32 rather than get_fdata()'s float64.
            vol = np.asanyarray(nib.load(img_path).dataobj, dtype=np.float32)
            seg = np.asanyarray(nib.load(seg_path).dataobj) > 0
            if vol.ndim != 3 or seg.shape != vol.shape:
                skipped += 1
                continue
            if normalize == "volume":
                vol = _norm01(vol)

            for z in self._pick_slices(seg, slices_per_case, select):
                m2 = seg[:, :, z].astype(np.float32)
                if min_tumor_pixels and m2.sum() < min_tumor_pixels:
                    continue
                img2 = vol[:, :, z]
                if normalize == "slice":
                    img2 = _norm01(img2)
                images.append(center_crop(img2.astype(np.float32), size))
                masks.append(center_crop(m2, size).astype(np.uint8))
                case_ids.append(ci)

            if log and (ci + 1) % 25 == 0:
                log(f"  read {ci + 1}/{len(case_dirs)} cases -> {len(images)} slices")

        if not images:
            raise RuntimeError(
                f"No BraTS slices found under {root!r} "
                f"({skipped} case folders skipped; check the modality tag and layout)")

        self.images = np.stack(images)                       # (N, size, size) f32
        self.masks = np.stack(masks)                         # (N, size, size) u8
        self.case_ids = np.asarray(case_ids, dtype=np.int32)  # (N,)
        if log:
            mb = (self.images.nbytes + self.masks.nbytes) / 1024 ** 2
            log(f"BraTS: {len(self)} slices from {len(set(case_ids))} cases "
                f"({mb:.0f} MB in memory)"
                + (f", {skipped} folders skipped" if skipped else ""))
        if cache_path:
            self._save_cache(cache_path, log)

    # ---------------------------------------------------------------- caching
    def _save_cache(self, path: str, log=print) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, images=self.images, masks=self.masks,
                            case_ids=self.case_ids, size=np.int32(self.size))
        if log:
            log(f"cached dataset -> {path}")

    def _load_cache(self, path: str, log=print) -> None:
        z = np.load(path)
        self.images = z["images"]
        self.masks = z["masks"]
        self.case_ids = z["case_ids"]
        self.size = int(z["size"])
        if log:
            log(f"loaded cached dataset from {path}: {len(self)} slices "
                f"from {len(set(self.case_ids.tolist()))} cases")

    # ------------------------------------------------------------- extraction
    @staticmethod
    def _find(case_dir: str, tags: list[str]) -> str | None:
        for tag in tags:
            hits = glob.glob(os.path.join(case_dir, f"*{tag}.nii*"))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _pick_slices(seg: np.ndarray, k: int, select: str) -> list[int]:
        """Choose which axial slice indices to keep from one case."""
        depth = seg.shape[2]
        if select == "tumor":
            areas = seg.reshape(-1, depth).sum(axis=0)
            order = np.argsort(-areas, kind="stable")
            picked = [int(z) for z in order[:k] if areas[z] > 0]
            return sorted(picked) if picked else []
        if select != "middle":
            raise ValueError(f"unknown slice selection {select!r}")
        mid, half = depth // 2, k // 2
        return list(range(max(0, mid - half), min(depth, mid + half)))

    # ---------------------------------------------------------------- Dataset
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        hr = torch.from_numpy(self.images[idx])[None]
        m = torch.from_numpy(self.masks[idx].astype(np.float32))[None]
        return {"hr": hr, "mask": m}

    def split_by_case(self, val_frac: float = 0.2, seed: int = 0):
        """Split into (train_idx, val_idx) with no case appearing in both.

        Splitting by slice index would put adjacent slices of the same scan on
        both sides, which leaks anatomy and flatters the test numbers. Held-out
        cases are the honest comparison.
        """
        cases = np.unique(self.case_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(cases)
        n_val = max(1, int(round(len(cases) * val_frac)))
        val_cases = set(cases[:n_val].tolist())
        val_idx = [i for i, c in enumerate(self.case_ids.tolist()) if c in val_cases]
        train_idx = [i for i, c in enumerate(self.case_ids.tolist()) if c not in val_cases]
        return train_idx, val_idx


def make_dataset(kind: str = "synthetic", **kwargs) -> Dataset:
    """Factory: 'synthetic' (local) or 'brats' (Kaggle real data)."""
    if kind == "synthetic":
        return SyntheticSliceDataset(**kwargs)
    if kind == "brats":
        return BraTSSliceDataset(**kwargs)
    raise ValueError(f"unknown dataset kind {kind!r}")
