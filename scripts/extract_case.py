"""Pull one real case out of MSD as a compact 3D volume for the demo scene.

The slice cache that training uses is a pile of independent 2D slices, which is
fine for training but cannot be stacked back into a brain: it mixes patients and
keeps only a dozen slices each. The 3D demo needs one patient, contiguous in z,
so this writes exactly that.

Kept small on purpose. A full case is 240x240x155 and the renderer only needs
enough slices to read as a head, so we center-crop in plane and take a
contiguous run of slices around the tumor.

    python scripts/extract_case.py --root Task01_BrainTumour --case BRATS_066 \
        --out data/real_case.npz --size 128 --depth 64 --region et
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import _norm01, center_crop  # noqa: E402

MODALITY_CHANNEL = {"flair": 0, "t1": 1, "t1c": 2, "t2": 3}


def region_mask(seg, region):
    if region == "wt":
        return (seg > 0).astype(np.uint8)
    if region == "tc":
        return np.isin(seg, (2, 3)).astype(np.uint8)
    return (seg == 3).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", default="", help="e.g. BRATS_066; default = the "
                    "case with the most enhancing tumor, which renders best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--region", default="et", choices=["wt", "tc", "et"])
    ap.add_argument("--modality", default="t1c")
    ap.add_argument("--search", type=int, default=40, help="cases to scan when "
                    "picking automatically")
    args = ap.parse_args()

    import nibabel as nib
    ch = MODALITY_CHANNEL[args.modality]
    img_dir = os.path.join(args.root, "imagesTr")
    lbl_dir = os.path.join(args.root, "labelsTr")

    if args.case:
        name = args.case if args.case.endswith(".nii.gz") else args.case + ".nii.gz"
    else:
        # Pick the case with the largest lesion: a bigger, more connected tumor
        # makes a legible 3D surface, and this is a display choice only.
        best, best_n = None, -1
        cands = sorted(f for f in os.listdir(lbl_dir) if not f.startswith("._"))
        for f in cands[:args.search]:
            seg = np.asarray(nib.load(os.path.join(lbl_dir, f)).dataobj)
            n = int(region_mask(seg, args.region).sum())
            if n > best_n:
                best, best_n = f, n
        name = best
        print(f"auto-selected {name} ({best_n:,} {args.region} voxels)")

    seg_full = np.asarray(nib.load(os.path.join(lbl_dir, name)).dataobj)
    mask_full = region_mask(seg_full, args.region)
    vol_obj = nib.load(os.path.join(img_dir, name)).dataobj

    # Center the slice run on the tumor so the lesion is inside the rendered slab.
    zs = np.where(mask_full.sum(axis=(0, 1)) > 0)[0]
    cz = int(zs.mean()) if len(zs) else seg_full.shape[2] // 2
    half = args.depth // 2
    z0 = max(0, min(cz - half, seg_full.shape[2] - args.depth))
    z1 = z0 + args.depth

    vol = np.stack([center_crop(_norm01(np.asarray(vol_obj[:, :, z, ch], dtype=np.float32)),
                                args.size) for z in range(z0, z1)])
    mask = np.stack([center_crop(mask_full[:, :, z].astype(np.float32), args.size)
                     for z in range(z0, z1)]).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, vol=vol.astype(np.float16), mask=mask,
                        case=name, region=args.region)
    print(f"wrote {args.out}: vol {vol.shape}, {int(mask.sum()):,} lesion voxels, "
          f"slices z={z0}..{z1}")


if __name__ == "__main__":
    main()
