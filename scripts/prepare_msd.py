"""Turn the Medical Segmentation Decathlon brain task into a slice cache.

MSD Task01_BrainTumour is BraTS data redistributed without a registration wall,
so it is the fastest route to real anatomy. Its layout differs from the BraTS
release that ``BraTSSliceDataset`` reads: MSD ships one 4-D file per case
(H, W, D, 4 modalities) plus a label volume, rather than a folder of per-
modality files. Rather than reshuffle 7 GB on disk, this script reads MSD
directly and writes the axial slices we actually train on into one compact
``.npz``.

Caching matters for speed. Decoding a gzipped 4-D volume costs seconds, and a
naive Dataset that reopened the case on every __getitem__ would spend the whole
run in nibabel rather than on the GPU.

    python scripts/prepare_msd.py --root /tmp/msd/Task01_BrainTumour \
                                  --out  /tmp/msd/slices_128.npz --size 128

Channel and label conventions come from the MSD dataset.json:
    modalities  0 FLAIR, 1 T1w, 2 T1gd, 3 T2w
    labels      0 background, 1 edema, 2 non-enhancing, 3 enhancing
We take T1gd (the contrast-enhanced T1, matching the ``t1c`` used elsewhere)
and binarize any positive label to a whole-tumor mask.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import _norm01, center_crop  # noqa: E402

MODALITY_CHANNEL = {"flair": 0, "t1": 1, "t1c": 2, "t2": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Task01_BrainTumour directory")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--modality", default="t1c", choices=list(MODALITY_CHANNEL))
    ap.add_argument("--slices-per-case", type=int, default=48,
                    help="axial slices per case, centred on the lesion. The "
                         "old default of 12 used 8%% of each volume and "
                         "dropped 27%% of patients entirely.")
    ap.add_argument("--min-tumor-pixels", type=int, default=20)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all")
    ap.add_argument("--region", default="wt", choices=["wt", "tc", "et"],
                    help="which nested region becomes the binary mask: whole "
                         "tumor (any label), tumor core (non-enhancing + "
                         "enhancing), or enhancing tumor only. The paper's "
                         "claim is about the small enhancing lesion, so 'et' "
                         "is the region that actually tests it.")
    args = ap.parse_args()

    def to_mask(seg_slice):
        if args.region == "wt":
            return (seg_slice > 0).astype(np.float32)
        if args.region == "tc":
            return np.isin(seg_slice, (2, 3)).astype(np.float32)
        return (seg_slice == 3).astype(np.float32)

    import nibabel as nib

    ch = MODALITY_CHANNEL[args.modality]
    img_dir = os.path.join(args.root, "imagesTr")
    lbl_dir = os.path.join(args.root, "labelsTr")
    # MSD ships macOS resource-fork files (._BRATS_001.nii.gz) that nibabel
    # cannot read; skip them rather than crashing halfway through 484 cases.
    cases = sorted(p for p in glob.glob(os.path.join(img_dir, "*.nii.gz"))
                   if not os.path.basename(p).startswith("._"))
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"{len(cases)} cases, modality {args.modality} (channel {ch})")

    hrs, masks, case_ids = [], [], []
    for n, img_path in enumerate(cases):
        name = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, name)
        if not os.path.exists(lbl_path):
            continue
        try:
            seg = np.asarray(nib.load(lbl_path).dataobj)
            vol = nib.load(img_path).dataobj
            depth = seg.shape[2]
            # Centre the window on the lesion, not on the middle of the volume.
            # A fixed mid-axial window silently discarded 27% of patients whose
            # enhancing tumor happened to sit above or below it, and biased the
            # set toward tumors in the central plane. Fall back to the volume
            # centre only when the case has no labelled voxel at all.
            zs = np.where(to_mask(seg).sum(axis=(0, 1)) > 0)[0]
            centre = int(zs.mean()) if len(zs) else depth // 2
            half = args.slices_per_case // 2
            z0 = max(0, min(centre - half, depth - args.slices_per_case))
            z1 = min(depth, z0 + args.slices_per_case)
            for z in range(z0, z1):
                m = to_mask(seg[:, :, z])
                if m.sum() < args.min_tumor_pixels:
                    continue
                # dataobj slicing avoids decoding the whole 4-D volume.
                img = np.asarray(vol[:, :, z, ch], dtype=np.float32)
                hrs.append(center_crop(_norm01(img), args.size).astype(np.float16))
                masks.append(center_crop(m, args.size).astype(np.uint8))
                case_ids.append(n)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__} {e}")
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(cases)} cases -> {len(hrs)} slices", flush=True)

    hr = np.stack(hrs)
    mask = np.stack(masks)
    cid = np.array(case_ids, dtype=np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, hr=hr, mask=mask, case_id=cid)
    frac = float(mask.mean())
    print(f"wrote {args.out}: {hr.shape} slices from {len(set(case_ids))} cases, "
          f"{100 * frac:.2f}% tumor pixels")


if __name__ == "__main__":
    main()
