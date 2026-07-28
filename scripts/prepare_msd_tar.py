"""Build the slice cache straight out of the MSD tar, without unpacking it.

The full archive is 7.6 GB and unpacking it needs another 7.6 GB. On a disk with
13 GB free that does not fit, so this reads the tar member by member: extract one
case to a temp file, take its slices, delete it, move on. Peak extra disk is one
case (~150 MB) rather than the whole dataset.

Same conventions and same fixes as prepare_msd.py:
  * the slice window is centred on the lesion, not the middle of the volume
  * the min-tumor-pixels check runs AFTER the centre crop

    python scripts/prepare_msd_tar.py --tar data/msd/Task01.tar \
        --out data/slices_et_full.npz --size 128 --region et --slices-per-case 48
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import _norm01, center_crop  # noqa: E402

MODALITY_CHANNEL = {"flair": 0, "t1": 1, "t1c": 2, "t2": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--modality", default="t1c", choices=list(MODALITY_CHANNEL))
    ap.add_argument("--slices-per-case", type=int, default=48)
    ap.add_argument("--min-tumor-pixels", type=int, default=10)
    ap.add_argument("--region", default="et", choices=["wt", "tc", "et"])
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    import nibabel as nib
    ch = MODALITY_CHANNEL[args.modality]

    def to_mask(seg):
        if args.region == "wt":
            return (seg > 0).astype(np.float32)
        if args.region == "tc":
            return np.isin(seg, (2, 3)).astype(np.float32)
        return (seg == 3).astype(np.float32)

    tf = tarfile.open(args.tar, "r")
    # Index the members once so image and label can be paired by basename.
    imgs, lbls = {}, {}
    for m in tf.getmembers():
        b = os.path.basename(m.name)
        if not m.isfile() or b.startswith("._") or not b.endswith(".nii.gz"):
            continue
        if "/imagesTr/" in m.name:
            imgs[b] = m
        elif "/labelsTr/" in m.name:
            lbls[b] = m
    names = sorted(set(imgs) & set(lbls))
    if args.max_cases:
        names = names[:args.max_cases]
    print(f"{len(names)} cases in {args.tar}", flush=True)

    def load(member, tmpdir, tag):
        path = os.path.join(tmpdir, tag + ".nii.gz")
        with tf.extractfile(member) as src, open(path, "wb") as dst:
            while chunk := src.read(1 << 22):
                dst.write(chunk)
        return path

    hrs, masks, case_ids = [], [], []
    for n, name in enumerate(names):
        try:
            with tempfile.TemporaryDirectory() as td:
                seg = np.asarray(nib.load(load(lbls[name], td, "seg")).dataobj)
                vol = nib.load(load(imgs[name], td, "img")).dataobj
                depth = seg.shape[2]
                zs = np.where(to_mask(seg).sum(axis=(0, 1)) > 0)[0]
                centre = int(zs.mean()) if len(zs) else depth // 2
                half = args.slices_per_case // 2
                z0 = max(0, min(centre - half, depth - args.slices_per_case))
                z1 = min(depth, z0 + args.slices_per_case)
                for z in range(z0, z1):
                    m = center_crop(to_mask(seg[:, :, z]), args.size)
                    if m.sum() < args.min_tumor_pixels:
                        continue
                    img = np.asarray(vol[:, :, z, ch], dtype=np.float32)
                    hrs.append(center_crop(_norm01(img), args.size).astype(np.float16))
                    masks.append(m.astype(np.uint8))
                    case_ids.append(n)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__} {e}", flush=True)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(names)} cases -> {len(hrs)} slices", flush=True)
    tf.close()

    hr = np.stack(hrs)
    mask = np.stack(masks)
    cid = np.array(case_ids, dtype=np.int32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, hr=hr, mask=mask, case_id=cid)
    print(f"wrote {args.out}: {hr.shape} from {len(set(case_ids))} cases, "
          f"{100 * float(mask.mean()):.2f}% tumor pixels")


if __name__ == "__main__":
    main()
