"""Build the slice cache straight out of the MSD tar, without unpacking it.

The full archive is 7.6 GB and unpacking it needs another 7.6 GB. On a disk with
limited headroom that does not fit, so this reads the tar member by member:
extract one case to a temp file, take its slices, delete it, move on. Peak extra
disk is one case (~150 MB) rather than the whole dataset.

The work is embarrassingly parallel -- 484 independent gzip streams -- and it is
pure CPU, so on a GPU box it is the step that leaves the accelerator idle. It
runs across a process pool by default. A serial run of the full archive took
about 40 minutes on 20 cores; the pool brings that down to a few.

Determinism matters here and is preserved: ``case_id`` is the index into the
sorted list of case names, not the order results happen to come back in. The
train/val/test split is a hash of that id, so parallelising must not renumber
cases or the splits would silently move between runs.

Same sampling fixes as prepare_msd.py:
  * the slice window is centred on the lesion, not the middle of the volume
  * the min-tumor-pixels check runs AFTER the centre crop

    python scripts/prepare_msd_tar.py --tar data/msd/Task01.tar \
        --out data/slices_et_full.npz --size 128 --region et \
        --slices-per-case 48 --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import _norm01, center_crop  # noqa: E402

MODALITY_CHANNEL = {"flair": 0, "t1": 1, "t1c": 2, "t2": 3}

# Set once per worker process. A tarfile handle cannot be shared across
# processes (they would fight over one file offset), so each worker opens the
# archive itself and keeps it for the life of the process.
_TAR = None
_CFG = None


def _region_mask(seg, region):
    if region == "wt":
        return (seg > 0).astype(np.float32)
    if region == "tc":
        return np.isin(seg, (2, 3)).astype(np.float32)
    return (seg == 3).astype(np.float32)


def _init(tar_path, cfg):
    global _TAR, _CFG
    _TAR = tarfile.open(tar_path, "r")
    _CFG = cfg


def _one_case(task):
    """Extract and slice a single case. Returns (index, hr, mask) or (index, None, None)."""
    idx, img_name, lbl_name = task
    cfg = _CFG
    import nibabel as nib

    def stage(member_name, tmpdir, tag):
        member = _TAR.getmember(member_name)
        path = os.path.join(tmpdir, tag + ".nii.gz")
        with _TAR.extractfile(member) as src, open(path, "wb") as dst:
            while chunk := src.read(1 << 22):
                dst.write(chunk)
        return path

    try:
        with tempfile.TemporaryDirectory() as td:
            seg = np.asarray(nib.load(stage(lbl_name, td, "seg")).dataobj)
            vol = nib.load(stage(img_name, td, "img")).dataobj
            depth = seg.shape[2]
            zs = np.where(_region_mask(seg, cfg["region"]).sum(axis=(0, 1)) > 0)[0]
            centre = int(zs.mean()) if len(zs) else depth // 2
            half = cfg["slices"] // 2
            z0 = max(0, min(centre - half, depth - cfg["slices"]))
            z1 = min(depth, z0 + cfg["slices"])

            hrs, masks = [], []
            for z in range(z0, z1):
                m = center_crop(_region_mask(seg[:, :, z], cfg["region"]), cfg["size"])
                if m.sum() < cfg["min_px"]:
                    continue
                img = np.asarray(vol[:, :, z, cfg["ch"]], dtype=np.float32)
                hrs.append(center_crop(_norm01(img), cfg["size"]).astype(np.float16))
                masks.append(m.astype(np.uint8))
            if not hrs:
                return idx, None, None
            return idx, np.stack(hrs), np.stack(masks)
    except Exception as e:
        print(f"  skip {os.path.basename(img_name)}: {type(e).__name__} {e}", flush=True)
        return idx, None, None


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
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = min(16, cpu_count). Each worker opens its own "
                         "handle on the archive.")
    args = ap.parse_args()

    # Index the archive once, in the parent, so case numbering is fixed before
    # any work is handed out.
    with tarfile.open(args.tar, "r") as tf:
        imgs, lbls = {}, {}
        for m in tf.getmembers():
            b = os.path.basename(m.name)
            if not m.isfile() or b.startswith("._") or not b.endswith(".nii.gz"):
                continue
            if "/imagesTr/" in m.name:
                imgs[b] = m.name
            elif "/labelsTr/" in m.name:
                lbls[b] = m.name
    names = sorted(set(imgs) & set(lbls))
    if args.max_cases:
        names = names[:args.max_cases]

    workers = args.workers or min(16, os.cpu_count() or 4)
    print(f"{len(names)} cases in {args.tar}, {workers} workers", flush=True)

    cfg = {"region": args.region, "size": args.size, "slices": args.slices_per_case,
           "min_px": args.min_tumor_pixels, "ch": MODALITY_CHANNEL[args.modality]}
    tasks = [(i, imgs[n], lbls[n]) for i, n in enumerate(names)]

    parts = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(args.tar, cfg)) as ex:
        for done, (idx, hr, mask) in enumerate(ex.map(_one_case, tasks, chunksize=1), 1):
            if hr is not None:
                parts[idx] = (hr, mask)
            if done % 25 == 0:
                total = sum(len(v[0]) for v in parts.values())
                print(f"  {done}/{len(tasks)} cases -> {total} slices", flush=True)

    # Reassemble in case order, so the output is byte-identical regardless of
    # how many workers ran or what order they finished in.
    hr = np.concatenate([parts[i][0] for i in sorted(parts)])
    mask = np.concatenate([parts[i][1] for i in sorted(parts)])
    cid = np.concatenate([np.full(len(parts[i][0]), i, dtype=np.int32)
                          for i in sorted(parts)])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, hr=hr, mask=mask, case_id=cid)
    print(f"wrote {args.out}: {hr.shape} from {len(parts)} cases, "
          f"{100 * float(mask.mean()):.2f}% tumor pixels")


if __name__ == "__main__":
    main()
