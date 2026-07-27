"""Train the segmenter and both SR models, then save a checkpoint.

Synthetic data by default so it runs anywhere with no download. For a real run,
pass --brats and --root.

    python scripts/train_demo.py                      # synthetic, quick
    python scripts/train_demo.py --brats --root /path/to/BraTS
    python scripts/train_demo.py --brats --root /path/to/BraTS \
        --max-cases 120 --select tumor --cache cache/brats128.npz --evaluate

BraTS notes:
  * The first --brats run decodes every volume once and, with --cache, writes
    the extracted slices to an .npz. Later runs load that file in seconds, so a
    disconnected Colab session does not pay the extraction cost again.
  * Train/test are split by case, never by slice, so no scan contributes to
    both sides.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Subset

from src.checkpoint import save_models
from src.data import make_dataset
from src.evaluate import evaluate_pipeline, format_results
from src.losses import make_sr_loss
from src.models import count_params, seg_unet, sr_unet
from src.train import train_segmenter, train_sr


def build_datasets(args):
    """Return (train_ds, test_ds) for whichever data source was requested."""
    if not args.brats:
        train_ds = make_dataset("synthetic", n=args.n, size=args.size, seed=1)
        test_ds = make_dataset("synthetic", n=max(16, args.n // 5),
                               size=args.size, seed=999)
        return train_ds, test_ds

    if not args.root and not (args.cache and os.path.exists(args.cache)):
        raise SystemExit("--brats needs --root (or an existing --cache file)")

    full = make_dataset(
        "brats", root=args.root, modality=args.modality, size=args.size,
        slices_per_case=args.slices_per_case, min_tumor_pixels=args.min_tumor_pixels,
        max_cases=args.max_cases, select=args.select, normalize=args.normalize,
        cache_path=args.cache or None,
    )
    train_idx, test_idx = full.split_by_case(val_frac=args.test_frac, seed=args.seed)
    return Subset(full, train_idx), Subset(full, test_idx)


def main():
    ap = argparse.ArgumentParser()
    # data source
    ap.add_argument("--brats", action="store_true", help="use real BraTS data")
    ap.add_argument("--root", default="", help="BraTS folder (one subfolder per case)")
    ap.add_argument("--modality", default="t1c", choices=["t1c", "t1", "t2", "flair"])
    ap.add_argument("--max-cases", type=int, default=None,
                    help="only read the first N cases (quick runs)")
    ap.add_argument("--slices-per-case", type=int, default=12)
    ap.add_argument("--min-tumor-pixels", type=int, default=20)
    ap.add_argument("--select", default="middle", choices=["middle", "tumor"],
                    help="'tumor' keeps the slices with the largest lesion area")
    ap.add_argument("--normalize", default="slice", choices=["slice", "volume"])
    ap.add_argument("--cache", default="", help="path to an .npz slice cache")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of CASES held out for evaluation")
    ap.add_argument("--n", type=int, default=240, help="synthetic slices")
    # forward model
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.03)
    # training
    ap.add_argument("--seg-epochs", type=int, default=10)
    ap.add_argument("--sr-epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--weight", type=float, default=40.0,
                    help="lesion up-weighting in the tumor-aware loss")
    ap.add_argument("--seed", type=int, default=0)
    # output
    ap.add_argument("--out", default="checkpoints/demo.pt")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the safety comparison on the held-out split")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train_ds, test_ds = build_datasets(args)
    print(f"device={device} train={len(train_ds)} test={len(test_ds)} size={args.size}")

    seg = seg_unet(base=args.base)
    print(f"seg U-Net params: {count_params(seg):,}")
    train_segmenter(seg, train_ds, epochs=args.seg_epochs, bs=args.bs, device=device)
    for p in seg.parameters():
        p.requires_grad_(False)

    sr_d = sr_unet(base=args.base, dropout=0.2)
    sr_t = sr_unet(base=args.base, dropout=0.2)
    print(f"SR U-Net params: {count_params(sr_d):,}")
    train_sr(sr_d, train_ds, make_sr_loss("distortion"), factor=args.factor,
             sigma=args.sigma, epochs=args.sr_epochs, bs=args.bs, device=device,
             tag="sr-distortion", seed=args.seed + 1000)
    train_sr(sr_t, train_ds, make_sr_loss("tumor_aware", weight=args.weight),
             factor=args.factor, sigma=args.sigma, epochs=args.sr_epochs,
             bs=args.bs, device=device, tag="sr-tumor-aware", seed=args.seed + 1000)

    save_models(args.out, seg, sr_d, sr_t,
                meta={"size": args.size, "factor": args.factor, "sigma": args.sigma,
                      "base": args.base, "weight": args.weight,
                      "kind": "brats" if args.brats else "synthetic"})
    print("saved", args.out)

    if args.evaluate:
        results = evaluate_pipeline(
            {"distortion": sr_d, "tumor_aware": sr_t}, seg, test_ds,
            factor=args.factor, sigma=args.sigma, device=device, mc_passes=10,
            size_edges=(50, 200))
        print("\n=== Held-out results ===")
        print(format_results(results))


if __name__ == "__main__":
    main()
