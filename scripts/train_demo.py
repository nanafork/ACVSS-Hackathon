"""Train models for the demo and save a checkpoint.

Synthetic data by default so it runs anywhere with no download. For a real demo,
pass --brats and --root. Produces checkpoints/demo.pt.

    python scripts/train_demo.py                 # synthetic, quick
    python scripts/train_demo.py --brats --root /path/to/BraTS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.checkpoint import save_models
from src.data import make_dataset
from src.losses import make_sr_loss
from src.models import seg_unet, sr_unet
from src.train import train_segmenter, train_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats", action="store_true")
    ap.add_argument("--root", default="")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seg-epochs", type=int, default=10)
    ap.add_argument("--sr-epochs", type=int, default=18)
    ap.add_argument("--weight", type=float, default=40.0)
    ap.add_argument("--out", default="checkpoints/demo.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    if args.brats:
        train_ds = make_dataset("brats", root=args.root, modality="t1c",
                                size=args.size, slices_per_case=12, min_tumor_pixels=20)
    else:
        train_ds = make_dataset("synthetic", n=args.n, size=args.size, seed=1)
    print(f"device={device} train={len(train_ds)} size={args.size}")

    seg = seg_unet(base=32)
    train_segmenter(seg, train_ds, epochs=args.seg_epochs, bs=8, device=device)
    for p in seg.parameters():
        p.requires_grad_(False)

    sr_d = sr_unet(base=32, dropout=0.2)
    sr_t = sr_unet(base=32, dropout=0.2)
    train_sr(sr_d, train_ds, make_sr_loss("distortion"), factor=args.factor,
             sigma=args.sigma, epochs=args.sr_epochs, bs=8, device=device, tag="sr-distortion")
    train_sr(sr_t, train_ds, make_sr_loss("tumor_aware", weight=args.weight),
             factor=args.factor, sigma=args.sigma, epochs=args.sr_epochs, bs=8,
             device=device, tag="sr-tumor-aware")

    save_models(args.out, seg, sr_d, sr_t,
                meta={"size": args.size, "factor": args.factor, "sigma": args.sigma,
                      "kind": "brats" if args.brats else "synthetic"})
    print("saved", args.out)


if __name__ == "__main__":
    main()
