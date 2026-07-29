"""How much image-quality score does deleting the whole tumor actually cost?

The premise of this project is that PSNR and SSIM cannot express "you erased
the lesion". That is usually argued from the fact that a small tumor is a tiny
fraction of the pixels. This measures it instead, and no model is involved:
take the true high-resolution slice, paint the enhancing lesion out with the
median intensity of the surrounding brain, and score the forgery against the
original.

The resulting number is the honest upper bound on what a pixel-error metric can
notice. If a deliberately tumor-free forgery scores *better* than the
reconstructions we actually train, then the metric prefers erasure, and no
amount of tuning a PSNR-shaped loss will find that out.

Validation split only. The test split is reserved for a single final
evaluation, and this is a property of the data rather than a model result, so
there is no reason to spend test on it.

    python scripts/metric_blindness.py --out results/metric_blindness.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data import make_dataset
from src.metrics import ssim


def measure(path: str, split: str = "val", min_ring: int = 50) -> dict:
    ds = make_dataset("cached", path=path, split=split)

    psnr_full, psnr_brain, ssim_vals, lesion_frac = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        hr = s["hr"][0].numpy().astype(np.float64)
        lesion = s["mask"][0].numpy() > 0.5
        if lesion.sum() == 0:
            continue
        brain = hr > 0.05
        ring = brain & ~lesion
        if ring.sum() < min_ring:
            continue

        forged = hr.copy()
        forged[lesion] = float(np.median(hr[ring]))

        err = (forged - hr) ** 2
        psnr_full.append(10 * np.log10(1.0 / max(float(err.mean()), 1e-12)))
        # Brain-masked as well, because roughly a tenth of a slice is empty
        # background that any reconstruction reproduces for free.
        psnr_brain.append(10 * np.log10(1.0 / max(float(err[brain].mean()), 1e-12)))
        ssim_vals.append(float(ssim(torch.from_numpy(forged)[None, None].float(),
                                    torch.from_numpy(hr)[None, None].float())))
        lesion_frac.append(float(lesion.sum() / max(brain.sum(), 1)))

    def stats(v):
        a = np.asarray(v)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}

    return {
        "what": "true HR slice with the enhancing lesion painted out, scored "
                "against the untouched true HR slice",
        "data": path,
        "split": split,
        "n_slices": len(psnr_full),
        "psnr_whole_image": stats(psnr_full),
        "psnr_brain_masked": stats(psnr_brain),
        "ssim": stats(ssim_vals),
        "lesion_fraction_of_brain": stats(lesion_frac),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slices", default="data/slices_et_full.npz")
    p.add_argument("--split", default="val", choices=["val", "test", "all"])
    p.add_argument("--out", default="results/metric_blindness.json")
    a = p.parse_args()

    out = measure(a.slices, a.split)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
