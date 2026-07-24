"""End-to-end smoke test on synthetic data (no download, CPU-only).

Runs the full pipeline at tiny scale to prove every piece connects and produces
sane numbers:
    synthetic data -> train segmenter -> train 2 SR models
    -> evaluate (PSNR/SSIM/Dice + safety rates + uncertainty) -> CPU benchmark.

This is NOT a scientific run (few epochs, tiny data). It is a correctness check.
Run:  python smoke_test.py
"""

import torch

from src.data import make_dataset
from src.evaluate import evaluate_pipeline, format_results
from src.losses import make_sr_loss
from src.models import count_params, seg_unet, sr_unet
from src.train import train_segmenter, train_sr
from src.uncertainty import cpu_benchmark

torch.manual_seed(0)
DEVICE = "cpu"
SIZE = 96
FACTOR = 4
SIGMA = 0.03


def main():
    train_ds = make_dataset("synthetic", n=48, size=SIZE, seed=1)
    test_ds = make_dataset("synthetic", n=16, size=SIZE, seed=999)
    print(f"train={len(train_ds)} test={len(test_ds)} slices, size={SIZE}")

    # 1. Segmenter on clean HR.
    seg = seg_unet(base=16)
    print(f"seg U-Net params: {count_params(seg):,}")
    train_segmenter(seg, train_ds, epochs=4, bs=8, device=DEVICE)
    for p in seg.parameters():
        p.requires_grad_(False)

    # 2. Two SR models: distortion-optimal vs tumor-aware.
    sr_d = sr_unet(base=16, dropout=0.2)
    sr_t = sr_unet(base=16, dropout=0.2)
    print(f"SR U-Net params: {count_params(sr_d):,}")

    loss_d = make_sr_loss("distortion")
    loss_t = make_sr_loss("tumor_aware", weight=30.0)  # seg-consistency off for speed
    train_sr(sr_d, train_ds, loss_d, factor=FACTOR, sigma=SIGMA, epochs=5,
             bs=8, device=DEVICE, tag="sr-distortion")
    train_sr(sr_t, train_ds, loss_t, factor=FACTOR, sigma=SIGMA, epochs=5,
             bs=8, device=DEVICE, tag="sr-tumor-aware")

    # 3. Evaluate the comparison.
    results = evaluate_pipeline(
        {"distortion": sr_d, "tumor_aware": sr_t}, seg, test_ds,
        factor=FACTOR, sigma=SIGMA, device=DEVICE, mc_passes=8,
        size_edges=(40, 150))
    print("\n=== Results ===")
    print(format_results(results))

    # 4. CPU deployment benchmark.
    bench = cpu_benchmark(sr_d, size=SIZE, reps=10)
    print(f"\nCPU inference: {bench['latency_ms']:.1f} ms/slice, "
          f"{bench['params']:,} params, {bench['param_memory_mb']:.2f} MB")

    # 5. Correctness assertions (validity of metrics, NOT the science; the
    #    scientific effect needs the full-scale run in the notebook).
    import math
    for name in ("lowres", "distortion", "tumor_aware"):
        r = results[name]
        assert math.isfinite(r["psnr"]) and r["psnr"] > 0, f"{name} PSNR invalid"
        assert 0.0 <= r["ssim"] <= 1.0, f"{name} SSIM out of range"
        assert 0.0 <= r["dice"] <= 1.0, f"{name} Dice out of range"
        er = r["safety"]["false_negative_erasure_rate"]
        assert er is None or 0.0 <= er <= 1.0, f"{name} erasure rate out of range"
        assert 0.0 <= r["safety"]["false_positive_detection_rate"] <= 1.0
    au = results["tumor_aware"]["uncertainty_error_auroc"]
    assert math.isnan(au) or 0.0 <= au <= 1.0, "AUROC out of range"
    print("\nOK: pipeline runs end-to-end and produces valid metrics.")
    print("(Numbers are from a tiny 5-epoch synthetic run; use the notebook for real results.)")


if __name__ == "__main__":
    main()
