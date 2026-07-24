"""Tumor-aware MRI super-resolution: safety-focused pipeline.

Modules:
    degrade      physics-informed forward degradation (k-space + Rician)
    data         BraTS / synthetic 2D slice datasets
    models       SR and segmentation U-Nets
    losses       distortion-optimal vs tumor-aware SR losses
    metrics      PSNR/SSIM/Dice + erasure/hallucination safety rates
    uncertainty  MC-dropout uncertainty + CPU benchmark
    train        training loops
"""
