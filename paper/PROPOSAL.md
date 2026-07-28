# Project Proposal: Reconstructing MRI Images from Undersampled Data

**Event:** ACVSS Hackathon (1 week)
**Team:** [add names]

## Summary

An MRI scanner is slow because it has to collect a large amount of measurement data. To scan faster, the scanner can skip part of that data. The problem is that skipping data leaves the reconstructed image blurry and full of artifacts. Our project recovers a clean image from the reduced measurements. We treat this as an inverse problem and train a small neural network to fill in what the scanner did not collect. The whole project runs on a free Kaggle GPU with light training, so no special hardware is needed.

## The Problem in Plain Terms

An MRI scanner does not record the image directly. It records the image in the frequency domain, which is called k-space. A full scan collects every line of k-space, and this takes time. To speed up the scan, the scanner collects only a fraction of the lines, for example one quarter of them. This is called undersampling.

If we simply convert this partial k-space back into an image, we get a poor result with blur and repeating ghost patterns. These are the artifacts we want to remove. Our goal is to take the undersampled data and produce an image that looks close to the full scan.

## Why This Is an Inverse Problem

We know how a clean image turns into measurements. The scanner applies a Fourier transform to the image and then keeps only some of the lines:

```
measurements = mask x FourierTransform x image
```

We are given the measurements and want to recover the image, which is the reverse direction. This is why it is called an inverse problem. The difficulty is that many different images could produce the same partial measurements, so the problem is underdetermined. We solve it by teaching a model what real MRI images look like, so it can pick the most plausible reconstruction.

## Our Approach

1. **Baseline (do nothing smart).** Fill the missing k-space lines with zeros and convert back to an image. This is the blurry starting point that we want to beat.
2. **Main method (learned reconstruction).** Feed the blurry baseline image into a U-Net, which is a standard image-to-image neural network. The U-Net learns to remove the artifacts and output a clean image. We use the U-Net that ships with the official `fastmri` package, so we do not design a network from scratch.
3. **Comparison.** Place the three images side by side: the undersampled input, our reconstruction, and the true full-scan image. We measure the difference with PSNR and SSIM, which are standard image quality scores.

We create the undersampled inputs ourselves by masking k-space, so we always have the true image to compare against.

## Why This Fits One Week and Light Compute

- We do not invent a new model. We import a published one.
- We do not reproduce the original paper, which trained for several days. We train a smaller version on a subset of the data for a few epochs, which takes a couple of hours on a free Kaggle GPU.
- We only need to beat the blurry baseline by a clear margin, not reach state-of-the-art scores.
- Inference on a single image takes seconds.

## Data

We use the fastMRI single-coil knee dataset, which is open and widely cited. The full set is about 90 GB, so we will **not** download all of it. We download a handful of volumes, which gives us a few thousand image slices. That is enough for light training and a convincing demo. As a backup, we can take any clean MRI images and simulate undersampling ourselves, which removes the need for the large download entirely.

## Weekly Plan

| Day | Goal |
|-----|------|
| 1 | Set up the k-space model, the undersampling mask, and the zero-filled baseline. |
| 2 | Load a data subset on Kaggle and confirm the baseline runs end to end. |
| 3 | Import the fastMRI U-Net and run the short training loop. |
| 4 | Add PSNR and SSIM scoring and the side-by-side comparison figure. |
| 5 | Test at two undersampling levels (4x and 8x) and produce a quality-versus-speed curve. |
| 6 | Write up results, prepare figures, and rehearse the demo. |
| 7 | Buffer for polish and fixes. |

Suggested roles: one person on data loading and masking, one on the model and training, one on evaluation and figures, one on the writeup and presentation. Adjust to team size.

## What We Deliver

- A working notebook that reconstructs MRI images from undersampled data.
- A comparison figure showing input, reconstruction, and ground truth.
- PSNR and SSIM numbers that prove the model beats the baseline.
- A short writeup explaining the inverse problem and our results.

## Success Criteria

The reconstruction beats the zero-filled baseline on PSNR and SSIM, and the improvement is visible to the eye at 4x undersampling.

## Stretch Goals

- Add a data-consistency step that keeps the measured k-space lines fixed, which moves us toward the VarNet method (the state-of-the-art extension of this baseline).
- Frame the same method for metal artifact reduction by treating corrupted k-space lines as missing data and filling them in.

## References

- Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI." arXiv:1811.08839.
- Sriram et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction." MICCAI 2020.
- fastMRI code and models: https://github.com/facebookresearch/fastMRI
