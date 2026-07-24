"""Monte Carlo dropout uncertainty and CPU benchmarking.

mc_predict runs the SR model several times with dropout active and returns the
mean prediction and a per-pixel uncertainty (std across passes). This is the
"safety mechanism" of the project: a map that flags where the model is unsure,
which we then test against where errors actually occur.

cpu_benchmark reports inference latency and memory on CPU only, to check the
pipeline is deployable without a GPU.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .models import enable_mc_dropout


@torch.no_grad()
def mc_predict(model: torch.nn.Module, x: torch.Tensor, passes: int = 10):
    """Monte Carlo dropout inference.

    Args:
        model: SR network (must contain Dropout layers).
        x: (B, 1, H, W) input.
        passes: number of stochastic forward passes.
    Returns:
        mean: (B, 1, H, W) mean prediction.
        unc:  (B, 1, H, W) per-pixel std across passes (the uncertainty map).
    """
    enable_mc_dropout(model)
    outs = torch.stack([model(x) for _ in range(passes)], dim=0)
    return outs.mean(0), outs.std(0)


def uncertainty_error_auroc(unc: np.ndarray, error: np.ndarray) -> float:
    """Does uncertainty predict where errors are? (rank-based AUROC.)

    Treats each pixel as a sample: label 1 if it is an error pixel, 0 otherwise,
    score = uncertainty. Returns AUROC via the Mann-Whitney U statistic, so no
    sklearn dependency. 0.5 = uncertainty is uninformative, 1.0 = perfect.
    """
    u = unc.reshape(-1).astype(np.float64)
    y = (error.reshape(-1) > 0.5)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(u, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(u) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(u, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[y].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cpu_benchmark(model: torch.nn.Module, size: int = 128, reps: int = 20,
                  warmup: int = 3) -> dict:
    """Benchmark single-slice inference on CPU.

    Returns latency (ms) and a rough peak-memory estimate for the model plus a
    single activation, to argue offline / edge deployability.
    """
    model = model.to("cpu").eval()
    x = torch.randn(1, 1, size, size)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        dt = (time.perf_counter() - t0) / reps

    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / (1024 ** 2)  # float32
    return {
        "latency_ms": dt * 1000.0,
        "params": n_params,
        "param_memory_mb": param_mb,
        "device": "cpu",
    }
