"""Unit tests for the pieces a real BraTS run depends on.

Runs in a few seconds on CPU and downloads nothing: the BraTS tests build a
fake case tree of .nii.gz files with the standard naming, so the loader is
exercised end to end without the real dataset.

    python -m pytest tests/ -q        # or: python tests/test_pipeline.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import BraTSSliceDataset
from src.degrade import degrade, degrade_torch
from src.models import sr_unet
from src.uncertainty import mc_predict, uncertainty_error_auroc


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_fake_brats(root: str, n_cases: int = 4, shape=(60, 60, 20)) -> None:
    """Write a minimal BraTS-shaped tree: one folder per case, t1c + seg."""
    import nibabel as nib

    rng = np.random.default_rng(0)
    for c in range(n_cases):
        d = os.path.join(root, f"BraTS20_Training_{c + 1:03d}")
        os.makedirs(d, exist_ok=True)
        vol = rng.uniform(0, 800, size=shape).astype(np.float32)
        seg = np.zeros(shape, dtype=np.uint8)
        # A cube of "tumor" in the middle slices, size varying by case.
        r = 3 + c
        cy, cx, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        seg[cy - r:cy + r, cx - r:cx + r, cz - 4:cz + 4] = 1
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(vol, aff), os.path.join(d, f"BraTS20_Training_{c + 1:03d}_t1c.nii.gz"))
        nib.save(nib.Nifti1Image(seg, aff), os.path.join(d, f"BraTS20_Training_{c + 1:03d}_seg.nii.gz"))


# --------------------------------------------------------------------------- #
# Degradation: torch port must match the numpy reference
# --------------------------------------------------------------------------- #
def test_degrade_parity_noiseless():
    """With sigma=0 the two implementations should agree to float32 precision."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)

    ref = degrade(img, factor=4, sigma=0.0)
    got = degrade_torch(torch.from_numpy(img)[None, None], factor=4, sigma=0.0)
    got = got[0, 0].numpy()

    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-5, np.abs(got - ref).max()


def test_degrade_parity_noise_statistics():
    """With noise on, the distributions should match even though draws differ."""
    rng = np.random.default_rng(1)
    img = rng.uniform(0.2, 0.8, size=(64, 64)).astype(np.float32)
    sigma = 0.05

    ref = np.stack([degrade(img, factor=4, sigma=sigma,
                            rng=np.random.default_rng(100 + i)) for i in range(16)])
    batch = torch.from_numpy(img)[None, None].repeat(16, 1, 1, 1)
    got = degrade_torch(batch, factor=4, sigma=sigma).numpy()[:, 0]

    assert abs(ref.mean() - got.mean()) < 0.01, (ref.mean(), got.mean())
    assert abs(ref.std() - got.std()) < 0.01, (ref.std(), got.std())


def test_degrade_torch_batched_and_clipped():
    x = torch.rand(5, 1, 32, 32)
    out = degrade_torch(x, factor=2, sigma=0.03)
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_kspace_truncation_actually_blurs():
    """Truncating k-space must remove high-frequency energy."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[::2, :] = 1.0  # a high-frequency stripe pattern
    out = degrade(img, factor=4, sigma=0.0)
    assert out.std() < img.std() * 0.5, (out.std(), img.std())


# --------------------------------------------------------------------------- #
# MC dropout must not leak stochastic state to the caller
# --------------------------------------------------------------------------- #
def test_mc_predict_restores_training_state():
    model = sr_unet(base=8, dropout=0.2)
    model.eval()
    x = torch.rand(1, 1, 32, 32)

    before = {name: m.training for name, m in model.named_modules()}
    mc_predict(model, x, passes=3)
    after = {name: m.training for name, m in model.named_modules()}
    assert before == after, "mc_predict changed the model's training flags"


def test_predictions_deterministic_after_mc_predict():
    """The bug this guards: eval-mode output must not change after MC dropout."""
    model = sr_unet(base=8, dropout=0.2)
    model.eval()
    x = torch.rand(1, 1, 32, 32)

    with torch.no_grad():
        first = model(x).clone()
    mc_predict(model, x, passes=3)
    with torch.no_grad():
        second = model(x)

    assert torch.allclose(first, second), (
        "model became stochastic after mc_predict; evaluation metrics would be noise")


def test_mc_predict_uncertainty_is_positive():
    model = sr_unet(base=8, dropout=0.3)
    mean, unc = mc_predict(model, torch.rand(1, 1, 32, 32), passes=8)
    assert mean.shape == unc.shape == (1, 1, 32, 32)
    assert float(unc.mean()) > 0.0, "dropout was not active during MC passes"


def test_auroc_ranks_correctly():
    unc = np.array([0.1, 0.2, 0.9, 0.8])
    err = np.array([0.0, 0.0, 1.0, 1.0])
    assert abs(uncertainty_error_auroc(unc, err) - 1.0) < 1e-9
    assert abs(uncertainty_error_auroc(unc, 1 - err) - 0.0) < 1e-9
    # All-ties must give chance level, not a crash.
    assert abs(uncertainty_error_auroc(np.ones(4), err) - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# BraTS loader
# --------------------------------------------------------------------------- #
def test_brats_loads_and_shapes():
    tmp = tempfile.mkdtemp()
    try:
        make_fake_brats(tmp, n_cases=4)
        ds = BraTSSliceDataset(tmp, size=48, slices_per_case=6,
                               min_tumor_pixels=0, log=None)
        assert len(ds) == 4 * 6, len(ds)
        s = ds[0]
        assert s["hr"].shape == (1, 48, 48)
        assert s["mask"].shape == (1, 48, 48)
        assert s["hr"].dtype == torch.float32 and s["mask"].dtype == torch.float32
        assert 0.0 <= float(s["hr"].min()) and float(s["hr"].max()) <= 1.0
        assert set(np.unique(s["mask"].numpy()).tolist()) <= {0.0, 1.0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_brats_max_cases_and_tumor_selection():
    tmp = tempfile.mkdtemp()
    try:
        make_fake_brats(tmp, n_cases=5)
        ds = BraTSSliceDataset(tmp, size=48, slices_per_case=4, max_cases=2,
                               select="tumor", min_tumor_pixels=1, log=None)
        assert len(set(ds.case_ids.tolist())) == 2
        # 'tumor' selection means every kept slice contains lesion.
        assert all(ds.masks[i].sum() > 0 for i in range(len(ds)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_brats_cache_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        make_fake_brats(tmp, n_cases=3)
        cache = os.path.join(tmp, "cache", "slices.npz")
        first = BraTSSliceDataset(tmp, size=48, slices_per_case=4,
                                  cache_path=cache, log=None)
        assert os.path.exists(cache)

        # Second build must come from the cache: point it at an empty dir, which
        # would raise if it tried to read volumes again.
        empty = tempfile.mkdtemp()
        second = BraTSSliceDataset(empty, size=48, cache_path=cache, log=None)
        shutil.rmtree(empty, ignore_errors=True)

        assert len(first) == len(second)
        assert np.array_equal(first.images, second.images)
        assert np.array_equal(first.masks, second.masks)
        assert np.array_equal(first.case_ids, second.case_ids)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_split_by_case_is_disjoint():
    tmp = tempfile.mkdtemp()
    try:
        make_fake_brats(tmp, n_cases=6)
        ds = BraTSSliceDataset(tmp, size=48, slices_per_case=4, log=None)
        train_idx, val_idx = ds.split_by_case(val_frac=0.34, seed=0)

        assert not set(train_idx) & set(val_idx), "index overlap"
        assert len(train_idx) + len(val_idx) == len(ds)
        train_cases = {ds.case_ids[i] for i in train_idx}
        val_cases = {ds.case_ids[i] for i in val_idx}
        assert not train_cases & val_cases, "a case appears in both splits"
        assert len(val_cases) == 2, len(val_cases)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_brats_empty_root_raises():
    tmp = tempfile.mkdtemp()
    try:
        try:
            BraTSSliceDataset(tmp, size=48, log=None)
        except RuntimeError as e:
            assert "No case folders" in str(e)
        else:
            raise AssertionError("expected RuntimeError on an empty root")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001 - a tiny standalone runner
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
