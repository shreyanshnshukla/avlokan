"""Cut a composite + label into training patches.

Train/validation split is spatial, not random: patches are grouped into
large blocks and whole blocks go to validation. Random splits leak (two
overlapping neighbours land on both sides) and give inflated scores.
"""

from __future__ import annotations

import numpy as np

from ..data.composite import COMPOSITE_BANDS


def band_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over valid pixels of (C, Y, X)."""
    c = x.shape[0]
    flat = x.reshape(c, -1)
    ok = np.isfinite(flat).all(axis=0)
    mean = flat[:, ok].mean(axis=1)
    std = flat[:, ok].std(axis=1) + 1e-6
    return mean.astype("float32"), std.astype("float32")


def normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise each band; NaN (no clear look) becomes 0 = band mean."""
    z = (x - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(z, nan=0.0).astype("float32")


def make_patches(
    x: np.ndarray,
    label: np.ndarray,
    size: int = 128,
    stride: int = 128,
    min_valid: float = 0.9,
    val_block: int = 4,
    seed: int = 0,
) -> dict:
    """Patches of (C, size, size) with labels; spatial train/val split.

    Patch positions form a grid; grid cells are grouped in val_block x
    val_block blocks, and one block in every four goes to validation.
    Patches with < min_valid labelled, cloud-free pixels are skipped.
    """
    c, h, w = x.shape
    finite = np.isfinite(x).all(axis=0)
    xs, ys, split = [], [], []
    rng = np.random.default_rng(seed)
    n_bx = (w - size) // stride + 1
    n_by = (h - size) // stride + 1
    block_ids = {}
    for iy in range(n_by):
        for ix in range(n_bx):
            r, q = iy * stride, ix * stride
            lab = label[r:r + size, q:q + size]
            ok = finite[r:r + size, q:q + size] & (lab != 255)
            if ok.mean() < min_valid:
                continue
            blk = (iy // val_block, ix // val_block)
            if blk not in block_ids:
                block_ids[blk] = rng.random() < 0.25
            xs.append(x[:, r:r + size, q:q + size])
            ys.append(lab)
            split.append("val" if block_ids[blk] else "train")
    if not xs:
        raise ValueError("no usable patches (too cloudy, or area smaller than one patch)")
    X = np.stack(xs).astype("float32")
    Y = np.stack(ys).astype("uint8")
    split = np.array(split)
    if not (split == "val").any():  # tiny areas: force one val patch
        split[-1] = "val"
    return {"X": X, "Y": Y, "split": split, "bands": np.array(COMPOSITE_BANDS)}
