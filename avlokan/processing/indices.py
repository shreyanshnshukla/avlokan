"""Spectral indices and reflectance handling for Sentinel-2 L2A."""

from __future__ import annotations

import numpy as np
import xarray as xr

# Since processing baseline 04.00 (25 Jan 2022), ESA adds a +1000 offset to
# L2A digital numbers. Without removing it, post-2022 images look brighter and
# any time series across that date gets a fake jump.
BOA_OFFSET = 1000
BOA_QUANTIFICATION = 10_000.0
OFFSET_BASELINE = 4.0


def to_reflectance(dn: xr.DataArray | np.ndarray, processing_baseline: float | None) -> xr.DataArray | np.ndarray:
    """Convert L2A digital numbers to surface reflectance (0-1)."""
    out = dn.astype("float32")
    if processing_baseline is not None and processing_baseline >= OFFSET_BASELINE:
        out = out - BOA_OFFSET
    out = out / BOA_QUANTIFICATION
    return out.clip(0, 1) if hasattr(out, "clip") else np.clip(out, 0, 1)


def normalized_difference(a, b):
    """(a - b) / (a + b), NaN where the denominator is 0."""
    num = a - b
    den = a + b
    if isinstance(den, xr.DataArray):
        return (num / den.where(den != 0)).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, np.nan).astype("float32")


def ndwi(green, nir):
    """McFeeters NDWI: water > 0. Good for clear open water."""
    return normalized_difference(green, nir)


def mndwi(green, swir):
    """Xu's Modified NDWI: suppresses built-up noise, better for urban lakes."""
    return normalized_difference(green, swir)
