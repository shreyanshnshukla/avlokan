"""Flood risk from terrain + observed flood history.

Three signals, each scaled 0-1 (1 = riskier):
  history    how often radar saw the place flooded in past monsoons
  low-lying  how far below its surroundings it sits (relative elevation)
  flatness   flat ground drains slowly

Score = weighted sum, then classes low / medium / high. This is a
screening map to decide where to look, not an engineering flood model.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEM_COLLECTION = "cop-dem-glo-30"

GUIDANCE = {
    "high": ("Flooded in past monsoons or sits well below its surroundings. Avoid basements and ground-floor "
             "storage; keep drains and natural channels clear; raise plinth levels for new buildings; plan "
             "evacuation routes before the monsoon."),
    "medium": ("Some flood history or low-lying ground. Check local drainage before building; keep plinth above "
               "street level; watch for waterlogging after heavy rain."),
    "low": "No flood history and relatively high ground. Normal precautions; keep street drains clear.",
}


def load_dem(aoi, grid) -> np.ndarray:
    import planetary_computer
    from odc.stac import load as odc_load

    from ..data.stac import open_catalog

    items = list(open_catalog().search(collections=[DEM_COLLECTION], bbox=aoi.bbox).items())
    if not items:
        raise RuntimeError("No Copernicus DEM tile for this area.")
    dem = odc_load(items, bands=["data"], geobox=grid.to_geobox(), resampling="bilinear",
                   patch_url=planetary_computer.sign)
    arr = dem["data"].values
    return np.asarray(arr[0] if arr.ndim == 3 else arr, dtype="float32")


def relative_elevation(dem: np.ndarray, window_m: float = 1000.0, pixel_m: float = 10.0) -> np.ndarray:
    """Height above the lowest point within window_m (a light stand-in for
    'height above nearest drainage'). Coarsened first, so a 1 km window
    over a whole city stays fast."""
    f = max(1, int(round(50 / pixel_m)))  # work at ~50 m
    small = dem[::f, ::f]
    w = max(3, int(round(window_m / (pixel_m * f))) | 1)
    lo = ndimage.minimum_filter(np.nan_to_num(small, nan=np.nanmax(small)), size=w)
    lo_full = np.repeat(np.repeat(lo, f, axis=0), f, axis=1)[:dem.shape[0], :dem.shape[1]]
    return (dem - lo_full).astype("float32")


def slope_deg(dem: np.ndarray, pixel_m: float = 10.0) -> np.ndarray:
    gy, gx = np.gradient(dem.astype("float32"), pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def risk_score(
    freq: np.ndarray | None,
    rel_elev: np.ndarray,
    slope: np.ndarray,
    weights=(0.5, 0.35, 0.15),
    rel_full_m: float = 15.0,
    slope_full_deg: float = 5.0,
) -> np.ndarray:
    """0-1 risk. History saturates at 30 % of monsoon scenes flooded;
    relative elevation at rel_full_m; slope at slope_full_deg."""
    w_h, w_l, w_f = weights
    low = 1 - np.clip(rel_elev / rel_full_m, 0, 1)
    flat = 1 - np.clip(slope / slope_full_deg, 0, 1)
    if freq is None:
        hist = np.zeros_like(low)
        w_h, w_l, w_f = 0.0, w_l / (w_l + w_f), w_f / (w_l + w_f)
    else:
        hist = np.clip(np.nan_to_num(freq, nan=0.0) / 0.3, 0, 1)
    return (w_h * hist + w_l * low + w_f * flat).astype("float32")


def risk_classes(score: np.ndarray, exclude: np.ndarray | None = None, cuts=(0.4, 0.6)) -> np.ndarray:
    """0 low, 1 medium, 2 high; -1 excluded (permanent water, no data)."""
    cls = np.digitize(score, cuts).astype("int8")
    cls[~np.isfinite(score)] = -1
    if exclude is not None:
        cls[exclude] = -1
    return cls
