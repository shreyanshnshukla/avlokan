"""Training labels and a no-ML baseline for built-up land.

Labels: ESA WorldCover (10 m, 2020 and 2021) class 50 = "built-up". It's
free and on Planetary Computer, but only exists for two years. That's the
reason to train a model at all: learn "what built-up looks like" from
2021, then apply it to every year 2017-2026.
"""

from __future__ import annotations

import numpy as np

from ..processing.indices import normalized_difference

WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_BUILT = 50
WORLDCOVER_WATER = 80


def load_worldcover(aoi, like, year: int = 2021) -> np.ndarray:
    """WorldCover class map on the exact grid of `like` (an odc-loaded
    stack, a GeoBox, or an avlokan Grid)."""
    import planetary_computer
    from odc.stac import load as odc_load

    from ..data.stac import _geobox, open_catalog

    items = list(open_catalog().search(collections=[WORLDCOVER_COLLECTION], bbox=aoi.bbox,
                                       datetime=f"{year}-01-01/{year}-12-31").items())
    if not items:
        raise RuntimeError(f"No WorldCover {year} tile for {aoi.name}")
    wc = odc_load(items, bands=["map"], geobox=_geobox(like), resampling="nearest",
                  groupby="solar_day", patch_url=planetary_computer.sign)
    arr = wc["map"].values
    return np.asarray(arr[0] if arr.ndim == 3 else arr)


def builtup_label(worldcover: np.ndarray) -> np.ndarray:
    """1 = built-up, 0 = everything else, 255 = no data."""
    wc = np.asarray(worldcover)
    lab = (wc == WORLDCOVER_BUILT).astype("uint8")
    lab[(wc == 0) | (wc == 255)] = 255
    return lab


def index_baseline(b: dict) -> np.ndarray:
    """Rule-based built-up map from a composite, for comparison with the
    model: built-up surfaces reflect more SWIR than NIR (NDBI > 0), aren't
    green and aren't water. Bare soil also has NDBI > 0, which is exactly
    the confusion the model should fix."""
    ndbi = normalized_difference(b["B11"], b["B08"])
    ndvi = normalized_difference(b["B08"], b["B04"])
    mndwi = normalized_difference(b["B03"], b["B11"])
    with np.errstate(invalid="ignore"):
        out = (ndbi > 0.0) & (ndvi < 0.25) & (mndwi < 0.0)
    return out & np.isfinite(ndbi)
