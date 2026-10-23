"""Cloud-free yearly composites from Sentinel-2.

One clear image per year is what construction detection needs: take every
scene in the dry season (Jan-Apr in Indore, fewest clouds and no monsoon
greenery), mask clouds/shadows per pixel, and keep the median. Median
ignores leftover haze and odd one-off values.
"""

from __future__ import annotations

import numpy as np

from ..config import SCL_CLEAR_CLASSES
from ..processing.indices import BOA_OFFSET, BOA_QUANTIFICATION, OFFSET_BASELINE

COMPOSITE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
DRY_SEASON = (1, 4)  # months, inclusive


def median_composite(dn: np.ndarray, scl: np.ndarray, baselines, min_obs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median reflectance over clear observations.

    dn:  (T, Y, X) digital numbers for one band
    scl: (T, Y, X) scene classification
    baselines: processing baseline per time step (removes the 2022 offset)
    Returns (median reflectance (Y, X) float32 with NaN where < min_obs
    clear looks, count of clear looks (Y, X)).
    """
    dn = np.asarray(dn, dtype="float32")
    off = np.where(np.asarray(baselines, dtype="float32") >= OFFSET_BASELINE, BOA_OFFSET, 0.0)
    refl = (dn - off[:, None, None]) / BOA_QUANTIFICATION
    clear = np.isin(scl, SCL_CLEAR_CLASSES) & (dn > 0)
    refl = np.where(clear, np.clip(refl, 0, 1), np.nan)
    n = clear.sum(axis=0)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(refl, axis=0).astype("float32")
    med[n < min_obs] = np.nan
    return med, n.astype("int16")


def composite_from_stack(stack, bands=COMPOSITE_BANDS, block_rows: int = 256,
                         reload_stack=None) -> tuple[dict, np.ndarray]:
    """Median composite for every band of a lazily loaded stack, built in
    horizontal strips so a whole city fits in laptop memory.

    reload_stack: optional callable returning a freshly loaded stack. A whole
    city takes longer to read than an asset token stays valid, so it is called
    per strip to re-sign the URLs the strip is about to read.
    Returns ({band: (Y, X) float32}, clear-look count (Y, X))."""
    ny = stack.sizes["y"]
    pb = np.asarray(stack["processing_baseline"].values, dtype="float32")
    out = {b: np.full((ny, stack.sizes["x"]), np.nan, dtype="float32") for b in bands}
    count = np.zeros((ny, stack.sizes["x"]), dtype="int16")
    for r0 in range(0, ny, block_rows):
        sl = slice(r0, min(ny, r0 + block_rows))
        strip = (reload_stack() if reload_stack else stack).isel(y=sl)
        scl = strip["SCL"].values
        for b in bands:
            med, n = median_composite(strip[b].values, scl, pb)
            out[b][sl] = med
        count[sl] = n
    return out, count


def yearly_composite(aoi, year: int, season=DRY_SEASON, max_cloud: float = 40.0):
    """Search, load and composite one year. Returns (bands dict, count, Grid, n_scenes)."""
    from ..data.stac import dedupe_by_day, fresh_stack, load_stack, search_scenes
    from ..geo import Grid

    m0, m1 = season
    end_day = 30 if m1 in (4, 6, 9, 11) else (28 if m1 == 2 else 31)
    items = dedupe_by_day(search_scenes(aoi, f"{year}-{m0:02d}-01", f"{year}-{m1:02d}-{end_day}", max_cloud))
    if not items:
        raise RuntimeError(f"No scenes for {aoi.name} in {year} months {m0}-{m1}")
    all_bands = COMPOSITE_BANDS + ("SCL",)
    stack = load_stack(items, aoi, bands=all_bands)
    bands, count = composite_from_stack(
        stack, reload_stack=lambda: fresh_stack(items, aoi, bands=all_bands))
    return bands, count, Grid.from_geobox(stack.odc.geobox), len(items)


def stack_bands(bands: dict, order=COMPOSITE_BANDS) -> np.ndarray:
    """dict of (Y, X) -> (C, Y, X) float32 in a fixed band order."""
    return np.stack([bands[b] for b in order]).astype("float32")
