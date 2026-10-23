"""Flood mapping from Sentinel-1 radar.

Floods happen in the monsoon, exactly when optical satellites only see
cloud. Radar sees through cloud: calm water reflects the radar pulse away
from the satellite, so it shows up very dark (low backscatter). We use
Sentinel-1 RTC (terrain-corrected) from Planetary Computer, VV polarisation.

Water = smoothed VV below a threshold (default -18 dB, or fitted per scene).
Flood = water now, but not permanent water (rivers and lakes from JRC).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..processing.water import otsu_threshold

S1_COLLECTION = "sentinel-1-rtc"
MONSOON = (6, 9)  # Jun-Sep


def to_db(linear: np.ndarray) -> np.ndarray:
    lin = np.asarray(linear, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(lin)
    db[~np.isfinite(db) | (lin <= 0)] = np.nan
    return db


def despeckle(db: np.ndarray, size: int = 5) -> np.ndarray:
    """Radar images are grainy ('speckle'); a median filter cleans it while
    keeping edges. NaNs are filled with a low-information value first."""
    fill = np.nanmedian(db) if np.isfinite(db).any() else 0.0
    out = ndimage.median_filter(np.where(np.isfinite(db), db, fill), size=size)
    out[~np.isfinite(db)] = np.nan
    return out.astype("float32")


def sar_water(vv_db: np.ndarray, method: str = "fixed", threshold: float = -18.0,
              bounds: tuple[float, float] = (-22.0, -14.0)) -> tuple[np.ndarray, float]:
    """Water from despeckled VV dB. 'otsu' fits the threshold per scene,
    clamped to physically sensible bounds."""
    v = vv_db[np.isfinite(vv_db)]
    t = threshold
    if method == "otsu" and v.size > 100:
        t_o = otsu_threshold(v)
        if np.isfinite(t_o):
            t = float(np.clip(t_o, *bounds))
    with np.errstate(invalid="ignore"):
        water = (vv_db < t) & np.isfinite(vv_db)
    return water, t


def flood_from_water(water: np.ndarray, permanent: np.ndarray, min_px: int = 10) -> np.ndarray:
    """Water that isn't permanent water, minus specks (radar shadows of
    single buildings, smooth tarmac)."""
    flood = water & ~permanent
    lab, n = ndimage.label(flood)
    if n == 0:
        return flood
    sizes = ndimage.sum(flood, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


class FloodCounter:
    """Running count of how often each pixel was flooded / observed."""

    def __init__(self, shape):
        self.flooded = np.zeros(shape, dtype="int32")
        self.observed = np.zeros(shape, dtype="int32")
        self.scenes = 0

    def add(self, flood: np.ndarray, observed: np.ndarray):
        self.flooded += flood & observed
        self.observed += observed
        self.scenes += 1

    def frequency(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.observed > 0, self.flooded / self.observed, np.nan).astype("float32")


def flood_frequency(floods, observed) -> tuple[np.ndarray, np.ndarray]:
    """Share of observed monsoon scenes in which each pixel was flooded."""
    fc = FloodCounter(floods[0].shape)
    for f, o in zip(floods, observed):
        fc.add(f, o)
    return fc.frequency(), fc.observed


def search_s1(aoi, start, end):
    from ..data.stac import open_catalog

    items = list(open_catalog().search(collections=[S1_COLLECTION], bbox=aoi.bbox,
                                       datetime=f"{start}/{end}").items())
    items.sort(key=lambda it: it.datetime)
    return items


def iter_vv_db(items, grid):
    """Yield (day, VV dB on our grid) one acquisition day at a time, so a
    whole city over many monsoons never sits in memory at once."""
    import planetary_computer
    from odc.stac import load as odc_load

    ds = odc_load(items, bands=["vv"], geobox=grid.to_geobox(), groupby="solar_day",
                  resampling="bilinear", chunks={}, patch_url=planetary_computer.sign)
    for i, day in enumerate(ds.time.to_index()):
        yield day, to_db(ds["vv"].isel(time=i).values)
