"""Turn an image stack into a water-area time series and yearly summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..config import DATA
from ..processing.indices import mndwi, ndwi, to_reflectance
from ..processing.water import area_hectares, clear_mask, water_mask

# Indore's seasons: lakes fill during the monsoon (Jun-Sep), peak just after
# it, and are lowest before the next one. Comparing the same season across
# years avoids mistaking normal seasonal drying for long-term shrinkage.
SEASONS = {
    "post_monsoon": (10, 11),        # Oct-Nov: fullest
    "winter": (12, 1, 2),
    "pre_monsoon": (3, 4, 5),        # Mar-May: lowest
    "monsoon": (6, 7, 8, 9),         # mostly cloudy, few usable scenes
}


def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(month)


def scene_water(scene: xr.Dataset, index: str = "mndwi", method: str = "fixed", threshold: float = 0.0) -> dict:
    """Water mask + stats for a single time step."""
    pb = float(scene["processing_baseline"].item()) if "processing_baseline" in scene.coords else None
    green = to_reflectance(scene["B03"].values, pb)
    if index == "mndwi":
        other = to_reflectance(scene["B11"].values, pb)
        idx = mndwi(green, other)
    elif index == "ndwi":
        other = to_reflectance(scene["B08"].values, pb)
        idx = ndwi(green, other)
    else:
        raise ValueError(index)
    clear = clear_mask(scene["SCL"].values)
    mask, t = water_mask(idx, clear, method=method, threshold=threshold)
    return {
        "mask": mask,
        "index": idx,
        "clear": clear,
        "valid_fraction": float(clear.mean()),
        "threshold": t,
        "water_ha": area_hectares(mask, DATA.resolution_m),
    }


def water_timeseries(
    stack: xr.Dataset,
    index: str = "mndwi",
    method: str = "fixed",
    threshold: float = 0.0,
    min_valid_fraction: float = DATA.min_valid_fraction,
    min_lake_clear_fraction: float = 0.98,
    potential_water_freq: float = 0.1,
) -> pd.DataFrame:
    """One row per date: water area, cloud cover, and whether to trust it.

    Two checks decide `usable`:
      1. `valid_fraction`: share of the whole box that is cloud-free.
      2. `lake_clear_fraction`: share of the *potential lake area* that is
         cloud-free. A small cloud over the lake hides water and makes the
         lake look smaller even when the rest of the box is clear, so this
         check matters more. Potential lake area = pixels seen as water in at
         least `potential_water_freq` of their clear observations.
    Unusable dates stay in the table (visible) but never enter summaries.
    """
    meta, water_bits, clear_bits = [], [], []
    shape = None
    for i, t in enumerate(stack.time.to_index()):
        res = scene_water(stack.isel(time=i), index=index, method=method, threshold=threshold)
        shape = res["mask"].shape
        water_bits.append(np.packbits(res["mask"]))  # 8x smaller in memory
        clear_bits.append(np.packbits(res["clear"]))
        meta.append((t, res))

    if not meta:
        return pd.DataFrame(columns=["date", "water_ha", "valid_fraction", "lake_clear_fraction",
                                     "threshold", "usable", "season", "year"])

    n_px = int(np.prod(shape))
    unpack = lambda b: np.unpackbits(b, count=n_px).astype(bool)  # noqa: E731
    water_count = np.zeros(n_px, dtype="int32")
    clear_count = np.zeros(n_px, dtype="int32")
    for wb, cb in zip(water_bits, clear_bits):
        water_count += unpack(wb)
        clear_count += unpack(cb)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(clear_count > 0, water_count / clear_count, 0.0)
    potential = (freq >= potential_water_freq) & (clear_count >= 3)

    rows = []
    for (t, res), cb in zip(meta, clear_bits):
        lake_clear = float(unpack(cb)[potential].mean()) if potential.any() else 1.0
        rows.append(
            {
                "date": t.normalize(),
                "water_ha": res["water_ha"],
                "valid_fraction": round(res["valid_fraction"], 3),
                "lake_clear_fraction": round(lake_clear, 3),
                "threshold": round(res["threshold"], 3),
                "usable": res["valid_fraction"] >= min_valid_fraction and lake_clear >= min_lake_clear_fraction,
                "season": season_of(t.month),
                "year": t.year,
            }
        )
    return pd.DataFrame(rows)


def yearly_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """Per year: post-monsoon peak and pre-monsoon low, from usable dates only.

    The pre-monsoon low is the most telling number: a lake that keeps less
    water every summer is shrinking, even if monsoon peaks look normal.
    """
    ok = ts[ts["usable"]]
    peak = ok[ok["season"] == "post_monsoon"].groupby("year")["water_ha"].max().rename("post_monsoon_peak_ha")
    low = ok[ok["season"] == "pre_monsoon"].groupby("year")["water_ha"].min().rename("pre_monsoon_low_ha")
    n = ok.groupby("year").size().rename("usable_scenes")
    out = pd.concat([peak, low, n], axis=1).sort_index()
    out.index.name = "year"
    return out


def trend_ha_per_year(summary: pd.DataFrame, column: str) -> float | None:
    """Least-squares slope of a summary column (ha/year). None if <3 years."""
    s = summary[column].dropna()
    if len(s) < 3:
        return None
    slope, _ = np.polyfit(s.index.astype(float), s.values.astype(float), 1)
    return float(slope)
