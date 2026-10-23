"""Turn an image stack into a lake time series and yearly summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..config import DATA
from ..processing.indices import mndwi, ndwi, to_reflectance
from ..processing.lake import VEG_NDVI, boundary_from_history, composition, ndvi
from ..processing.water import area_hectares, clear_mask, lake_water_mask, water_mask

# Indore's seasons: lakes fill during the monsoon (Jun-Sep), peak just after
# it, and are lowest before the next one. Comparing the same season across
# years avoids mistaking normal seasonal drying for long-term shrinkage.
SEASONS = {
    "post_monsoon": (10, 11),        # Oct-Nov: fullest
    "winter": (12, 1, 2),
    "pre_monsoon": (3, 4, 5),        # Mar-May: lowest
    "monsoon": (6, 7, 8, 9),         # mostly cloudy, few usable scenes
}

COLUMNS = ["date", "water_ha", "open_water_ha", "vegetated_ha", "bare_ha", "lake_extent_ha",
           "valid_fraction", "lake_clear_fraction", "threshold", "usable", "outlier", "season", "year"]


def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(month)


def scene_water(scene: xr.Dataset, index: str = "mndwi", method: str = "fixed", threshold: float = 0.0,
                boundary: np.ndarray | None = None) -> dict:
    """Water mask, vegetation mask and stats for a single time step.

    method: "fixed" (index > threshold), "otsu" (whole-scene adaptive), or
    "lake" (strict outside the lake boundary, looser inside; needs boundary).
    """
    pb = float(scene["processing_baseline"].item()) if "processing_baseline" in scene.coords else None
    green = to_reflectance(scene["B03"].values, pb)
    if index == "mndwi":
        idx = mndwi(green, to_reflectance(scene["B11"].values, pb))
    elif index == "ndwi":
        idx = ndwi(green, to_reflectance(scene["B08"].values, pb))
    else:
        raise ValueError(index)
    scl = scene["SCL"].values
    clear = clear_mask(scl)
    nd = ndvi(to_reflectance(scene["B04"].values, pb), to_reflectance(scene["B08"].values, pb))
    if method == "lake":
        if boundary is None:
            raise ValueError("method 'lake' needs a lake boundary (JRC)")
        mask, t = lake_water_mask(idx, scl, clear, boundary, base_threshold=threshold, ndvi=nd, veg_ndvi=VEG_NDVI)
    else:
        mask, t = water_mask(idx, clear, method=method, threshold=threshold)
    veg = (nd > VEG_NDVI) & clear
    return {
        "mask": mask,
        "veg": veg,
        "index": idx,
        "clear": clear,
        "valid_fraction": float(clear.mean()),
        "threshold": t,
        "water_ha": area_hectares(mask, DATA.resolution_m),
    }


def flag_outliers(
    ts: pd.DataFrame,
    column: str = "water_ha",
    window_days: int = 30,
    k: float = 4.0,
    rel: float = 0.35,
    min_abs_ha: float = 5.0,
    min_neighbors: int = 3,
) -> pd.DataFrame:
    """Mark usable dates that disagree wildly with their neighbours.

    Lakes don't double or vanish in a week; a single date that does is thin
    haze or a missed cloud shadow. For each usable date we take the usable
    dates within +/- window_days (excluding itself). If it differs from their
    median by more than all of: k robust standard deviations, `rel` of the
    median, and `min_abs_ha`, it's flagged `outlier` and made unusable.
    Dates with too few neighbours are left alone (no basis to judge).
    """
    ts = ts.copy()
    ts["outlier"] = False
    ok = ts.index[ts["usable"]]
    dates = ts.loc[ok, "date"].values
    vals = ts.loc[ok, column].values.astype(float)
    win = np.timedelta64(window_days, "D")
    for j, i in enumerate(ok):
        near = (np.abs(dates - dates[j]) <= win)
        near[j] = False
        if near.sum() < min_neighbors:
            continue
        nb = vals[near]
        med = float(np.median(nb))
        mad = 1.4826 * float(np.median(np.abs(nb - med)))
        dev = abs(vals[j] - med)
        if dev > k * mad and dev > rel * med and dev > min_abs_ha:
            ts.at[i, "outlier"] = True
    ts.loc[ts["outlier"], "usable"] = False
    return ts


def water_timeseries(
    stack: xr.Dataset,
    index: str = "mndwi",
    method: str = "fixed",
    threshold: float = 0.0,
    min_valid_fraction: float = DATA.min_valid_fraction,
    min_lake_clear_fraction: float = 0.98,
    potential_water_freq: float = 0.1,
    lake_boundary: np.ndarray | None = None,
    remove_outliers: bool = True,
    return_boundary: bool = False,
):
    """One row per date: water area, lake composition, cloud cover, and
    whether to trust it.

    `usable` requires:
      1. `valid_fraction`: enough of the whole box is cloud-free, and
      2. `lake_clear_fraction`: the lake itself is cloud-free (a cloud over
         the lake hides water and fakes shrinkage), and
      3. not an `outlier` (see flag_outliers).
    Lake area for check 2 is `lake_boundary` if given, else pixels seen as
    water in >= potential_water_freq of their clear observations.

    Composition columns (open water / vegetated / bare / lake extent) are
    measured inside `lake_boundary`, or a boundary built from this stack's
    own history when none is given. Attrs record which was used.
    Unusable dates stay in the table (visible) but never enter summaries.

    With return_boundary=True returns (table, info) where info has the
    boundary mask (2-D), its source ("provided"/"history") and area in ha.
    """
    if method == "lake" and lake_boundary is None:
        raise ValueError("method 'lake' needs lake_boundary (from JRC)")
    b2d = np.asarray(lake_boundary, dtype=bool) if lake_boundary is not None else None
    meta, water_bits, clear_bits, veg_bits = [], [], [], []
    shape = None
    for i, t in enumerate(stack.time.to_index()):
        res = scene_water(stack.isel(time=i), index=index, method=method, threshold=threshold, boundary=b2d)
        shape = res["mask"].shape
        water_bits.append(np.packbits(res["mask"]))  # 8x smaller in memory
        clear_bits.append(np.packbits(res["clear"]))
        veg_bits.append(np.packbits(res["veg"]))
        meta.append((t, res["water_ha"], res["valid_fraction"], res["threshold"]))

    if not meta:
        empty = pd.DataFrame(columns=COLUMNS)
        return (empty, {"mask": None, "source": None, "boundary_ha": 0.0}) if return_boundary else empty

    n_px = int(np.prod(shape))
    unpack = lambda b: np.unpackbits(b, count=n_px).astype(bool)  # noqa: E731
    water_count = np.zeros(n_px, dtype="int32")
    clear_count = np.zeros(n_px, dtype="int32")
    for wb, cb in zip(water_bits, clear_bits):
        water_count += unpack(wb)
        clear_count += unpack(cb)

    if lake_boundary is not None:
        boundary = np.asarray(lake_boundary, dtype=bool).reshape(-1)
        if boundary.size != n_px:
            raise ValueError(f"lake_boundary has {boundary.size} pixels, stack has {n_px}")
        source = "provided"
    else:
        boundary = boundary_from_history(water_count, clear_count, min_freq=0.05)
        source = "history"
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(clear_count > 0, water_count / clear_count, 0.0)
    potential = boundary if lake_boundary is not None else ((freq >= potential_water_freq) & (clear_count >= 3))

    px_ha = DATA.resolution_m**2 / 10_000.0
    rows = []
    for (t, water_ha, valid, thr), wb, cb, vb in zip(meta, water_bits, clear_bits, veg_bits):
        clear = unpack(cb)
        lake_clear = float(clear[potential].mean()) if potential.any() else 1.0
        comp = composition(unpack(wb), unpack(vb), clear, boundary, px_ha)
        rows.append(
            {
                "date": t.normalize(),
                "water_ha": water_ha,
                "open_water_ha": comp["open_water_ha"],
                "vegetated_ha": comp["vegetated_ha"],
                "bare_ha": comp["bare_ha"],
                "lake_extent_ha": comp["lake_extent_ha"],
                "valid_fraction": round(valid, 3),
                "lake_clear_fraction": round(lake_clear, 3),
                "threshold": round(thr, 3),
                "usable": valid >= min_valid_fraction and lake_clear >= min_lake_clear_fraction,
                "season": season_of(t.month),
                "year": t.year,
            }
        )
    ts = pd.DataFrame(rows)
    ts = flag_outliers(ts) if remove_outliers else ts.assign(outlier=False)
    ts = ts[COLUMNS]
    if return_boundary:
        return ts, {"mask": boundary.reshape(shape), "source": source,
                    "boundary_ha": float(boundary.sum()) * px_ha}
    return ts


def yearly_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """Per year, from usable dates only:

    - post-monsoon peak of open water and of lake extent (water + vegetation)
    - pre-monsoon low of open water (extent isn't used pre-monsoon: grass on
      the dry bed looks like weeds)
    - post-monsoon median share of the lake covered by vegetation (weeds)

    A lake whose *extent* keeps falling is shrinking. A lake whose open water
    falls while extent holds is getting weedier, which is a different problem.
    """
    ok = ts[ts["usable"]]
    pm = ok[ok["season"] == "post_monsoon"].groupby("year")
    pre = ok[ok["season"] == "pre_monsoon"].groupby("year")
    okpm = ok[ok["season"] == "post_monsoon"]
    with np.errstate(divide="ignore", invalid="ignore"):
        veg_share = (okpm["vegetated_ha"] / okpm["lake_extent_ha"]).where(okpm["lake_extent_ha"] > 0)
    parts = [
        pm["water_ha"].max().rename("post_monsoon_peak_ha"),
        pre["water_ha"].min().rename("pre_monsoon_low_ha"),
        pm["lake_extent_ha"].max().rename("post_monsoon_extent_ha"),
        veg_share.groupby(okpm["year"]).median().round(3).rename("post_monsoon_vegetated_share"),
        ok.groupby("year").size().rename("usable_scenes"),
    ]
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "year"
    return out


def trend_ha_per_year(summary: pd.DataFrame, column: str) -> float | None:
    """Least-squares slope of a summary column (per year). None if <3 years."""
    s = summary[column].dropna()
    if len(s) < 3:
        return None
    slope, _ = np.polyfit(s.index.astype(float), s.values.astype(float), 1)
    return float(slope)
