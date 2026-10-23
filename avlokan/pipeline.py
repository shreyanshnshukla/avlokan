"""Phase 1 end-to-end: AOI -> scenes -> lake time series -> report files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .analysis.timeseries import scene_water, trend_ha_per_year, water_timeseries, yearly_summary
from .config import AOI
from .viz.plots import plot_before_after, plot_composition, plot_timeseries


def pick_before_after(ts: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Earliest and latest usable post-monsoon dates (same season = fair
    comparison). Falls back to earliest/latest usable date of any season."""
    ok = ts[ts["usable"]]
    if len(ok) < 2:
        return None
    pm = ok[ok["season"] == "post_monsoon"]
    pool = pm if pm["year"].nunique() >= 2 else ok
    return pool["date"].min(), pool["date"].max()


def _round(x, nd=2):
    return None if x is None else round(float(x), nd)


def run_from_stack(
    aoi: AOI,
    stack: xr.Dataset,
    out_dir: str | Path,
    index: str = "mndwi",
    method: str = "fixed",
    threshold: float = 0.0,
    lake_boundary: np.ndarray | None = None,
    boundary_note: str | None = None,
) -> dict:
    """Everything after download. Separate so it can be tested offline."""
    out = Path(out_dir) / aoi.key
    out.mkdir(parents=True, exist_ok=True)

    ts, binfo = water_timeseries(stack, index=index, method=method, threshold=threshold,
                                 lake_boundary=lake_boundary, return_boundary=True)
    ts.to_csv(out / "water_timeseries.csv", index=False)

    summary = yearly_summary(ts)
    summary.to_csv(out / "yearly_summary.csv")

    plot_timeseries(ts, f"{aoi.name} — open water area", out / "water_timeseries.png")
    plot_composition(ts, f"{aoi.name} — what fills the lake", out / "lake_composition.png")

    pair = pick_before_after(ts)
    if pair:
        times = stack.time.to_index().normalize()
        scenes = [stack.isel(time=int(times.get_loc(d))) for d in pair]
        res = [scene_water(s, index=index, method=method, threshold=threshold, boundary=binfo["mask"])
               for s in scenes]
        plot_before_after(scenes[0], scenes[1], res[0]["mask"], res[1]["mask"],
                          f"{aoi.name}: then vs now", out / "before_after.png",
                          veg_before=res[0]["veg"], veg_after=res[1]["veg"], boundary=binfo["mask"])

    report = {
        "aoi": aoi.key,
        "name": aoi.name,
        "bbox": aoi.bbox,
        "bbox_verified": aoi.verified,
        "index": index,
        "method": method,
        "lake_boundary_source": binfo["source"] if not boundary_note else f"{binfo['source']} ({boundary_note})",
        "lake_boundary_ha": _round(binfo["boundary_ha"]),
        "scenes_total": int(len(ts)),
        "scenes_usable": int(ts["usable"].sum()),
        "scenes_outliers_removed": int(ts["outlier"].sum()),
        "first_date": str(ts["date"].min().date()) if len(ts) else None,
        "last_date": str(ts["date"].max().date()) if len(ts) else None,
        "trends_per_year": {
            "open_water_post_monsoon_ha": _round(trend_ha_per_year(summary, "post_monsoon_peak_ha")),
            "open_water_pre_monsoon_ha": _round(trend_ha_per_year(summary, "pre_monsoon_low_ha")),
            "lake_extent_post_monsoon_ha": _round(trend_ha_per_year(summary, "post_monsoon_extent_ha")),
            "vegetated_share_post_monsoon": _round(trend_ha_per_year(summary, "post_monsoon_vegetated_share"), 4),
        },
        "before_after_dates": [str(d.date()) for d in pair] if pair else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_lake(aoi: AOI, start: str, end: str, out_dir: str | Path = "outputs",
             use_jrc: bool = True, **kwargs) -> dict:
    """Search Planetary Computer, load the stack, then run the analysis.

    The lake boundary comes from JRC Global Surface Water when available (a
    40-year record, unaffected by today's weeds); otherwise it's built from
    this run's own history and the report says so.
    """
    from .data.stac import dedupe_by_day, load_jrc_occurrence, load_stack, search_scenes
    from .processing.lake import boundary_from_jrc

    items = dedupe_by_day(search_scenes(aoi, start, end))
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {aoi.name} between {start} and {end}.")
    print(f"{aoi.name}: {len(items)} scenes found, loading...")
    stack = load_stack(items, aoi)

    boundary, note = None, None
    if use_jrc:
        try:
            boundary = boundary_from_jrc(load_jrc_occurrence(aoi, stack))
            if not boundary.any():
                boundary, note = None, "JRC had no water here; fell back"
            else:
                note = "JRC Global Surface Water, occurrence >= 5%"
        except Exception as e:  # network, missing tile, API change: don't lose the run
            print(f"JRC boundary unavailable ({type(e).__name__}: {e}); using this run's history instead.")
            note = "JRC unavailable; fell back"
    if kwargs.get("method") == "lake" and boundary is None:
        print("method 'lake' needs the JRC boundary, which isn't available; using 'fixed' instead.")
        kwargs["method"] = "fixed"
    return run_from_stack(aoi, stack, out_dir, lake_boundary=boundary, boundary_note=note, **kwargs)
