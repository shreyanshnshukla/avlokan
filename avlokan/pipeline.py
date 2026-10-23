"""Phase 1 end-to-end: AOI -> scenes -> water time series -> report files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr

from .analysis.timeseries import scene_water, trend_ha_per_year, water_timeseries, yearly_summary
from .config import AOI
from .viz.plots import plot_before_after, plot_timeseries


def pick_before_after(ts: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Earliest and latest usable post-monsoon dates (same season = fair
    comparison). Falls back to earliest/latest usable date of any season."""
    ok = ts[ts["usable"]]
    if len(ok) < 2:
        return None
    pm = ok[ok["season"] == "post_monsoon"]
    pool = pm if pm["year"].nunique() >= 2 else ok
    return pool["date"].min(), pool["date"].max()


def run_from_stack(
    aoi: AOI,
    stack: xr.Dataset,
    out_dir: str | Path,
    index: str = "mndwi",
    method: str = "fixed",
    threshold: float = 0.0,
) -> dict:
    """Everything after download. Separate so it can be tested offline."""
    out = Path(out_dir) / aoi.key
    out.mkdir(parents=True, exist_ok=True)

    ts = water_timeseries(stack, index=index, method=method, threshold=threshold)
    ts.to_csv(out / "water_timeseries.csv", index=False)

    summary = yearly_summary(ts)
    summary.to_csv(out / "yearly_summary.csv")

    plot_timeseries(ts, f"{aoi.name} — water surface area", out / "water_timeseries.png")

    pair = pick_before_after(ts)
    if pair:
        times = stack.time.to_index().normalize()
        scenes = [stack.isel(time=int(times.get_loc(d))) for d in pair]
        masks = [scene_water(s, index=index, method=method, threshold=threshold)["mask"] for s in scenes]
        plot_before_after(scenes[0], scenes[1], masks[0], masks[1],
                          f"{aoi.name}: then vs now", out / "before_after.png")

    report = {
        "aoi": aoi.key,
        "name": aoi.name,
        "bbox": aoi.bbox,
        "bbox_verified": aoi.verified,
        "index": index,
        "method": method,
        "scenes_total": int(len(ts)),
        "scenes_usable": int(ts["usable"].sum()),
        "first_date": str(ts["date"].min().date()) if len(ts) else None,
        "last_date": str(ts["date"].max().date()) if len(ts) else None,
        "trend_post_monsoon_ha_per_year": trend_ha_per_year(summary, "post_monsoon_peak_ha"),
        "trend_pre_monsoon_ha_per_year": trend_ha_per_year(summary, "pre_monsoon_low_ha"),
        "before_after_dates": [str(d.date()) for d in pair] if pair else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_lake(aoi: AOI, start: str, end: str, out_dir: str | Path = "outputs", **kwargs) -> dict:
    """Search Planetary Computer, load the stack, then run the analysis."""
    from .data.stac import dedupe_by_day, load_stack, search_scenes

    items = dedupe_by_day(search_scenes(aoi, start, end))
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {aoi.name} between {start} and {end}.")
    print(f"{aoi.name}: {len(items)} scenes found, loading...")
    stack = load_stack(items, aoi)
    return run_from_stack(aoi, stack, out_dir, **kwargs)
