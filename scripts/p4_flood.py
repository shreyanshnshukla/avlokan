"""Phase 4: flood history from radar + flood risk map.

    python scripts/p4_flood.py --area indore --years 2019-2026
    python scripts/p4_flood.py --area indore --event 2025-08-14     # one flood date

History mode: every Sentinel-1 scene in each monsoon (Jun-Sep), flooded
pixels counted, combined with terrain (Copernicus DEM) into a risk map.
Event mode: flood map for the scene nearest one date.

Writes outputs/<area>/flood/: flood.npz, flood_frequency.png, flood_risk.png,
summary.json (with prevention guidance per risk level), and for --event an
event_<date>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from avlokan.config import get_area  # noqa: E402


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def reference_grid(aoi, composites: Path):
    """Reuse a composite's grid so flood layers line up with Phases 2-3."""
    from avlokan.geo import load_layers

    files = sorted(composites.glob(f"{aoi.key}_*.npz"))
    if not files:
        return None, None
    grid, layers = load_layers(files[-1])
    return grid, layers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="indore")
    p.add_argument("--years", default="2019-2026")
    p.add_argument("--event", help="YYYY-MM-DD: map one flood event instead of history")
    p.add_argument("--method", choices=["fixed", "otsu"], default="fixed")
    p.add_argument("--threshold", type=float, default=-18.0, help="VV dB below this = water")
    p.add_argument("--composites", default="outputs/composites")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    from avlokan.data.stac import load_jrc_occurrence
    from avlokan.encroachment.zones import water_bodies
    from avlokan.flood.risk import (GUIDANCE, load_dem, relative_elevation, risk_classes, risk_score,
                                    slope_deg)
    from avlokan.flood.sar import FloodCounter, despeckle, flood_from_water, iter_vv_db, sar_water, search_s1
    from avlokan.geo import save_layers
    from avlokan.viz.maps import plot_flood, plot_risk, rgb_from_bands

    aoi = get_area(a.area)
    grid, layers = reference_grid(aoi, Path(a.composites))
    if grid is None:
        print(f"Build at least one composite first: python scripts/build_composites.py --area {a.area} --years 2021")
        return 1
    rgb = rgb_from_bands(layers)
    out = Path(a.out) / aoi.key / "flood"
    out.mkdir(parents=True, exist_ok=True)

    permanent = water_bodies(load_jrc_occurrence(aoi, grid), min_occurrence=50)
    print(f"permanent water: {permanent.sum() * grid.pixel_area_m2 / 1e4:,.0f} ha")

    if a.event:
        d = date.fromisoformat(a.event)
        items = search_s1(aoi, d - timedelta(days=6), d + timedelta(days=6))
        if not items:
            print("No Sentinel-1 scene within 6 days of that date.")
            return 1
        day, db = min(iter_vv_db(items, grid), key=lambda t: abs((t[0].date() - d).days))
        water, t = sar_water(despeckle(db), a.method, a.threshold)
        flood = flood_from_water(water, permanent)
        ha = float(flood.sum()) * grid.pixel_area_m2 / 1e4
        freq = flood.astype("float32")
        plot_flood(rgb, freq, permanent, f"{aoi.name}: flooding on {day.date()}", out / f"event_{day.date()}.png")
        save_layers(out / f"event_{day.date()}.npz", grid, flood=flood, vv_db=db)
        print(json.dumps({"scene_day": str(day.date()), "threshold_db": round(t, 2), "flooded_ha": round(ha, 1)},
                         indent=2))
        return 0

    fc = FloodCounter(grid.shape)
    m0, m1 = 6, 9
    for year in parse_years(a.years):
        items = search_s1(aoi, f"{year}-{m0:02d}-01", f"{year}-{m1:02d}-30")
        if not items:
            print(f"{year}: no Sentinel-1 scenes")
            continue
        n0 = fc.scenes
        for day, db in iter_vv_db(items, grid):
            obs = np.isfinite(db)
            if obs.mean() < 0.5:
                continue  # scene only clips the edge of the area
            water, _ = sar_water(despeckle(db), a.method, a.threshold)
            fc.add(flood_from_water(water, permanent), obs)
        print(f"{year}: {fc.scenes - n0} monsoon scenes used")
    if fc.scenes == 0:
        print("No usable scenes; can't build flood history.")
        return 1
    freq = fc.frequency()

    dem = load_dem(aoi, grid)
    rel = relative_elevation(dem, pixel_m=grid.res)
    score = risk_score(freq, rel, slope_deg(dem, grid.res))
    cls = risk_classes(score, exclude=permanent)
    save_layers(out / "flood.npz", grid, flood_frequency=freq, observed=fc.observed, risk_score=score,
                risk_class=cls, rel_elevation=rel)
    plot_flood(rgb, freq, permanent, f"{aoi.name}: how often flooded in monsoons {a.years}",
               out / "flood_frequency.png")
    plot_risk(rgb, cls, f"{aoi.name}: flood risk (screening)", out / "flood_risk.png")

    ha = grid.pixel_area_m2 / 1e4
    summary = {
        "area": aoi.key, "years": a.years, "monsoon_scenes": fc.scenes,
        "ever_flooded_ha": round(float((freq > 0).sum()) * ha, 1),
        "risk_ha": {name: round(float((cls == i).sum()) * ha, 1) for i, name in enumerate(["low", "medium", "high"])},
        "guidance": GUIDANCE,
        "note": "Screening map from radar history and terrain, not an engineering flood model.",
    }
    cpath = Path(a.out) / aoi.key / "construction" / "construction.npz"
    if cpath.exists():  # Phase 2 link: how much recent construction sits in high-risk zones
        from avlokan.geo import load_layers

        cg, c = load_layers(cpath)
        if cg.shape == grid.shape:
            new = c["first_built_year"] > 0
            summary["new_construction_in_high_risk_ha"] = round(float((new & (cls == 2)).sum()) * ha, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "guidance"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
