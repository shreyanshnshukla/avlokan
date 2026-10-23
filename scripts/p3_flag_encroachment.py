"""Phase 3: flag new construction inside restricted zones.

    python scripts/p3_flag_encroachment.py --area indore
    python scripts/p3_flag_encroachment.py --area indore --zones zones/indore_zones.geojson --buffer-m 30

Needs Phase 2 output (outputs/<area>/construction/construction.npz).
Zones: every water body from JRC plus a buffer, and optionally your own
GeoJSON zones (draw them on geojson.io, give each a "name" property).

Writes outputs/<area>/encroachment/: flags.csv, flags.geojson,
flags_map.png, summary.json. Every flag is "flagged for review".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from avlokan.config import get_area  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="indore")
    p.add_argument("--out", default="outputs")
    p.add_argument("--composites", default="outputs/composites")
    p.add_argument("--zones", help="optional GeoJSON of extra restricted zones")
    p.add_argument("--buffer-m", type=float, default=30.0, help="water-body buffer (placeholder; use local rules)")
    p.add_argument("--min-area-m2", type=float, default=500.0)
    p.add_argument("--since", type=int, help="only flag construction first seen in/after this year")
    p.add_argument("--no-water", action="store_true", help="skip automatic water-body zones")
    a = p.parse_args()

    from avlokan.encroachment.flags import add_coordinates, find_flags, to_geojson
    from avlokan.encroachment.zones import buffer_zone, water_bodies, zones_from_geojson
    from avlokan.geo import load_layers, save_layers
    from avlokan.viz.maps import plot_flags, rgb_from_bands

    aoi = get_area(a.area)
    cpath = Path(a.out) / aoi.key / "construction" / "construction.npz"
    if not cpath.exists():
        print(f"Missing {cpath}. Run scripts/p2_detect_change.py first.")
        return 1
    grid, c = load_layers(cpath)
    first = c["first_built_year"]
    prob = c.get("prob_last_year")

    zones: dict[str, np.ndarray] = {}
    if not a.no_water:
        from avlokan.data.stac import load_jrc_occurrence

        occ = load_jrc_occurrence(aoi, grid)
        water = water_bodies(occ)
        zones[f"water bodies + {a.buffer_m:g} m"] = buffer_zone(water, a.buffer_m, grid.res)
        print(f"water bodies: {water.sum() * grid.pixel_area_m2 / 1e4:,.0f} ha")
    if a.zones:
        extra = zones_from_geojson(a.zones, grid)
        zones.update(extra)
        print(f"loaded {len(extra)} zone(s) from {a.zones}")
    if not zones:
        print("No zones to check.")
        return 1

    flags = add_coordinates(find_flags(first, zones, grid, prob, a.min_area_m2, a.since), grid)
    out = Path(a.out) / aoi.key / "encroachment"
    out.mkdir(parents=True, exist_ok=True)
    flags.to_csv(out / "flags.csv", index=False)
    to_geojson(flags, out / "flags.geojson")
    flagged = np.zeros(grid.shape, bool)
    for z in zones.values():
        flagged |= z & (first > 0) & ((first >= a.since) if a.since else True)
    save_layers(out / "zones.npz", grid, **{f"zone_{i}": z for i, z in enumerate(zones.values())})

    last = sorted(Path(a.composites).glob(f"{aoi.key}_*.npz"))
    if last:
        _, layers = load_layers(last[-1])
        plot_flags(rgb_from_bands(layers), zones, flagged, f"{aoi.name}: construction in restricted zones",
                   out / "flags_map.png")
    summary = {
        "area": aoi.key,
        "zones": {k: round(float(v.sum()) * grid.pixel_area_m2 / 1e4, 1) for k, v in zones.items()},
        "flags": int(len(flags)),
        "flagged_area_ha": round(float(flags["area_m2"].sum()) / 1e4, 2) if len(flags) else 0.0,
        "by_zone": flags.groupby("zone").size().to_dict() if len(flags) else {},
        "by_year": {str(k): int(v) for k, v in flags.groupby("first_seen_year").size().items()} if len(flags) else {},
        "buffer_m": a.buffer_m,
        "note": "Candidates for human review, not findings of illegality. Buffer width is a placeholder.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(flags):
        print(flags.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
