"""Build one cloud-free dry-season composite per year (shared by Phases 2-4).

    python scripts/build_composites.py --area indore --years 2019-2026
    python scripts/build_composites.py --area indore --years 2021 --show-bbox

Saves outputs/composites/<area>_<year>.npz (bands + clear-look count + grid).
Already-built years are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avlokan.config import get_area  # noqa: E402


def parse_years(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="indore")
    p.add_argument("--years", default="2019-2026", help="e.g. 2019-2026 or 2019,2021")
    p.add_argument("--out", default="outputs/composites")
    p.add_argument("--max-cloud", type=float, default=40.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--show-bbox", action="store_true")
    a = p.parse_args()
    aoi = get_area(a.area)
    if a.show_bbox:
        x0, y0, x1, y1 = aoi.bbox
        print(json.dumps({"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}))
        return 0

    from avlokan.data.composite import yearly_composite
    from avlokan.geo import save_layers

    for year in parse_years(a.years):
        path = Path(a.out) / f"{aoi.key}_{year}.npz"
        if path.exists() and not a.force:
            print(f"{year}: exists, skipping ({path})")
            continue
        print(f"{year}: searching and compositing...")
        bands, count, grid, n = yearly_composite(aoi, year, max_cloud=a.max_cloud)
        save_layers(path, grid, count=count, **bands)
        valid = float((count >= 2).mean())
        print(f"{year}: {n} scenes, {valid:.0%} of pixels have >= 2 clear looks -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
