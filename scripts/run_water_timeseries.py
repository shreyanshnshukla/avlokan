"""Phase 1 CLI: water-area history for one or all Indore lakes.

Examples
    python scripts/run_water_timeseries.py --lake sirpur
    python scripts/run_water_timeseries.py --lake all --start 2019-01-01
    python scripts/run_water_timeseries.py --lake bilawali --method otsu
    python scripts/run_water_timeseries.py --lake pipliyapala --show-bbox
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avlokan.config import LAKES  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Avlokan Phase 1: lake water-area time series")
    p.add_argument("--lake", default="sirpur", help=f"one of {', '.join(LAKES)} or 'all'")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--index", choices=["mndwi", "ndwi"], default="mndwi")
    p.add_argument("--method", choices=["fixed", "otsu", "lake"], default="fixed",
                   help="lake = strict 0.0 outside the lake; inside it, index > -0.15 or classifier says water, never weeds (needs JRC)")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-jrc", action="store_true",
                   help="don't use the JRC lake boundary; build it from this run's own history")
    p.add_argument("--show-bbox", action="store_true", help="print the AOI box as GeoJSON and exit")
    a = p.parse_args()

    keys = list(LAKES) if a.lake == "all" else [a.lake]
    unknown = [k for k in keys if k not in LAKES]
    if unknown:
        p.error(f"unknown lake(s): {unknown}")

    if a.show_bbox:
        for k in keys:
            x0, y0, x1, y1 = LAKES[k].bbox
            geo = {"type": "Feature", "properties": {"name": LAKES[k].name},
                   "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}
            print(json.dumps(geo))
        return 0

    from avlokan.pipeline import run_lake

    for k in keys:
        report = run_lake(LAKES[k], a.start, a.end, a.out, use_jrc=not a.no_jrc,
                          index=a.index, method=a.method, threshold=a.threshold)
        print(json.dumps(report, indent=2))
        if not LAKES[k].verified:
            print(f"Note: the {LAKES[k].name} bbox hasn't been verified on a map yet (see avlokan/config.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
