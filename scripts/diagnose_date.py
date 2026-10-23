"""Diagnose water detection on one date.

Example (the 2023 Sirpur image where the dark basin was missed):
    python scripts/diagnose_date.py --lake sirpur --date 2023-10-06

Writes outputs/<lake>/diagnose_<date>.png and .json and prints the likely cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avlokan.config import LAKES  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Avlokan: diagnose water detection on one date")
    p.add_argument("--lake", default="sirpur", choices=list(LAKES))
    p.add_argument("--date", required=True, help="YYYY-MM-DD (nearest scene within +/- 2 days is used)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-jrc", action="store_true", help="skip the JRC lake boundary")
    a = p.parse_args()

    from avlokan.data.stac import dedupe_by_day, load_jrc_occurrence, load_stack, search_scenes
    from avlokan.diagnose import diagnose_scene
    from avlokan.processing.lake import boundary_from_jrc

    aoi = LAKES[a.lake]
    d = date.fromisoformat(a.date)
    items = dedupe_by_day(search_scenes(aoi, d - timedelta(days=2), d + timedelta(days=2), max_cloud=100))
    if not items:
        print(f"No scene within 2 days of {d}.")
        return 1
    item = min(items, key=lambda it: abs((it.datetime.date() - d).days))
    print(f"Using scene {item.id} ({item.datetime.date()}, scene cloud {item.properties.get('eo:cloud_cover')}%)")
    stack = load_stack([item], aoi)
    scene = stack.isel(time=0).compute()

    boundary = None
    if not a.no_jrc:
        try:
            boundary = boundary_from_jrc(load_jrc_occurrence(aoi, stack))
        except Exception as e:
            print(f"JRC boundary unavailable ({type(e).__name__}: {e}); using the whole box.")

    out = Path(a.out) / aoi.key
    out.mkdir(parents=True, exist_ok=True)
    tag = item.datetime.date().isoformat()
    stats = diagnose_scene(scene, boundary, out / f"diagnose_{tag}.png")
    stats["scene_id"] = item.id
    (out / f"diagnose_{tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nImage: {out / f'diagnose_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
