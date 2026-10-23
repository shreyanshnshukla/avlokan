"""Rebuild yearly_summary.csv, trends and the chart from an existing run's
water_timeseries.csv. No downloads: use after summary logic changes.

    python scripts/resummarize.py --lake sirpur --out outputs_lake
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lake", default="sirpur")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    from avlokan.analysis.timeseries import summary_trends, yearly_summary
    from avlokan.config import get_area
    from avlokan.viz.plots import plot_composition, plot_timeseries

    aoi = get_area(a.lake)
    d = Path(a.out) / aoi.key
    csv = d / "water_timeseries.csv"
    if not csv.exists():
        print(f"Missing {csv}")
        return 1
    ts = pd.read_csv(csv, parse_dates=["date"])
    for c in ("usable", "outlier"):
        if c in ts and ts[c].dtype != bool:
            ts[c] = ts[c].astype(str).str.lower().eq("true")
    summary = yearly_summary(ts)
    summary.to_csv(d / "yearly_summary.csv")
    plot_timeseries(ts, f"{aoi.name} — open water in the lake", d / "water_timeseries.png")
    plot_composition(ts, f"{aoi.name} — what fills the lake", d / "lake_composition.png")

    rp = d / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    report["trends_per_year"] = summary_trends(summary)
    report["summary_version"] = 2
    rp.write_text(json.dumps(report, indent=2))
    print(summary.to_string())
    print(json.dumps(report["trends_per_year"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
