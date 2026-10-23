"""Turn "new construction inside a restricted zone" into reviewable flags.

Every flag is a candidate for a human to check, never a verdict. Each one
carries what a reviewer needs: where, how big, when it first appeared,
which zone, and how confident the model was.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

STATUS = "flagged_for_review"


def find_flags(
    first_built_year: np.ndarray,
    zones: dict[str, np.ndarray],
    grid,
    prob: np.ndarray | None = None,
    min_area_m2: float = 500.0,
    since_year: int | None = None,
) -> pd.DataFrame:
    """Connected patches of new construction inside each zone.

    first_built_year: from construction.change (year > 0 = new construction)
    prob: optional 0-255 model probability, reported as confidence
    min_area_m2: ignore patches smaller than this (5 pixels at 10 m)
    """
    new = first_built_year > 0
    if since_year is not None:
        new &= first_built_year >= since_year
    min_px = max(1, int(np.ceil(min_area_m2 / grid.pixel_area_m2)))
    rows = []
    for zone, zmask in zones.items():
        hit = new & zmask
        lab, n = ndimage.label(hit, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(hit, lab, idx)
        centres = ndimage.center_of_mass(hit, lab, idx)
        first = ndimage.minimum(first_built_year, lab, idx)
        confs = ndimage.mean(prob, lab, idx) if prob is not None else [None] * n
        for size, (r, c), fy, cf in zip(sizes, centres, first, confs):
            if size < min_px:
                continue
            conf = float(cf) / 255 if cf is not None else None
            rows.append({
                "zone": zone,
                "area_m2": round(float(size) * grid.pixel_area_m2, 0),
                "first_seen_year": int(fy),
                "row": float(r), "col": float(c),
                "confidence": round(conf, 3) if conf is not None else None,
                "status": STATUS,
            })
    df = pd.DataFrame(rows, columns=["zone", "area_m2", "first_seen_year", "row", "col", "confidence", "status"])
    if len(df):
        df = df.sort_values(["first_seen_year", "area_m2"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "flag_id", [f"AVL-{i + 1:04d}" for i in range(len(df))])
    return df


def add_coordinates(df: pd.DataFrame, grid) -> pd.DataFrame:
    if not len(df):
        return df.assign(lon=[], lat=[])
    lon, lat = grid.pixel_to_lonlat(df["row"].values, df["col"].values)
    return df.assign(lon=np.round(lon, 6), lat=np.round(lat, 6))


def to_geojson(df: pd.DataFrame, path: str | Path) -> Path:
    """Points at each flag's centre, with all attributes. Opens in
    geojson.io, QGIS or Google Earth for review."""
    feats = []
    for _, r in df.iterrows():
        props = {k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("lon", "lat", "row", "col")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": props})
    path = Path(path)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1, default=float))
    return path
