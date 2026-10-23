"""From yearly built-up maps to "new construction, and when".

Single-year maps flicker: a dusty field can look built-up one year and not
the next. So a pixel only counts as newly built in year Y if it is built in
Y *and stays built* in the following `persist` years (when they exist), and
was not built in the baseline years. Construction is effectively permanent,
so requiring persistence removes most false alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def first_built_year(
    built: dict[int, np.ndarray],
    valid: dict[int, np.ndarray] | None = None,
    persist: int = 1,
    baseline_years: int = 1,
) -> np.ndarray:
    """Year each pixel first became (persistently) built-up.

    built: {year: bool (Y, X)}; valid: {year: bool} where the year's map is
    trustworthy (cloud-free composite). Returns int16 (Y, X):
      0      = never built in the period
      -1     = already built in the baseline years (old construction)
      year   = first year of persistent new construction
    """
    years = sorted(built)
    if len(years) <= baseline_years:
        raise ValueError("need more years than baseline_years")
    shape = built[years[0]].shape
    ok = {y: (valid[y] if valid else np.ones(shape, bool)) for y in years}
    base = np.zeros(shape, bool)
    for y in years[:baseline_years]:
        base |= built[y] & ok[y]
    out = np.zeros(shape, dtype="int16")
    out[base] = -1
    undecided = ~base
    for i, y in enumerate(years[baseline_years:], start=baseline_years):
        nxt = years[i + 1:i + 1 + persist]
        stays = built[y] & ok[y]
        for z in nxt:  # must still be built later (if that year was observed)
            stays &= built[z] | ~ok[z]
        hit = undecided & stays
        out[hit] = y
        undecided &= ~hit
    return out


def construction_table(first_year: np.ndarray, built: dict[int, np.ndarray], pixel_area_m2: float) -> pd.DataFrame:
    """Per year: total built-up area and newly built area, in hectares."""
    ha = pixel_area_m2 / 10_000.0
    rows = []
    for y in sorted(built):
        rows.append({"year": y,
                     "built_up_ha": round(float(built[y].sum()) * ha, 2),
                     "new_construction_ha": round(float((first_year == y).sum()) * ha, 2)})
    return pd.DataFrame(rows)
