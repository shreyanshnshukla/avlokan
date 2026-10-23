"""Lake-level measurements beyond open water.

From space, a lake choked with water hyacinth looks like vegetation, not
water. So "open water area" alone can't tell a shrinking lake apart from a
weedy one. Inside a fixed lake boundary we split every clear pixel into:

  open water   index says water
  vegetated    NDVI says plants (floating weeds, or grass on a dry bed)
  bare         neither: exposed dry bed, mud, or new construction

Lake extent = open water + vegetated.

Caveat: optical indices can't tell floating weeds from grass growing on a
dried-out lake bed. After the monsoon the bed is flooded, so vegetation
inside the boundary is floating/emergent weeds and "extent" is meaningful.
Before the monsoon it isn't, so summaries only use extent post-monsoon.
Bare ground growing inside the boundary year after year is what Phase 3
will check for encroachment.
"""

from __future__ import annotations

import numpy as np

from .indices import normalized_difference

VEG_NDVI = 0.4  # water hyacinth is typically 0.5-0.8; sparse grass and soil sit lower


def ndvi(red, nir):
    return normalized_difference(nir, red)


def boundary_from_history(water_count: np.ndarray, clear_count: np.ndarray,
                          min_freq: float = 0.05, min_obs: int = 3) -> np.ndarray:
    """Fallback boundary: pixels seen as open water in >= min_freq of their
    clear observations. Misses parts that were always weed-covered, so the
    JRC boundary is preferred when available."""
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(clear_count > 0, water_count / clear_count, 0.0)
    return (freq >= min_freq) & (clear_count >= min_obs)


def boundary_from_jrc(occurrence: np.ndarray, min_occurrence: float = 5.0) -> np.ndarray:
    """Boundary from JRC Global Surface Water occurrence (% of months with
    water, 1984-2021). Independent of today's weeds, so it's the fairer
    reference for 'how big should this lake be'. 255 = no data."""
    occ = np.asarray(occurrence, dtype="float32")
    return (occ >= min_occurrence) & (occ <= 100)


def composition(water: np.ndarray, veg: np.ndarray, clear: np.ndarray, boundary: np.ndarray,
                pixel_area_ha: float) -> dict:
    """Hectares of open water / vegetated / bare inside the boundary, plus how
    much of the boundary was cloud-free."""
    inside = boundary & clear
    w = water & inside
    v = veg & inside & ~water
    b = inside & ~w & ~v
    n_boundary = int(boundary.sum())
    return {
        "open_water_ha": float(w.sum()) * pixel_area_ha,
        "vegetated_ha": float(v.sum()) * pixel_area_ha,
        "bare_ha": float(b.sum()) * pixel_area_ha,
        "lake_extent_ha": float(w.sum() + v.sum()) * pixel_area_ha,
        "boundary_ha": n_boundary * pixel_area_ha,
        "boundary_clear_fraction": float(inside.sum() / n_boundary) if n_boundary else 1.0,
    }
