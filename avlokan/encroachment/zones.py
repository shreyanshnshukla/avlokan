"""Restricted zones as raster masks on the analysis grid.

Two sources:
  1. Water bodies + buffer, derived automatically from JRC Global Surface
     Water: every lake/pond/river in the area, grown by `buffer_m`.
  2. Your own zones from a GeoJSON file (green belts, forest land, master
     plan zones), drawn on geojson.io and saved under zones/.

Buffer widths differ by rule and state; 30 m is a placeholder to be
replaced with the applicable local regulation before any real use.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def disk(radius_px: int) -> np.ndarray:
    r = int(max(radius_px, 0))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy**2 + xx**2) <= r * r


def water_bodies(occurrence: np.ndarray, min_occurrence: float = 25.0, min_area_px: int = 20) -> np.ndarray:
    """Water bodies from JRC occurrence: water in >= min_occurrence % of
    months since 1984, ignoring specks smaller than min_area_px."""
    occ = np.asarray(occurrence, dtype="float32")
    water = (occ >= min_occurrence) & (occ <= 100)
    lab, n = ndimage.label(water)
    if n == 0:
        return water
    sizes = ndimage.sum(water, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_area_px
    return keep[lab]


def buffer_zone(mask: np.ndarray, buffer_m: float, pixel_m: float = 10.0) -> np.ndarray:
    """mask grown by buffer_m (includes the mask itself: building *in* a lake
    is as much an encroachment as building beside it)."""
    r = int(round(buffer_m / pixel_m))
    if r <= 0:
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=disk(r))


def zones_from_geojson(path: str | Path, grid) -> dict[str, np.ndarray]:
    """Rasterise a GeoJSON FeatureCollection (WGS84) onto `grid`.
    Each feature's properties.name (or 'zone N') becomes a zone."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    gj = json.loads(Path(path).read_text())
    to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform
    zones: dict[str, np.ndarray] = {}
    for i, feat in enumerate(gj.get("features", [])):
        name = (feat.get("properties") or {}).get("name") or f"zone {i + 1}"
        geom = shp_transform(to_grid, shape(feat["geometry"]))
        m = rasterize([(geom, 1)], out_shape=grid.shape, transform=grid.affine(), fill=0, dtype="uint8").astype(bool)
        zones[name] = zones.get(name, np.zeros(grid.shape, bool)) | m
    return zones
