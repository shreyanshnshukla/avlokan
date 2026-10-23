"""Grid metadata shared by every phase.

Rasters are plain numpy arrays on a common grid (UTM 43N, 10 m). A `Grid`
records where that array sits on Earth, so results can be saved, reloaded,
turned into coordinates, and lined up with other layers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: tuple[float, float, float, float, float, float]  # affine a, b, c, d, e, f
    shape: tuple[int, int]  # (height, width)

    @property
    def res(self) -> float:
        return abs(self.transform[0])

    @property
    def pixel_area_m2(self) -> float:
        return abs(self.transform[0] * self.transform[4])

    def pixel_to_xy(self, row, col):
        """Pixel centre -> projected x, y (metres)."""
        a, b, c, d, e, f = self.transform
        row = np.asarray(row, dtype="float64") + 0.5
        col = np.asarray(col, dtype="float64") + 0.5
        return a * col + b * row + c, d * col + e * row + f

    def xy_to_lonlat(self, x, y):
        from pyproj import Transformer

        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return t.transform(x, y)

    def pixel_to_lonlat(self, row, col):
        return self.xy_to_lonlat(*self.pixel_to_xy(row, col))

    def affine(self):
        from affine import Affine

        return Affine(*self.transform)

    def to_geobox(self):
        """odc-geo GeoBox, so other datasets can be loaded onto this grid."""
        from odc.geo.geobox import GeoBox

        return GeoBox(self.shape, self.affine(), self.crs)

    def to_json(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform), "shape": list(self.shape)}

    @classmethod
    def from_json(cls, d: dict) -> "Grid":
        return cls(crs=d["crs"], transform=tuple(d["transform"]), shape=tuple(d["shape"]))

    @classmethod
    def from_geobox(cls, gbox) -> "Grid":
        """From an odc-geo GeoBox (what odc.stac.load produces)."""
        a = gbox.affine
        return cls(crs=str(gbox.crs), transform=(a.a, a.b, a.c, a.d, a.e, a.f), shape=tuple(gbox.shape))


def save_layers(path: str | Path, grid: Grid, **layers: np.ndarray) -> Path:
    """Save named arrays + their grid in one compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __grid__=json.dumps(grid.to_json()), **layers)
    return path


def load_layers(path: str | Path) -> tuple[Grid, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        grid = Grid.from_json(json.loads(str(z["__grid__"])))
        return grid, {k: z[k] for k in z.files if k != "__grid__"}
