"""Project-wide configuration: areas of interest and data settings.

Coordinates are WGS84 (EPSG:4326). Each area of interest (AOI) is a centre
point plus a padding in degrees, which becomes a bounding box. Padding is
deliberately generous so the whole water body (and a margin of land around
it) is inside the box. Tighten it once you've checked the box on a map.

To verify or adjust an AOI, run:
    python scripts/run_water_timeseries.py --lake <key> --show-bbox
and paste the printed bbox into https://geojson.io or Google Maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AOI:
    key: str
    name: str
    lat: float
    lon: float
    pad_deg: float
    notes: str = ""
    verified: bool = False  # flip to True once the bbox has been checked on a map
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        return (
            round(self.lon - self.pad_deg, 5),
            round(self.lat - self.pad_deg, 5),
            round(self.lon + self.pad_deg, 5),
            round(self.lat + self.pad_deg, 5),
        )


# Indore water bodies for Phase 1.
# Sources: Wikipedia / Ramsar site pages. Sirpur's Wikipedia longitude has a
# typo (78.8 E); the lake is in west Indore at ~75.81 E. Pipliyapala's centre
# is approximate. Check every box on a map before trusting area numbers.
LAKES: dict[str, AOI] = {
    "sirpur": AOI(
        key="sirpur",
        name="Sirpur Lake",
        lat=22.7006,
        lon=75.8128,
        pad_deg=0.015,
        notes="Ramsar wetland in west Indore.",
        tags=("lake", "ramsar"),
    ),
    "bilawali": AOI(
        key="bilawali",
        name="Bilawali Lake",
        lat=22.633,
        lon=75.833,
        pad_deg=0.015,
        notes="South Indore; ~2-2.5 sq km surface area.",
        tags=("lake",),
    ),
    "yeshwant_sagar": AOI(
        key="yeshwant_sagar",
        name="Yeshwant Sagar",
        lat=22.8093,
        lon=75.6944,
        pad_deg=0.04,
        notes="Large reservoir NW of Indore; Ramsar site (~2,650 ha).",
        tags=("reservoir", "ramsar"),
    ),
    "pipliyapala": AOI(
        key="pipliyapala",
        name="Pipliyapala Lake (Regional Park)",
        lat=22.6935,
        lon=75.8835,
        pad_deg=0.008,
        notes="APPROXIMATE centre. Small urban lake (~80 acres).",
        tags=("lake", "urban"),
    ),
}


@dataclass(frozen=True)
class DataSettings:
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "sentinel-2-l2a"
    # Bands we need: visible for RGB previews, NIR + SWIR for water indices,
    # SCL (scene classification) for cloud masking.
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "SCL")
    resolution_m: int = 10
    crs: str = "EPSG:32643"  # UTM zone 43N, covers Indore; metres
    max_scene_cloud_pct: float = 30.0  # scene-level filter before download
    min_valid_fraction: float = 0.8  # AOI-level: drop dates with >20% masked


DATA = DataSettings()

# Sentinel-2 Scene Classification (SCL) classes treated as clear ground/water.
# 2 dark area, 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
# Excluded: 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow.
SCL_CLEAR_CLASSES: tuple[int, ...] = (2, 4, 5, 6, 7)
