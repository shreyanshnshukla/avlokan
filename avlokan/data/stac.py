"""Find and load Sentinel-2 imagery from Microsoft Planetary Computer.

No API key needed: the `planetary_computer` package signs asset URLs with
short-lived anonymous tokens. Signing happens at load time, not search
time -- see `open_catalog`.
"""

from __future__ import annotations

from datetime import date

import planetary_computer
import planetary_computer.sas
import pystac_client
import xarray as xr
from odc.stac import load as odc_load

from ..config import AOI, DATA


def open_catalog() -> pystac_client.Client:
    """Deliberately unsigned: `planetary_computer.sign` leaves a URL alone once
    it carries a token, so signing here would freeze every asset to one
    45-minute token and a longer run would die mid-read. Items stay unsigned
    and every odc.stac load signs them itself via `patch_url`."""
    return pystac_client.Client.open(DATA.stac_url)


def search_scenes(
    aoi: AOI,
    start: str | date,
    end: str | date,
    max_cloud: float = DATA.max_scene_cloud_pct,
):
    """Return STAC items intersecting the AOI, sorted oldest first."""
    catalog = open_catalog()
    search = catalog.search(
        collections=[DATA.collection],
        bbox=aoi.bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime)
    return items


def dedupe_by_day(items):
    """Sentinel-2 tiles overlap; keep one item per acquisition day (least cloudy)."""
    best = {}
    for it in items:
        day = it.datetime.date()
        cc = it.properties.get("eo:cloud_cover", 100)
        if day not in best or cc < best[day].properties.get("eo:cloud_cover", 100):
            best[day] = it
    return [best[d] for d in sorted(best)]


def load_stack(items, aoi: AOI, bands=DATA.bands) -> xr.Dataset:
    """Load the AOI cut-out for every item into one (time, y, x) dataset.

    Also attaches `processing_baseline` per time step, needed to remove the
    post-2022 reflectance offset.
    """
    ds = odc_load(
        items,
        bands=list(bands),
        bbox=aoi.bbox,
        crs=DATA.crs,
        resolution=DATA.resolution_m,
        groupby="solar_day",
        chunks={},
        patch_url=planetary_computer.sign,
    )
    baselines = {}
    for it in items:
        try:
            baselines[it.datetime.date()] = float(it.properties.get("s2:processing_baseline", "0"))
        except ValueError:
            baselines[it.datetime.date()] = 0.0
    pb = [baselines.get(t.date(), 0.0) for t in ds.time.to_index()]
    return ds.assign_coords(processing_baseline=("time", pb))


def refresh_asset_tokens() -> None:
    """Drop cached SAS tokens so the next `sign` mints a full-life one.

    Planetary Computer's tokens live ~45 minutes and its cache only replaces
    one with under a minute left, which is not enough to finish a read that
    takes several minutes. Call this before re-building a load, so a job that
    runs longer than one token keeps reading with a valid one.
    """
    planetary_computer.sas.TOKEN_CACHE.clear()


def fresh_stack(items, aoi: AOI, bands=DATA.bands) -> xr.Dataset:
    """`load_stack` with brand-new asset tokens."""
    refresh_asset_tokens()
    return load_stack(items, aoi, bands=bands)


JRC_COLLECTION = "jrc-gsw"


def _geobox(like):
    """Accept an odc-loaded dataset, a GeoBox, or an avlokan Grid."""
    if hasattr(like, "odc"):
        return like.odc.geobox
    if hasattr(like, "to_geobox"):
        return like.to_geobox()
    return like


def load_jrc_occurrence(aoi: AOI, like) -> "np.ndarray":
    """JRC Global Surface Water occurrence (% of months with water since
    1984), resampled onto the exact grid of `like` (our Sentinel-2 stack).

    It's 30 m data, so lake edges are a little blocky at our 10 m grid;
    fine for a boundary, not for edge-level detail.
    """
    import numpy as np

    catalog = open_catalog()
    items = list(catalog.search(collections=[JRC_COLLECTION], bbox=aoi.bbox).items())
    if not items:
        raise RuntimeError("No JRC Global Surface Water tile covers this AOI.")
    jrc = odc_load(
        items[:1], bands=["occurrence"], geobox=_geobox(like), resampling="nearest",
        patch_url=planetary_computer.sign,
    )
    occ = jrc["occurrence"].isel(time=0).values if "time" in jrc.dims else jrc["occurrence"].values
    return np.asarray(occ)
