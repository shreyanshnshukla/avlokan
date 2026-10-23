"""Find and load Sentinel-2 imagery from Microsoft Planetary Computer.

No API key needed: the `planetary_computer` package signs asset URLs with
short-lived anonymous tokens.
"""

from __future__ import annotations

from datetime import date

import planetary_computer
import pystac_client
import xarray as xr
from odc.stac import load as odc_load

from ..config import AOI, DATA


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(DATA.stac_url, modifier=planetary_computer.sign_inplace)


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


JRC_COLLECTION = "jrc-gsw"


def load_jrc_occurrence(aoi: AOI, like: xr.Dataset) -> "np.ndarray":
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
        items[:1], bands=["occurrence"], geobox=like.odc.geobox, resampling="nearest",
        patch_url=planetary_computer.sign,
    )
    occ = jrc["occurrence"].isel(time=0).values if "time" in jrc.dims else jrc["occurrence"].values
    return np.asarray(occ)
