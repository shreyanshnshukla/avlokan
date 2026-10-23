"""Fake Sentinel-2 stacks for tests and offline demos.

Builds a circular "lake" that fills each monsoon and shrinks a little every
year, with random clouds, so the full pipeline can run without internet.
Optionally a band of floating weeds covers the lake's western side: it's
lake, but spectrally it's vegetation, like hyacinth on Sirpur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def synthetic_stack(
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    every_days: int = 10,
    size: int = 120,
    base_radius_px: float = 35.0,
    shrink_per_year: float = 0.04,
    cloud_prob: float = 0.25,
    weeds: bool = False,
    seed: int = 7,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq=f"{every_days}D")
    yy, xx = np.mgrid[:size, :size]
    dist = np.hypot(yy - size / 2, xx - size / 2)

    data = {b: np.empty((len(times), size, size), dtype="uint16") for b in ("B02", "B03", "B04", "B08", "B11")}
    scl = np.empty((len(times), size, size), dtype="uint8")
    baselines = []

    for i, t in enumerate(times):
        years = (t - times[0]).days / 365.25
        seasonal = 0.85 + 0.15 * np.cos(2 * np.pi * (t.day_of_year - 290) / 365.25)  # peak ~mid-Oct
        r = base_radius_px * seasonal * (1 - shrink_per_year) ** years
        lake = dist < r
        weed = lake & (xx < size / 2 - 0.3 * r) if weeds else np.zeros_like(lake)
        water = lake & ~weed

        pb = 4.0 if t >= pd.Timestamp("2022-01-25") else 2.0
        baselines.append(pb)
        off = 1000 if pb >= 4.0 else 0
        noise = lambda: rng.normal(0, 60, (size, size))  # noqa: E731

        # Reflectance x 10,000. Water: green > NIR, SWIR. Land: the reverse.
        # Weeds: bright NIR (NDVI ~0.65), SWIR > green (MNDWI < 0).
        # Land is urban/bare like Indore's surroundings (NDVI ~0.15).
        means = {"B02": (600, 900, 500), "B03": (800, 1000, 900), "B04": (500, 1500, 600),
                 "B08": (300, 2000, 3000), "B11": (150, 2200, 1500)}
        for b, (w, land, wd) in means.items():
            base = np.where(water, w, np.where(weed, wd, land))
            # clip before casting so negative noise can't wrap around in uint16
            data[b][i] = np.clip(base + noise() + off, 1, 65535)

        s = np.where(water, 6, 4).astype("uint8")
        if rng.random() < cloud_prob:
            cy, cx = rng.integers(0, size, 2)
            rad = rng.integers(size // 6, size // 1.5)
            s[np.hypot(yy - cy, xx - cx) < rad] = 9
        scl[i] = s

    ds = xr.Dataset(
        {b: (("time", "y", "x"), arr) for b, arr in data.items()} | {"SCL": (("time", "y", "x"), scl)},
        coords={"time": times, "y": np.arange(size), "x": np.arange(size)},
    )
    return ds.assign_coords(processing_baseline=("time", baselines))
