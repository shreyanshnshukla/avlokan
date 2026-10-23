"""Look inside one date: why did water detection do what it did?

Built to answer a specific question from the first real Sirpur run: on
06 Oct 2023 the dark eastern basin was clearly water in true colour but
wasn't detected. Possible causes, each visible in the output:

  A. The satellite's own scene classification (SCL) labelled the dark water
     as cloud shadow or similar, so it was masked out before we looked.
  B. The water index was just below the fixed 0.0 threshold (very dark or
     algae-rich water), which Otsu or a lower threshold would fix.
  C. Something else (haze, sensor artefact): neither index sees water.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .analysis.timeseries import scene_water  # noqa: E402
from .config import SCL_CLEAR_CLASSES  # noqa: E402
from .viz.plots import rgb  # noqa: E402

SCL_NAMES = {0: "no data", 1: "saturated", 2: "dark area", 3: "cloud shadow", 4: "vegetation",
             5: "bare soil", 6: "water", 7: "unclassified", 8: "cloud (medium)", 9: "cloud (high)",
             10: "thin cirrus", 11: "snow"}
SCL_COLOURS = ["#000000", "#ff0000", "#2f2f2f", "#643200", "#00a000", "#ffe65a", "#0000ff",
               "#808080", "#c0c0c0", "#ffffff", "#64c8ff", "#ff96ff"]


def diagnose_scene(scene: xr.Dataset, boundary: np.ndarray | None, out_png: str | Path) -> dict:
    """Six panels + numbers comparing SCL, MNDWI, NDWI, fixed vs Otsu."""
    fixed = scene_water(scene, index="mndwi", method="fixed", threshold=0.0)
    otsu = scene_water(scene, index="mndwi", method="otsu")
    ndwi_fixed = scene_water(scene, index="ndwi", method="fixed", threshold=0.0)
    has_b = boundary is not None and boundary.any()
    lake = scene_water(scene, index="mndwi", method="lake", boundary=boundary) if has_b else None
    scl = scene["SCL"].values.astype(int)

    # MNDWI without the cloud mask: what the index says even where SCL masked
    raw_mndwi = fixed["index"]
    region = boundary if boundary is not None and boundary.any() else np.ones_like(scl, bool)

    scl_counts = {SCL_NAMES.get(int(c), str(c)): int(n)
                  for c, n in zip(*np.unique(scl[region], return_counts=True))}
    masked_by_scl = region & ~np.isin(scl, SCL_CLEAR_CLASSES)
    wet_but_masked = masked_by_scl & (raw_mndwi > 0)
    px_ha = 0.01  # 10 m pixels
    v = raw_mndwi[region & np.isfinite(raw_mndwi)]
    stats = {
        "region": "lake boundary" if boundary is not None else "whole box",
        "region_ha": round(float(region.sum()) * px_ha, 2),
        "scl_classes_in_region_px": scl_counts,
        "masked_by_scl_ha": round(float(masked_by_scl.sum()) * px_ha, 2),
        "masked_by_scl_but_mndwi_says_water_ha": round(float(wet_but_masked.sum()) * px_ha, 2),
        "mndwi_percentiles_in_region": {p: round(float(np.percentile(v, p)), 3) for p in (5, 25, 50, 75, 95)} if v.size else {},
        "scl_says_water_in_region_ha": round(float((region & (scl == 6)).sum()) * px_ha, 2),
        "water_ha_whole_box": {
            "mndwi_fixed_0": round(fixed["water_ha"], 2),
            "mndwi_otsu": round(otsu["water_ha"], 2),
            "ndwi_fixed_0": round(ndwi_fixed["water_ha"], 2),
            **({"mndwi_lake": round(lake["water_ha"], 2)} if lake else {}),
        },
        "water_ha_inside_lake": {
            "mndwi_fixed_0": round(float((fixed["mask"] & region).sum()) * px_ha, 2),
            "mndwi_otsu": round(float((otsu["mask"] & region).sum()) * px_ha, 2),
            "ndwi_fixed_0": round(float((ndwi_fixed["mask"] & region).sum()) * px_ha, 2),
            **({"mndwi_lake": round(float((lake["mask"] & region).sum()) * px_ha, 2)} if lake else {}),
        },
        "otsu_threshold": round(otsu["threshold"], 3),
        **({"lake_threshold": round(lake["threshold"], 3)} if lake else {}),
    }

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    ax[0, 0].imshow(rgb(scene)); ax[0, 0].set_title("True colour")
    ax[0, 1].imshow(scl, cmap=ListedColormap(SCL_COLOURS), vmin=0, vmax=11, interpolation="nearest")
    ax[0, 1].set_title("Satellite's own classification (SCL)\nbrown = cloud shadow, grey = dark area, blue = water")
    im = ax[0, 2].imshow(raw_mndwi, cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax[0, 2].set_title("MNDWI (blue > 0 = water)"); fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    third = (lake, f"Lake method (inside lake > {lake['threshold']:.2f})") if lake else (ndwi_fixed, "NDWI fixed 0.0")
    for a, (res, name) in zip(ax[1], ((fixed, "MNDWI fixed 0.0"), (otsu, f"MNDWI Otsu {otsu['threshold']:.2f}"),
                                      third)):
        o = rgb(scene) * 0.5
        o[res["mask"]] = [0.12, 0.45, 0.85]
        o[region & ~res["clear"]] = [0.85, 0.3, 0.3]
        a.imshow(o); a.set_title(f"{name}: {res['water_ha']:.1f} ha\n(red = masked by SCL)")
    for a in ax.flat:
        if boundary is not None and boundary.any():
            a.contour(boundary.astype(float), levels=[0.5], colors="#f2d14b", linewidths=0.7)
        a.set_axis_off()
    import pandas as pd
    date = pd.Timestamp(np.asarray(scene.time.values).reshape(-1)[0]).strftime("%d %b %Y")
    fig.suptitle(f"Detection diagnostic — {date}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    stats["likely_cause"] = _verdict(stats)
    return stats


def _verdict(s: dict) -> str:
    """Judge on what's inside the lake, and reject any method that finds more
    water in the box than the lake outline can hold (it's counting land)."""
    w = s["water_ha_inside_lake"]
    box = s["water_ha_whole_box"]
    cap = s["region_ha"] if s["region"] == "lake boundary" else float("inf")
    fixed = max(w["mndwi_fixed_0"], 0.1)
    notes = []
    if box["mndwi_otsu"] > 1.1 * cap:
        notes.append(f"Otsu finds {box['mndwi_otsu']} ha in the box but the lake holds at most {cap} ha: "
                     "it's counting land, don't use it here.")
    if s["masked_by_scl_but_mndwi_says_water_ha"] > 2:
        cause = ("A: SCL masked water as shadow/dark area. Fix: trust the water index inside the "
                 "lake boundary even where SCL says shadow/dark area.")
    elif s["scl_says_water_in_region_ha"] > 1.25 * fixed or w.get("mndwi_lake", 0) > 1.25 * fixed:
        cause = (f"B: dark lake water sits just under the 0.0 threshold. Fixed finds {w['mndwi_fixed_0']} ha in the "
                 f"lake, the satellite's classifier says {s['scl_says_water_in_region_ha']} ha"
                 + (f", lake method finds {w['mndwi_lake']} ha" if "mndwi_lake" in w else "")
                 + ". Fix: --method lake (looser only inside the lake).")
    elif w["ndwi_fixed_0"] > 1.25 * fixed:
        cause = "B: MNDWI misses it but NDWI sees it. Fix: try --index ndwi for this lake."
    elif s["scl_says_water_in_region_ha"] <= 1.1 * fixed and w.get("mndwi_lake", fixed) <= 1.1 * fixed:
        cause = "OK: the fixed threshold already finds what the other methods and the satellite's classifier find."
    else:
        cause = "C: no index sees much water here. Likely haze/sensor issue; this date should probably be excluded."
    return " ".join([cause] + notes)
