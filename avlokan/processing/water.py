"""Water segmentation (rule-based baseline) and area measurement.

Phase 1 uses index thresholds. This gives a reliable baseline and, later,
training labels to compare the U-Net against.
"""

from __future__ import annotations

import numpy as np

from ..config import SCL_CLEAR_CLASSES


def clear_mask(scl: np.ndarray) -> np.ndarray:
    """True where the pixel is not cloud, shadow, cirrus, snow or no-data."""
    return np.isin(scl, SCL_CLEAR_CLASSES)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on finite values. Picks the split that best separates
    water from land for this particular scene."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0, dtype=float), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0, dtype=float), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    # When water and land are well separated, every empty bin between them
    # scores the same. Take the middle of that plateau, not its first edge.
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(centres[best].mean())


def water_mask(
    index: np.ndarray,
    clear: np.ndarray | None = None,
    method: str = "fixed",
    threshold: float = 0.0,
    otsu_bounds: tuple[float, float] = (-0.2, 0.3),
) -> tuple[np.ndarray, float]:
    """Classify water pixels from an index image (NDWI or MNDWI).

    method="fixed": index > threshold (0.0 is the textbook default).
    method="otsu":  scene-adaptive threshold, clamped to otsu_bounds so a
                    scene with almost no water can't produce a silly split.
    Returns (mask, threshold_used). Masked-out (cloudy) pixels are False.
    """
    idx = np.asarray(index, dtype="float32")
    valid = np.isfinite(idx)
    if clear is not None:
        valid &= clear
    if method == "otsu":
        t = otsu_threshold(idx[valid])
        if not np.isfinite(t):
            t = threshold
        t = float(np.clip(t, *otsu_bounds))
    elif method == "fixed":
        t = threshold
    else:
        raise ValueError(f"unknown method {method!r}")
    return (idx > t) & valid, t


def lake_water_mask(
    index: np.ndarray,
    scl: np.ndarray,
    clear: np.ndarray,
    boundary: np.ndarray,
    base_threshold: float = 0.0,
    lake_threshold: float = -0.15,
    ndvi: np.ndarray | None = None,
    veg_ndvi: float = 0.4,
) -> tuple[np.ndarray, float]:
    """Water mask that is strict outside the lake and looser inside it.

    Why: dark, algae-rich lake water can sit just below index 0.0 (Sirpur,
    Oct 2023: median MNDWI -0.10 inside the lake), but lowering the threshold
    everywhere turns roads and fields into "water". Inside the known lake
    boundary we can be looser because the land question is already settled.

    Inside the boundary a pixel is water if it is not vegetation
    (NDVI <= veg_ndvi, so floating weeds never count) and either:
      - its index is above `lake_threshold` (-0.15), or
      - the satellite's own classifier (SCL) labels it water.
    Outside the boundary the strict `base_threshold` applies.
    Returns (mask, lake_threshold).
    """
    idx = np.asarray(index, dtype="float32")
    valid = np.isfinite(idx) & clear
    inside = np.asarray(boundary, bool) & valid
    if ndvi is not None:
        inside &= ~(np.asarray(ndvi) > veg_ndvi)
    base = (idx > base_threshold) & valid
    lake = inside & ((idx > lake_threshold) | (np.asarray(scl) == 6))
    return base | lake, float(lake_threshold)


def area_hectares(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    """Pixel count x pixel area, in hectares (1 ha = 10,000 m²)."""
    return float(np.count_nonzero(mask)) * pixel_size_m**2 / 10_000.0
