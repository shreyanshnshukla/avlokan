"""Charts and before/after maps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from ..processing.indices import to_reflectance  # noqa: E402

SEASON_COLOURS = {
    "post_monsoon": "#1f6fb2",
    "winter": "#5aa0a8",
    "pre_monsoon": "#c9772b",
    "monsoon": "#9a9a9a",
}


def plot_timeseries(ts: pd.DataFrame, title: str, path=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ok = ts[ts["usable"]]
    for season, grp in ok.groupby("season"):
        ax.scatter(grp["date"], grp["water_ha"], s=16, label=season.replace("_", "-"),
                   color=SEASON_COLOURS.get(season, "#555"))
    if len(ok) > 2:
        smooth = ok.set_index("date")["water_ha"].rolling("120D", min_periods=1).median()
        ax.plot(smooth.index, smooth.values, color="#222", lw=1.2, alpha=0.7, label="120-day median")
    ax.set_title(title)
    ax.set_ylabel("Water surface (hectares)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def rgb(scene: xr.Dataset, gain: float = 3.0) -> np.ndarray:
    pb = float(scene["processing_baseline"].item()) if "processing_baseline" in scene.coords else None
    bands = [to_reflectance(scene[b].values, pb) for b in ("B04", "B03", "B02")]
    return np.clip(np.dstack(bands) * gain, 0, 1)


def plot_before_after(before: xr.Dataset, after: xr.Dataset, mask_before, mask_after, title: str, path=None):
    """2x2: true colour on top, detected water below, for two dates."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for col, (scene, mask) in enumerate(((before, mask_before), (after, mask_after))):
        date = pd.Timestamp(np.asarray(scene.time.values).reshape(-1)[0]).strftime("%d %b %Y")
        axes[0, col].imshow(rgb(scene))
        axes[0, col].set_title(f"{date} — true colour")
        overlay = rgb(scene) * 0.55
        overlay[mask] = [0.12, 0.45, 0.85]
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{date} — water detected")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
