"""Map figures for Phases 2-4."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def rgb_from_bands(b: dict, gain: float = 3.0) -> np.ndarray:
    img = np.dstack([b["B04"], b["B03"], b["B02"]])
    return np.clip(np.nan_to_num(img) * gain, 0, 1)


def _save(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_construction_years(rgb: np.ndarray, first_year: np.ndarray, title: str, path=None):
    """True colour background; old built-up grey; new construction coloured by year."""
    years = sorted(int(y) for y in np.unique(first_year) if y > 0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb * 0.6)
    old = np.ma.masked_where(first_year != -1, np.ones_like(first_year))
    ax.imshow(old, cmap=ListedColormap(["#9a9a9a"]), alpha=0.55, interpolation="nearest")
    handles = [Patch(color="#9a9a9a", label="built before")]
    if years:
        cmap = plt.get_cmap("plasma", len(years))
        norm = BoundaryNorm(np.arange(len(years) + 1) - 0.5, len(years))
        idx = np.full(first_year.shape, -1)
        for i, y in enumerate(years):
            idx[first_year == y] = i
        ax.imshow(np.ma.masked_less(idx, 0), cmap=cmap, norm=norm, interpolation="nearest")
        handles += [Patch(color=cmap(i), label=str(y)) for i, y in enumerate(years)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85, title="New construction")
    ax.set_title(title)
    ax.set_axis_off()
    return _save(fig, path)


def plot_flags(rgb: np.ndarray, zones: dict[str, np.ndarray], flags_mask: np.ndarray, title: str, path=None):
    """Restricted zones outlined, flagged construction in red."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb * 0.6)
    colours = plt.get_cmap("tab10")
    handles = []
    for i, (name, z) in enumerate(zones.items()):
        if z.any():
            ax.contour(z.astype(float), levels=[0.5], colors=[colours(i % 10)], linewidths=0.8)
            handles.append(Patch(edgecolor=colours(i % 10), facecolor="none", label=name))
    ax.imshow(np.ma.masked_where(~flags_mask, np.ones_like(flags_mask, float)),
              cmap=ListedColormap(["#e0302b"]), interpolation="nearest")
    handles.append(Patch(color="#e0302b", label="flagged for review"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(title)
    ax.set_axis_off()
    return _save(fig, path)


def plot_flood(rgb: np.ndarray, freq: np.ndarray, permanent: np.ndarray, title: str, path=None):
    """How often each pixel flooded during monsoon scenes."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb * 0.6)
    shown = np.ma.masked_where((freq <= 0) | permanent, freq)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(0.3, float(np.nanmax(freq)) if freq.size else 1),
                   interpolation="nearest")
    ax.imshow(np.ma.masked_where(~permanent, np.ones_like(freq)), cmap=ListedColormap(["#123a6b"]),
              interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.035, label="share of monsoon scenes flooded")
    ax.legend(handles=[Patch(color="#123a6b", label="permanent water")], loc="lower left", fontsize=8)
    ax.set_title(title)
    ax.set_axis_off()
    return _save(fig, path)


RISK_COLOURS = ["#2c9c4a", "#f2c14e", "#e0302b"]


def plot_risk(rgb: np.ndarray, risk_class: np.ndarray, title: str, path=None):
    """risk_class: 0 low, 1 medium, 2 high, -1 no data / water."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb * 0.6)
    ax.imshow(np.ma.masked_less(risk_class, 0), cmap=ListedColormap(RISK_COLOURS), vmin=-0.5, vmax=2.5,
              alpha=0.55, interpolation="nearest")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(RISK_COLOURS, ["low", "medium", "high"])],
              loc="lower left", fontsize=8, title="Flood risk")
    ax.set_title(title)
    ax.set_axis_off()
    return _save(fig, path)
