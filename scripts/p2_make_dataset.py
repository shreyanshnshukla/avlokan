"""Phase 2, step 1: training patches from a composite + WorldCover labels.

    python scripts/build_composites.py --area indore --years 2021
    python scripts/p2_make_dataset.py --area indore --year 2021

Writes data/p2/<area>_<year>.npz (patches, labels, train/val split, band stats).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from avlokan.config import get_area  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="indore")
    p.add_argument("--year", type=int, default=2021, help="WorldCover exists for 2020 and 2021")
    p.add_argument("--composites", default="outputs/composites")
    p.add_argument("--out", default="data/p2")
    p.add_argument("--patch", type=int, default=128)
    a = p.parse_args()

    from avlokan.construction.dataset import band_stats, make_patches, normalise
    from avlokan.construction.labels import builtup_label, index_baseline, load_worldcover
    from avlokan.data.composite import stack_bands
    from avlokan.geo import load_layers, save_layers

    aoi = get_area(a.area)
    comp = Path(a.composites) / f"{aoi.key}_{a.year}.npz"
    if not comp.exists():
        print(f"Missing {comp}. Run: python scripts/build_composites.py --area {a.area} --years {a.year}")
        return 1
    grid, layers = load_layers(comp)
    x = stack_bands(layers)
    print(f"composite {grid.shape}, loading WorldCover {a.year} on the same grid...")
    wc = load_worldcover(aoi, grid, a.year)
    label = builtup_label(wc)
    save_layers(Path(a.out) / f"{aoi.key}_{a.year}_labels.npz", grid, worldcover=wc, label=label)

    mean, std = band_stats(x)
    d = make_patches(normalise(x, mean, std), label, size=a.patch, stride=a.patch)
    out = Path(a.out) / f"{aoi.key}_{a.year}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mean=mean, std=std, **d)

    lab_ok = label != 255
    base = index_baseline(layers)
    inter = (base & (label == 1) & lab_ok).sum()
    union = ((base | (label == 1)) & lab_ok).sum()
    info = {
        "patches_train": int((d["split"] == "train").sum()),
        "patches_val": int((d["split"] == "val").sum()),
        "built_up_share": round(float((label == 1)[lab_ok].mean()), 3),
        "index_baseline_iou_vs_worldcover": round(float(inter / union), 3) if union else None,
    }
    print(json.dumps(info, indent=2))
    print(f"-> {out}\nThe model should beat index_baseline_iou_vs_worldcover to be worth it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
