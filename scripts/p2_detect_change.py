"""Phase 2, step 3: built-up map for every year + new construction.

    python scripts/build_composites.py --area indore --years 2019-2026
    python scripts/p2_detect_change.py --area indore --years 2019-2026

Uses the trained U-Net (models/builtup_unet.pt). Use --method index to run
the no-ML baseline instead (works without PyTorch).

Writes outputs/<area>/construction/: construction.npz (first-built year,
probabilities), construction_by_year.csv, new_construction.png, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from avlokan.config import get_area  # noqa: E402


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="indore")
    p.add_argument("--years", default="2019-2026")
    p.add_argument("--composites", default="outputs/composites")
    p.add_argument("--model", default="models/builtup_unet.pt")
    p.add_argument("--method", choices=["unet", "index"], default="unet")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--persist", type=int, default=1, help="years a change must persist to count")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    from avlokan.construction.change import construction_table, first_built_year
    from avlokan.construction.labels import index_baseline
    from avlokan.data.composite import stack_bands
    from avlokan.geo import load_layers, save_layers
    from avlokan.viz.maps import plot_construction_years, rgb_from_bands

    aoi = get_area(a.area)
    years = [y for y in parse_years(a.years) if (Path(a.composites) / f"{aoi.key}_{y}.npz").exists()]
    missing = sorted(set(parse_years(a.years)) - set(years))
    if missing:
        print(f"No composite for {missing}; build them with scripts/build_composites.py. Continuing without.")
    if len(years) < 2:
        print("Need at least two years of composites.")
        return 1

    model = None
    if a.method == "unet":
        try:
            from avlokan.construction.train import load_model, predict_probability
        except ImportError:
            print("PyTorch isn't installed; use --method index or see requirements-ml.txt.")
            return 1
        model, mean, std, dev = load_model(a.model)
        from avlokan.construction.dataset import normalise

    built, valid, probs, grid, last = {}, {}, {}, None, None
    for y in years:
        grid_y, layers = load_layers(Path(a.composites) / f"{aoi.key}_{y}.npz")
        if grid is None:
            grid = grid_y
        elif grid_y.shape != grid.shape:
            print(f"{y}: grid differs from {years[0]}; skipping")
            continue
        ok = np.isfinite(stack_bands(layers)).all(axis=0)
        if model is not None:
            prob = predict_probability(model, normalise(stack_bands(layers), mean, std), dev)
            probs[y] = (prob * 255).astype("uint8")
            b = prob > a.threshold
        else:
            b = index_baseline(layers)
        built[y], valid[y], last = b & ok, ok, layers
        print(f"{y}: built-up {float(built[y].sum()) * grid.pixel_area_m2 / 1e4:,.0f} ha, "
              f"{ok.mean():.0%} of pixels observed")

    first = first_built_year(built, valid, persist=a.persist)
    table = construction_table(first, built, grid.pixel_area_m2)
    out = Path(a.out) / aoi.key / "construction"
    out.mkdir(parents=True, exist_ok=True)
    layers_out = {"first_built_year": first}
    if probs:
        layers_out["prob_last_year"] = probs[max(probs)]
    save_layers(out / "construction.npz", grid, **layers_out)
    table.to_csv(out / "construction_by_year.csv", index=False)
    plot_construction_years(rgb_from_bands(last), first, f"{aoi.name}: new construction by year ({a.method})",
                            out / "new_construction.png")
    summary = {
        "area": aoi.key, "method": a.method, "years": list(built), "persist": a.persist,
        "new_construction_ha_total": round(float(table["new_construction_ha"].sum()), 1),
        "built_up_ha_first_year": float(table["built_up_ha"].iloc[0]),
        "built_up_ha_last_year": float(table["built_up_ha"].iloc[-1]),
        "note": "The last year can't be checked for persistence; its new construction is provisional.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
