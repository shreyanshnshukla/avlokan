# Avlokan

**Divya Drishti for every city.**

Like Sanjaya, who watched the whole of Kurukshetra from afar and reported to
the king, Avlokan (अवलोकन, *observation*) watches a city and its surroundings
from space. It detects risks and violations with machine learning (shrinking
lakes, encroachment, new construction in protected zones, floods,
landslides) and gives officials an admin panel to review findings and act.

It runs on free Sentinel satellite data, so it's affordable for the
tier-2 cities that don't have a monitoring system today.

> Status: **Phase 1 — water bodies** (Indore). See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Phase 1: how much water do Indore's lakes hold, and is it shrinking?

For each lake, Avlokan:

1. Finds every Sentinel-2 image since 2017 on Microsoft Planetary Computer
   (free, no key needed).
2. Masks clouds and shadows using the scene classification layer.
3. Removes the 2022 reflectance offset ESA introduced, so the time series
   doesn't jump.
4. Detects water with MNDWI (fixed or Otsu threshold).
5. Rejects dates where clouds cover part of the lake itself (a small cloud
   over water would otherwise look like the lake shrinking).
6. Compares like with like: post-monsoon peaks and pre-monsoon lows per year,
   plus a trend in hectares/year.

Lakes configured: Sirpur, Bilawali, Yeshwant Sagar, Pipliyapala
([avlokan/config.py](avlokan/config.py)).

### Outputs (per lake, in `outputs/<lake>/`)

| File | What it is |
|---|---|
| `water_timeseries.csv` | Every date: water area (ha), cloud-free share, usable or not |
| `yearly_summary.csv` | Post-monsoon peak and pre-monsoon low per year |
| `water_timeseries.png` | The chart |
| `before_after.png` | Earliest vs latest post-monsoon image, with detected water |
| `report.json` | Trends and run metadata |

---

## Setup

Python 3.10 or 3.11 recommended.

```bash
git clone https://github.com/shreyanshnshukla/avlokan.git
cd avlokan
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q          # should print 12 passed
```

On Windows, if `rasterio` fails to install with pip, install it via conda
(`conda install -c conda-forge rasterio`) and rerun.

## Run

```bash
# 1. Check the lake box on a map first (paste the output into geojson.io)
python scripts/run_water_timeseries.py --lake sirpur --show-bbox

# 2. Run one lake (start small: a couple of years)
python scripts/run_water_timeseries.py --lake sirpur --start 2023-01-01

# 3. Full history for every lake
python scripts/run_water_timeseries.py --lake all
```

Options: `--index mndwi|ndwi`, `--method fixed|otsu`, `--threshold 0.0`,
`--start`, `--end`, `--out`.

Or open the walkthrough notebook:
[notebooks/01_water_bodies_indore.ipynb](notebooks/01_water_bodies_indore.ipynb).
It runs offline on synthetic data first, then on real imagery.

## Project layout

```
avlokan/
  config.py            lakes (AOIs) and data settings
  data/stac.py         search + load Sentinel-2 from Planetary Computer
  data/synthetic.py    fake image stacks for tests and offline demos
  processing/          reflectance, water indices, cloud mask, water mask
  analysis/            time series, seasonal summary, trends
  viz/                 charts and before/after maps
  pipeline.py          end-to-end run for one lake
scripts/               command-line entry points
notebooks/             walkthroughs
tests/                 unit + end-to-end tests (offline)
docs/ROADMAP.md        all phases
```

## Caveats

- Lake boxes are from public sources and not yet checked on a map;
  Pipliyapala's centre is approximate. `report.json` says `bbox_verified: false`
  until you flip `verified=True` in `config.py`.
- Sentinel-2 is 10 m per pixel: good for lakes and colonies, not single
  buildings.
- Area is what the satellite sees as open water. Lakes covered in weeds
  or hyacinth read as smaller than their actual extent.

## Data credits

Contains modified Copernicus Sentinel data, accessed via Microsoft Planetary
Computer.
