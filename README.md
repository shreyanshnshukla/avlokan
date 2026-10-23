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
6. Rejects outliers: single dates that jump far from their neighbours
   (thin haze or cloud shadows the satellite's own mask missed).
7. Splits the lake into **open water, vegetation (weeds) and bare ground**
   inside a fixed lake boundary from JRC Global Surface Water (40 years of
   records). Open water alone can't tell a shrinking lake from a weedy one.
8. Compares like with like: post-monsoon peaks and pre-monsoon lows per year,
   plus trends in hectares/year.

Lakes configured: Sirpur, Bilawali, Yeshwant Sagar, Pipliyapala
([avlokan/config.py](avlokan/config.py)).

### Outputs (per lake, in `outputs/<lake>/`)

| File | What it is |
|---|---|
| `water_timeseries.csv` | Every date: open water, vegetated, bare, lake extent (ha), cloud-free share, outlier, usable |
| `yearly_summary.csv` | Per year: open-water peak/low, post-monsoon lake extent, weed share |
| `water_timeseries.png` | Open water over time (removed outliers shown as red x) |
| `lake_composition.png` | Monthly: open water vs vegetation vs bare ground inside the lake |
| `before_after.png` | Earliest vs latest post-monsoon image: water (blue), weeds (green), boundary (yellow) |
| `report.json` | Trends and run metadata |

### Reading the numbers

- **Lake extent falling** (post-monsoon) → the lake is actually shrinking.
- **Open water falling, extent steady** → it's getting weedier, not smaller.
- **Extent is only reported post-monsoon.** In summer, grass on the dry bed
  looks like weeds from space.

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

Options: `--index mndwi|ndwi`, `--method fixed|otsu|lake`, `--threshold 0.0`,
`--start`, `--end`, `--out`, `--no-jrc` (build the boundary from the run's own
history instead).

**`--method lake`** (experimental, needs the JRC boundary): strict 0.0 threshold
outside the lake; inside it, a pixel is water if its index is above -0.15 or
the satellite's own classifier calls it water, and never if it's vegetation.
Built for dark, algae-rich water like Sirpur's, which sits just under 0.0.
Wet mud at the edge of a drying lake may be counted as water.
Avoid `--method otsu` on lakes in cities: it can count land as water.

Diagnose a single date (why was water missed or over-counted?):

```bash
python scripts/diagnose_date.py --lake sirpur --date 2023-10-06
```

It compares the satellite's own cloud/shadow labels, two water indices and
two thresholds, and prints the likely cause.

Or open the walkthrough notebook:
[notebooks/01_water_bodies_indore.ipynb](notebooks/01_water_bodies_indore.ipynb).
It runs offline on synthetic data first, then on real imagery.

## Phases 2-4 (built, first real runs pending)

All three work on a city grid (Indore, ~24 x 24 km) and share yearly
cloud-free composites, so build those first.

```bash
pip install -r requirements.txt            # adds scipy
python scripts/build_composites.py --area indore --years 2019-2026
```

**Phase 2: construction detection.** A U-Net learns "what built-up looks
like" from ESA WorldCover 2021 labels, then maps built-up land for every
year. New construction must persist a year to count (no flicker).

```bash
pip install -r requirements-ml.txt         # PyTorch; read the file first (large download)
python scripts/p2_make_dataset.py --area indore --year 2021
python scripts/p2_train.py --data data/p2/indore_2021.npz
python scripts/p2_detect_change.py --area indore --years 2019-2026
# no GPU / no PyTorch yet? rule-based baseline:
python scripts/p2_detect_change.py --area indore --years 2019-2026 --method index
```

**Phase 3: encroachment.** New construction inside restricted zones (every
water body + buffer, automatically; plus your own zones from `zones/`)
becomes a flag with location, size, first-seen year and confidence.
Flags are candidates for review, never findings of illegality.

```bash
python scripts/p3_flag_encroachment.py --area indore
```

**Phase 4: floods.** Sentinel-1 radar sees through monsoon clouds. Flood
history (how often each place flooded, 2019 onwards) plus terrain gives a
screening risk map with prevention guidance per risk level.

```bash
python scripts/p4_flood.py --area indore --years 2019-2026
python scripts/p4_flood.py --area indore --event 2025-08-14   # one event
```

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
  geo.py               grid metadata, save/load layers
  data/composite.py    yearly cloud-free composites (Phases 2-4)
  construction/        Phase 2: labels, patches, U-Net, training, change
  encroachment/        Phase 3: zones, flags
  flood/               Phase 4: radar flood mapping, terrain risk
scripts/               command-line entry points (p2_*, p3_*, p4_* per phase)
zones/                 your restricted-zone GeoJSON files (Phase 3)
notebooks/             walkthroughs
tests/                 unit + end-to-end tests (offline)
docs/ROADMAP.md        all phases
```

## Caveats

- Sirpur's box is verified. The other lake boxes are from public sources and
  not yet checked; Pipliyapala's centre is approximate. `report.json` says `bbox_verified: false`
  until you flip `verified=True` in `config.py`.
- Sentinel-2 is 10 m per pixel: good for lakes and colonies, not single
  buildings.
- Area is what the satellite sees as open water. Lakes covered in weeds
  or hyacinth read as smaller than their actual extent.

## Data credits

Contains modified Copernicus Sentinel data, accessed via Microsoft Planetary
Computer.
