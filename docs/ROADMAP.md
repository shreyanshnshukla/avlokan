# Avlokan roadmap

Phases 0-3 plus a basic admin panel make a complete product. Later phases
extend it.

## Phase 0 — Foundation
- [x] Sentinel-2 pipeline via Planetary Computer, cloud masking, reflectance offset fix
- [ ] Sentinel-1 radar (sees through monsoon clouds)
- [ ] Elevation data (Copernicus DEM), Sentinel-5P (air pollution)
- [ ] Scheduled refresh every few days (Azure Functions)
- [ ] Geospatial database for detections and zones (PostGIS)

## Phase 1 — Water bodies  ← current
- [x] Water detection (MNDWI, fixed/Otsu), lake-aware cloud filtering
- [x] Area time series, seasonal peaks/lows, trend per year, before/after maps
- [x] First real run (Sirpur, 2023-2026): 158 scenes, pipeline works end to end
- [x] Outlier filter (haze / missed shadows)
- [x] Lake composition: open water / vegetation / bare inside a JRC boundary
- [x] Per-date diagnostic tool
- [ ] Fix the dark-water miss (cause to be confirmed with the diagnostic)
- [x] Sirpur box verified; [ ] other lakes
- [ ] Alert when a lake is below its normal size for the season
- [ ] Validate against known figures (Ramsar/municipal records)

## Phase 2 — Construction detection (first real ML)
- [ ] Built-up segmentation with U-Net (labels from ESA WorldCover / Dynamic World)
- [ ] Train locally (RTX 3050) for experiments, Azure ML for full runs
- [ ] Change detection: new construction by month/year, growth per ward

## Phase 3 — Encroachment and illegal construction
- [ ] Restricted zones: lake buffers, riverbanks, green belts, forest land, master-plan zones
- [ ] Flag new construction inside them ("flagged for review", never "illegal")
- [ ] Remote-area monitoring: forests, hills, river islands

## Phase 4 — Floods
- [ ] Live flood mapping with Sentinel-1 radar
- [ ] Flood risk from elevation, drainage, rainfall history
- [ ] Construction blocking natural drainage

## Phase 5 — Landslides
- [ ] Susceptibility maps (slope, soil, vegetation loss, rainfall)
- [ ] Fresh landslide scars after heavy rain (showcase region: Uttarakhand / Western Ghats)

## Phase 6 — More environmental risks
- [ ] Green cover loss, illegal sand mining, garbage dumps
- [ ] Urban heat islands, air pollution (NO₂), crop stress, fires

## Phase 7 — Prevention and caution engine
- [ ] Risk score per area; caution advice per zone; pre-monsoon reports
- [ ] **Sanjaya**: AI assistant in the panel that explains findings in plain language (Azure OpenAI)

## Phase 8 — Admin panel
- [ ] Map with layers, before/after slider, timeline
- [ ] Alert queue: review, approve, dismiss
- [ ] Dashboards; roles (admin, reviewer, department official)

## Phase 9 — Official notifications (human-approved only)
- [ ] Report: before/after images, coordinates, dates, area, confidence
- [ ] Route to the right department; email/SMS; status tracking

## Phase 10 — Public side (optional)
- [ ] Location risk check before buying land
- [ ] Citizen reports cross-checked by satellite
- [ ] Public map of verified violations
