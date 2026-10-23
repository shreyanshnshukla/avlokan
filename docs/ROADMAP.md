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
- [x] Diagnose the dark-water miss (Sirpur 2023-10-06): index just under 0.0 inside the lake;
      whole-scene Otsu over-counts land, so not a fix
- [ ] Compare `--method lake` vs `fixed` on full history; pick the default
- [x] Sirpur box verified; [ ] other lakes
- [ ] Alert when a lake is below its normal size for the season
- [ ] Validate against known figures (Ramsar/municipal records)

## Shared
- [x] Yearly cloud-free composites (dry season, median, cloud-masked)

## Phase 2 — Construction detection (first real ML)
- [x] Labels from ESA WorldCover 2021; spatially split training patches
- [x] U-Net (sized for RTX 3050), Dice+BCE, early stopping on val IoU
- [x] Sliding-window inference; rule-based (NDBI) baseline for comparison
- [x] Change detection with persistence; new construction per year; map
- [ ] First real training run on Indore; beat the index baseline
- [ ] Azure ML for larger runs; growth per ward

## Phase 3 — Encroachment and illegal construction
- [x] Water-body zones + buffer from JRC (all lakes, ponds, rivers in the area)
- [x] Custom zones from GeoJSON (green belts, master-plan zones)
- [x] Flags: location, size, first-seen year, confidence, "flagged_for_review"; CSV + GeoJSON + map
- [ ] Real buffer widths from applicable rules; sourced zone boundaries
- [ ] Remote-area monitoring: forests, hills, river islands

## Phase 4 — Floods
- [x] Flood mapping from Sentinel-1 RTC radar (despeckle, threshold, minus permanent water)
- [x] Flood frequency across monsoons; single-event maps
- [x] Screening risk: flood history + relative elevation + slope (Copernicus DEM); guidance per level
- [x] Link to Phase 2: new construction in high-risk areas
- [ ] First real run; check against known flood events
- [ ] Rainfall; construction blocking natural drainage

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
