# Restricted zones

Put GeoJSON files here describing areas where new construction should be
flagged for review (green belts, forest land, master-plan open spaces,
riverbanks). Water bodies are added automatically from JRC data.

How to make one:
1. Open https://geojson.io and draw polygons over the zones.
2. For each polygon, add a property `name` (e.g. "Green belt - Rau").
3. Save as `zones/<area>_zones.geojson`.

Then run:

    python scripts/p3_flag_encroachment.py --area indore --zones zones/indore_zones.geojson

Use only zone boundaries you have a source for (master plan maps, notified
boundaries). A wrong zone boundary produces wrong flags.
