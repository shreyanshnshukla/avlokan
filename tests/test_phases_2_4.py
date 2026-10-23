"""Offline tests for composites (shared), Phase 2 (construction),
Phase 3 (encroachment) and Phase 4 (flood). No network, no GPU."""

import json

import numpy as np
import pytest

from avlokan.construction.change import construction_table, first_built_year
from avlokan.construction.dataset import band_stats, make_patches, normalise
from avlokan.construction.labels import builtup_label, index_baseline
from avlokan.data.composite import median_composite, stack_bands
from avlokan.encroachment.flags import find_flags, to_geojson
from avlokan.encroachment.zones import buffer_zone, water_bodies
from avlokan.flood.risk import relative_elevation, risk_classes, risk_score, slope_deg
from avlokan.flood.sar import FloodCounter, despeckle, flood_from_water, sar_water, to_db
from avlokan.geo import Grid, load_layers, save_layers

GRID = Grid(crs="EPSG:32643", transform=(10.0, 0.0, 700000.0, 0.0, -10.0, 2520000.0), shape=(60, 80))


# ---------- shared: grid + composites ----------

def test_grid_math_and_roundtrip(tmp_path):
    x, y = GRID.pixel_to_xy(0, 0)
    assert (x, y) == (700005.0, 2519995.0) and GRID.pixel_area_m2 == 100.0
    path = save_layers(tmp_path / "a.npz", GRID, foo=np.arange(6).reshape(2, 3))
    g, layers = load_layers(path)
    assert g == GRID and layers["foo"].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_median_composite_masks_clouds_and_offset():
    dn = np.array([[[1500]], [[9000]], [[2600]], [[2400]]], dtype="uint16")   # 9000 = cloud
    scl = np.array([[[4]], [[9]], [[4]], [[4]]])
    pb = [2.0, 2.0, 4.0, 4.0]                                                 # last two carry +1000
    med, n = median_composite(dn, scl, pb)
    assert n[0, 0] == 3 and med[0, 0] == pytest.approx(0.15)               # median of .15, .16, .14
    med2, _ = median_composite(dn[:2], scl[:2], pb[:2], min_obs=2)
    assert np.isnan(med2[0, 0])                                             # only one clear look


# ---------- Phase 2 ----------

def _bands(h=40, w=40, built=None):
    built = np.zeros((h, w), bool) if built is None else built
    base = {"B02": 0.08, "B03": 0.10, "B04": 0.08, "B08": 0.35, "B11": 0.20, "B12": 0.12}   # vegetation
    urb = {"B02": 0.14, "B03": 0.15, "B04": 0.17, "B08": 0.22, "B11": 0.28, "B12": 0.25}    # built-up
    return {b: np.where(built, urb[b], base[b]).astype("float32") for b in base}


def test_index_baseline_and_labels():
    built = np.zeros((40, 40), bool)
    built[10:20, 10:30] = True
    assert (index_baseline(_bands(built=built)) == built).all()
    assert builtup_label(np.array([50, 10, 0, 80])).tolist() == [1, 0, 255, 0]


def test_patches_spatial_split_and_skip_cloudy():
    x = stack_bands(_bands(256, 256))
    x[:, :128, :128] = np.nan                       # one cloudy quadrant
    label = np.zeros((256, 256), "uint8")
    mean, std = band_stats(x)
    d = make_patches(normalise(x, mean, std), label, size=64, stride=64, val_block=2)
    assert d["X"].shape[1:] == (6, 64, 64)
    assert len(d["X"]) <= 16 and np.isfinite(d["X"]).all()
    assert set(d["split"]) <= {"train", "val"} and (d["split"] == "val").any()


def test_new_construction_needs_to_persist():
    s = (1, 4)
    old, new, flicker, gap = (np.zeros(s, bool) for _ in range(4))
    built = {y: np.zeros(s, bool) for y in (2019, 2020, 2021, 2022)}
    built[2019][0, 0] = built[2020][0, 0] = built[2021][0, 0] = built[2022][0, 0] = True   # old building
    built[2021][0, 1] = built[2022][0, 1] = True                                          # new in 2021
    built[2020][0, 2] = True                                                               # flicker
    built[2020][0, 3] = built[2022][0, 3] = True                                           # 2021 cloudy
    valid = {y: np.ones(s, bool) for y in built}
    valid[2021][0, 3] = False
    fy = first_built_year(built, valid, persist=1)
    assert fy.tolist() == [[-1, 2021, 0, 2020]]
    t = construction_table(fy, built, 100.0)
    assert t.set_index("year").loc[2021, "new_construction_ha"] == 0.01


def test_unet_shapes_and_seamless_inference():
    torch = pytest.importorskip("torch")
    from avlokan.construction.model import UNet, dice_bce_loss
    from avlokan.construction.train import predict_probability

    m = UNet(in_ch=6, base=8).eval()
    x = torch.randn(2, 6, 64, 64)
    y = torch.zeros(2, 64, 64)
    assert m(x).shape == (2, 64, 64)
    assert torch.isfinite(dice_bce_loss(m(x), y, torch.ones_like(y, dtype=torch.bool)))
    p = predict_probability(m, np.zeros((6, 150, 170), "float32"), "cpu", tile=64, overlap=8)
    assert p.shape == (150, 170) and np.ptp(p) < 1e-4          # constant input -> no tile seams


# ---------- Phase 3 ----------

def test_water_bodies_and_buffer():
    occ = np.zeros((50, 50))
    occ[20:30, 20:30] = 80
    occ[2, 2] = 90                                               # speck
    w = water_bodies(occ, min_area_px=5)
    assert w.sum() == 100 and not w[2, 2]
    z = buffer_zone(w, 30, 10)                                   # 3 px
    assert z[20:30, 20:30].all() and z[25, 17] and not z[25, 15]


def test_flags_inside_zone_only_with_min_area(tmp_path):
    fy = np.zeros((60, 80), "int16")
    fy[10:15, 10:15] = 2023                                       # 25 px inside zone
    fy[40:45, 60:65] = 2024                                       # outside zone
    fy[12, 30] = 2025                                             # 1 px: too small
    fy[5:8, 5:8] = -1                                             # old construction: never flagged
    zone = np.zeros((60, 80), bool)
    zone[0:30, 0:40] = True
    prob = np.full((60, 80), 204, "uint8")
    df = find_flags(fy, {"lake buffer": zone}, GRID, prob, min_area_m2=500)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["first_seen_year"] == 2023 and r["area_m2"] == 2500 and r["confidence"] == pytest.approx(0.8)
    assert r["status"] == "flagged_for_review" and r["flag_id"] == "AVL-0001"
    assert find_flags(fy, {"z": zone}, GRID, since_year=2024).empty
    df = df.assign(lon=75.8, lat=22.7)
    gj = json.loads(to_geojson(df, tmp_path / "f.geojson").read_text())
    assert gj["features"][0]["geometry"]["coordinates"] == [75.8, 22.7]
    assert gj["features"][0]["properties"]["zone"] == "lake buffer"


# ---------- Phase 4 ----------

def test_sar_water_and_flood():
    rng = np.random.default_rng(0)
    db = rng.normal(-8, 1.5, (60, 60)).astype("float32")         # land
    db[10:30, 10:30] = rng.normal(-23, 1.5, (20, 20))            # water
    assert np.isnan(to_db(np.array([0.0]))[0]) and to_db(np.array([0.1]))[0] == pytest.approx(-10)
    w, t = sar_water(despeckle(db))
    assert t == -18 and w[12:28, 12:28].all() and not w[40:, 40:].any()
    w2, t2 = sar_water(despeckle(db), method="otsu")
    assert -22 <= t2 <= -14 and (w2 == w).mean() > 0.98
    perm = np.zeros_like(w)
    perm[10:20, 10:30] = True                                    # half is the river
    speck = np.zeros_like(w)
    speck[50, 50] = True
    f = flood_from_water(w | speck, perm, min_px=10)
    assert f[25, 20] and not f[15, 20] and not f[50, 50]


def test_flood_counter_frequency():
    fc = FloodCounter((1, 3))
    fc.add(np.array([[1, 0, 1]], bool), np.array([[1, 1, 0]], bool))
    fc.add(np.array([[1, 0, 0]], bool), np.array([[1, 1, 0]], bool))
    f = fc.frequency()
    assert f[0, 0] == 1.0 and f[0, 1] == 0.0 and np.isnan(f[0, 2])


def test_risk_prefers_low_flat_flooded_ground():
    yy, xx = np.mgrid[:100, :100]
    dem = (300 + 0.002 * ((yy - 50) ** 2 + (xx - 50) ** 2)).astype("float32")   # bowl, low centre
    rel = relative_elevation(dem, window_m=1000, pixel_m=10)
    assert rel[50, 50] < rel[0, 0] and rel.min() >= -0.5
    sl = slope_deg(dem, 10)
    freq = np.zeros_like(dem)
    freq[45:55, 45:55] = 0.5
    score = risk_score(freq, rel, sl)
    assert score[50, 50] > score[0, 0]
    cls = risk_classes(score, exclude=(yy == 0))
    assert cls[50, 50] == 2 and (cls[0] == -1).all()
    no_hist = risk_score(None, rel, sl)
    assert no_hist[50, 50] > no_hist[0, 0] and no_hist.max() <= 1
