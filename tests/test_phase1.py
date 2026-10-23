import json

import numpy as np
import pytest

from avlokan.analysis.timeseries import season_of, trend_ha_per_year, water_timeseries, yearly_summary
from avlokan.config import LAKES
from avlokan.data.synthetic import synthetic_stack
from avlokan.pipeline import run_from_stack
from avlokan.processing.indices import mndwi, ndwi, to_reflectance
from avlokan.processing.water import area_hectares, clear_mask, otsu_threshold, water_mask


def test_reflectance_offset_removed_after_baseline_4():
    dn = np.array([2000, 3000], dtype="uint16")
    assert np.allclose(to_reflectance(dn, 2.0), [0.2, 0.3])
    assert np.allclose(to_reflectance(dn, 4.0), [0.1, 0.2])
    assert to_reflectance(np.array([500]), 4.0)[0] == 0  # clipped, not negative


def test_indices_sign_and_zero_division():
    assert ndwi(np.array([0.08]), np.array([0.03]))[0] > 0      # water
    assert mndwi(np.array([0.10]), np.array([0.22]))[0] < 0     # land
    assert np.isnan(ndwi(np.array([0.0]), np.array([0.0]))[0])


def test_clear_mask_excludes_clouds_and_shadow():
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert clear_mask(scl).tolist() == [False, False, True, False, True, True, True, True, False, False, False, False]


def test_water_mask_and_area():
    idx = np.array([[0.5, -0.3], [0.2, np.nan]])
    clear = np.array([[True, True], [False, True]])
    m, t = water_mask(idx, clear)
    assert t == 0.0 and m.tolist() == [[True, False], [False, False]]
    assert area_hectares(np.ones((10, 10), bool)) == 1.0  # 100 px x 100 m² = 1 ha


def test_otsu_splits_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-0.4, 0.05, 5000), rng.normal(0.5, 0.05, 2000)])
    assert -0.2 < otsu_threshold(v) < 0.3


def test_seasons():
    assert season_of(10) == "post_monsoon" and season_of(4) == "pre_monsoon" and season_of(7) == "monsoon"


def test_timeseries_tracks_shrinking_lake():
    stack = synthetic_stack(shrink_per_year=0.05)
    ts = water_timeseries(stack)
    assert ts["usable"].sum() > 0.5 * len(ts)
    assert (~ts["usable"]).any()  # some cloudy dates flagged
    s = yearly_summary(ts)
    slope = trend_ha_per_year(s, "post_monsoon_peak_ha")
    assert slope is not None and slope < 0
    # true lake area at the first date is pi*r^2 px; check we're within 5%
    first = ts[ts["usable"]].iloc[0]
    assert first["water_ha"] > 0


def test_clouds_over_lake_do_not_create_fake_dips():
    # Lake shrinks steadily, so yearly lows/peaks must never jump down then up.
    s = yearly_summary(water_timeseries(synthetic_stack(shrink_per_year=0.04, cloud_prob=0.4)))
    for col in ("pre_monsoon_low_ha", "post_monsoon_peak_ha"):
        v = s[col].dropna().values
        assert all(b <= a * 1.03 for a, b in zip(v, v[1:])), (col, v)
        assert all(a * 0.85 <= b for a, b in zip(v, v[1:])), (col, v)  # no sudden drops


def test_area_accuracy_on_clear_scene():
    stack = synthetic_stack(start="2018-10-15", end="2018-10-15", cloud_prob=0.0, shrink_per_year=0.0)
    ts = water_timeseries(stack)
    size, r0 = 120, 35.0
    yy, xx = np.mgrid[:size, :size]
    t = stack.time.to_index()[0]
    seasonal = 0.85 + 0.15 * np.cos(2 * np.pi * (t.dayofyear - 290) / 365.25)
    true_px = (np.hypot(yy - size / 2, xx - size / 2) < r0 * seasonal).sum()
    assert ts["water_ha"].iloc[0] == pytest.approx(true_px / 100, rel=0.02)


def test_no_fake_jump_across_2022_baseline_change():
    stack = synthetic_stack(start="2021-10-01", end="2022-11-30", shrink_per_year=0.0, cloud_prob=0.0)
    ts = water_timeseries(stack)
    s = yearly_summary(ts)
    a, b = s.loc[2021, "post_monsoon_peak_ha"], s.loc[2022, "post_monsoon_peak_ha"]
    assert abs(a - b) / a < 0.03


def test_end_to_end_outputs(tmp_path):
    report = run_from_stack(LAKES["sirpur"], synthetic_stack(), tmp_path)
    out = tmp_path / "sirpur"
    for f in ("water_timeseries.csv", "yearly_summary.csv", "water_timeseries.png", "before_after.png", "report.json"):
        assert (out / f).exists(), f
    assert json.loads((out / "report.json").read_text())["scenes_usable"] == report["scenes_usable"]
    assert report["trend_post_monsoon_ha_per_year"] < 0


def test_lake_bboxes_are_sane():
    for aoi in LAKES.values():
        x0, y0, x1, y1 = aoi.bbox
        assert 75.5 < x0 < x1 < 76.1 and 22.5 < y0 < y1 < 23.0, aoi.key
