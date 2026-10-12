import numpy as np
import pandas as pd
import pytest

from elevenwinters import qc
from elevenwinters.analysis.common import festival_mask, in_window
from elevenwinters.fires import _points_in_geometry
from elevenwinters.geo import bearing_deg, haversine_km, uv_to_direction, wind_to_uv
from elevenwinters.ingest.openaq import site_key


def test_haversine_delhi_ludhiana():
    # Delhi to Ludhiana is roughly 270-280 km
    d = haversine_km(28.6139, 77.2090, 30.9010, 75.8573)
    assert 265 < d < 290


def test_bearing_to_northwest():
    b = bearing_deg(28.6139, 77.2090, 30.9010, 75.8573)
    assert 300 < b < 340


@pytest.mark.parametrize("direction", [0, 45, 135, 225, 315])
def test_wind_roundtrip(direction):
    u, v = wind_to_uv(5.0, direction)
    assert uv_to_direction(u, v) == pytest.approx(direction, abs=1e-6)


def test_northwest_wind_has_positive_u():
    # wind FROM the north-west blows towards the south-east: u > 0, v < 0
    u, v = wind_to_uv(5.0, 315)
    assert u > 0 and v < 0


def _hourly(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"pm25": values, "pm10": [v * 2 if v == v else v for v in values]}, index=idx)


def test_flatline_flagged():
    vals = [50, 52, 60, 60, 60, 60, 60, 60, 61, 58] + [55] * 2
    cleaned, rep = qc.clean(_hourly(vals))
    assert rep["pm25"]["flatline"] == 6
    assert cleaned["pm25"].isna().sum() >= 6


def test_isolated_spike_flagged_but_sustained_event_kept():
    spike = [50, 52, 900, 55, 53, 51]
    cleaned, rep = qc.clean(_hourly(spike))
    assert rep["pm25"]["spike"] == 1
    diwali_like = [80, 150, 400, 700, 900, 650, 400, 200]
    cleaned, rep = qc.clean(_hourly(diwali_like))
    assert rep["pm25"]["spike"] == 0
    assert cleaned["pm25"].notna().all()


def test_stall_before_gap_keeps_first_reading():
    df = pd.DataFrame({"pm25": [300, 652, 652, 652, np.nan, 500]},
                      index=pd.date_range("2020-11-14 20:00", periods=6, freq="h"))
    cleaned, rep = qc.clean(df)
    assert rep["pm25"]["stall"] == 2
    assert cleaned["pm25"].iloc[1] == 652 and cleaned["pm25"].iloc[2:4].isna().all()
    # an ordinary repeat that is not followed by a gap is left alone
    df2 = pd.DataFrame({"pm25": [300, 652, 652, 640]}, index=pd.date_range("2020-11-14", periods=4, freq="h"))
    assert qc.clean(df2)[1]["pm25"]["stall"] == 0


def test_pm25_above_pm10_removed():
    df = _hourly([50, 60, 70])
    df.loc[df.index[1], "pm10"] = 40
    cleaned, rep = qc.clean(df)
    assert rep["pm25"]["pm_ratio"] == 1
    assert np.isnan(cleaned["pm25"].iloc[1])


def test_hour_ending_to_hour_start():
    raw = pd.DataFrame({
        "sensors_id": [1, 1],
        "datetime": pd.to_datetime(["2025-11-01 01:00", "2025-11-01 02:00"]),
        "parameter": ["pm25", "pm25"], "units": ["µg/m³", "µg/m³"], "value": [116.0, 126.0],
    })
    h = qc.to_hourly(raw)
    assert h.index[0] == pd.Timestamp("2025-11-01 00:00")
    assert h["pm25"].iloc[0] == 116.0


def test_co_units_to_mg():
    raw = pd.DataFrame({
        "sensors_id": [1], "datetime": pd.to_datetime(["2025-01-01 01:00"]),
        "parameter": ["co"], "units": ["µg/m³"], "value": [1500.0],
    })
    assert qc.to_hourly(raw)["co"].iloc[0] == pytest.approx(1.5)


def test_site_key_merges_feeds():
    a = site_key("R K Puram, Delhi - DPCC", "cpcb", "Delhi")
    b = site_key("R K Puram, New Delhi - DPCC", "cpcb", "Delhi")
    c = site_key("Pusa, Delhi - IMD", "cpcb", "Delhi")
    d = site_key("Pusa, Delhi - DPCC", "cpcb", "Delhi")
    assert a == b and c != d
    assert site_key("New Delhi", "us_mission", "Delhi") == "us_delhi"


def test_point_in_polygon_with_hole():
    square = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                                                 [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]]}
    lon = np.array([1.0, 5.0, 11.0])
    lat = np.array([1.0, 5.0, 5.0])
    assert _points_in_geometry(lon, lat, square).tolist() == [True, False, False]


def test_festival_mask_and_window():
    idx = pd.date_range("2024-10-27", "2024-11-05", freq="D")
    m = festival_mask(idx, before=3, after=3)  # Diwali 2024 = 31 Oct
    assert m.tolist() == [False, True, True, True, True, True, True, True, False, False]
    assert in_window(pd.DatetimeIndex(["2024-10-09", "2024-10-10", "2024-11-30", "2024-12-01"]),
                     "10-10", "11-30").tolist() == [False, True, True, False]
