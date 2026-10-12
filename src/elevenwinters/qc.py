"""Quality control for station data: units, hourly averaging, and flags.

Conventions
- OpenAQ archive timestamps mark the END of the averaging period. We label each
  hour by its START (01:00 stamp -> the 00:00-01:00 hour), the usual convention
  for diurnal plots.
- Concentrations: PM, NO2, SO2 in ug/m3; CO in mg/m3 (CPCB convention).

Flags (a flagged value is set to NaN and counted in the QC report)
- invalid:    value <= 0 or at/above the instrument ceiling in settings.yaml
- flatline:   the same value repeated for >= N consecutive hours (stuck sensor)
- stall:      a reading of 150+ repeated in the hours just before the record goes missing
              (the monitor holding its last value as it fails); the first reading is kept
- spike:      a single hour far above BOTH neighbours (isolated glitch). Multi-hour
              events such as Diwali nights are deliberately NOT removed by this rule.
- pm_ratio:   PM2.5 greater than PM10 in the same hour (physically impossible)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from elevenwinters import config

# molar-volume conversions at 25 C, 1 atm
PPB_TO_UGM3 = {"no2": 1.88, "so2": 2.62}
CO_PPM_TO_MGM3 = 1.145


def _norm_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every row to the study units; drop rows with units we can't map."""
    u = df["units"].astype(str).str.lower()
    p = df["parameter"]
    val = df["value"].astype(float)
    out = pd.Series(np.nan, index=df.index)
    is_ug = u.str.contains("g/m") & ~u.str.startswith("m")

    pm = p.isin(["pm25", "pm10"])
    out[pm & is_ug] = val[pm & is_ug]

    for gas, factor in PPB_TO_UGM3.items():
        g = p == gas
        out[g & is_ug] = val[g & is_ug]
        out[g & (u == "ppb")] = val[g & (u == "ppb")] * factor
        out[g & (u == "ppm")] = val[g & (u == "ppm")] * factor * 1000

    co = p == "co"
    out[co & is_ug] = val[co & is_ug] / 1000.0
    out[co & u.str.startswith("mg")] = val[co & u.str.startswith("mg")]
    out[co & (u == "ppm")] = val[co & (u == "ppm")] * CO_PPM_TO_MGM3
    out[co & (u == "ppb")] = val[co & (u == "ppb")] * CO_PPM_TO_MGM3 / 1000
    return df.assign(value=out).dropna(subset=["value"])


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Long raw rows -> wide hourly table (hour-start labels), median across sensors."""
    df = _norm_units(raw)
    hour = (df["datetime"] - pd.Timedelta(minutes=1)).dt.floor("h")
    df = df.assign(hour=hour)
    # several sensors of one parameter can share a location; average within sensor, median across
    per_sensor = df.groupby(["parameter", "sensors_id", "hour"], as_index=False)["value"].mean()
    hourly = per_sensor.groupby(["parameter", "hour"])["value"].median().unstack("parameter")
    hourly.index.name = "datetime"
    return hourly.sort_index()


def _flatline_mask(s: pd.Series, n: int) -> pd.Series:
    """True for values inside a run of >= n identical consecutive hours."""
    full = s.asfreq("h")
    same = full.diff().eq(0)
    run_id = (~same).cumsum()
    run_len = full.groupby(run_id).transform("size")
    mask = (run_len >= n) & full.notna()
    return mask.reindex(s.index, fill_value=False)


def _spike_mask(s: pd.Series) -> pd.Series:
    full = s.asfreq("h")
    prev, nxt = full.shift(1), full.shift(-1)
    hi = np.fmax(prev, nxt)
    mask = (full > 4 * hi) & (full - hi > 100) & prev.notna() & nxt.notna()
    return mask.reindex(s.index, fill_value=False)


def _stall_mask(s: pd.Series) -> pd.Series:
    """Monitors that are about to fail often repeat their last reading for an hour or two
    and then go missing. Keep the first reading of such a run, flag the repeats."""
    full = s.asfreq("h")
    vals = full.to_numpy(dtype=float)
    same = np.r_[False, vals[1:] == vals[:-1]]
    run_id = np.cumsum(~same)
    last_idx = pd.Series(np.arange(len(vals))).groupby(run_id).transform("max").to_numpy()
    nxt = np.r_[vals[1:], np.nan]
    mask = same & np.isnan(nxt[last_idx]) & (vals >= 150)
    return pd.Series(mask, index=full.index).reindex(s.index, fill_value=False)


def clean(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply flags. Returns the cleaned table and counts per parameter and flag."""
    cfg = config.settings()["qc"]
    out = hourly.copy()
    report: dict[str, dict[str, int]] = {}
    for p in out.columns:
        s = out[p]
        n = int(s.notna().sum())
        invalid = (s <= 0) | (s >= cfg["max_valid"].get(p, np.inf))
        pm = p in ("pm25", "pm10")
        flat = _flatline_mask(s.where(~invalid), cfg["flatline_hours"]) if pm else invalid & False
        stall = _stall_mask(s.where(~invalid)) if pm else invalid & False
        spike = _spike_mask(s.where(~invalid))
        flags = {"invalid": invalid, "flatline": flat, "stall": stall & ~flat, "spike": spike}
        bad = invalid | flat | stall | spike
        report[p] = {"n": n, **{k: int(v.sum()) for k, v in flags.items()}}
        out[p] = s.where(~bad)
    if {"pm25", "pm10"} <= set(out.columns):
        ratio_bad = out["pm25"] > out["pm10"]
        report["pm25"]["pm_ratio"] = int(ratio_bad.sum())
        out.loc[ratio_bad, ["pm25", "pm10"]] = np.nan
    return out, report
