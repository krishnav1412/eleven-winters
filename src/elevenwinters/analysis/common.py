"""Shared helpers for the analysis parts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from elevenwinters import config

US_DELHI = "us_delhi"


def diwali_dates() -> dict[int, pd.Timestamp]:
    return {d["year"]: pd.Timestamp(d["date"]) for d in config.events()["diwali"]}


def regimes() -> dict[int, str]:
    return {r["year"]: r["regime"] for r in config.events()["cracker_rules_delhi"]}


def daily_mean(hourly: pd.Series, min_hours: int | None = None) -> pd.Series:
    """Daily mean of an hourly series, NaN when fewer than min_hours valid hours."""
    min_hours = min_hours or config.settings()["qc"]["min_station_hours_per_day"]
    g = hourly.groupby(hourly.index.normalize())
    mean, n = g.mean(), g.count()
    out = mean.where(n >= min_hours)
    out.index.name = "date"
    return out


def in_window(dates: pd.DatetimeIndex, start_md: str, end_md: str) -> np.ndarray:
    """True where month-day falls inside [start_md, end_md] (MM-DD strings, same year)."""
    md = dates.strftime("%m-%d")
    return np.asarray((md >= start_md) & (md <= end_md))


def festival_mask(index: pd.DatetimeIndex, before: int = 3, after: int = 3) -> np.ndarray:
    """True for timestamps within [Diwali - before, Diwali + after] days of any year."""
    day = index.normalize()
    mask = np.zeros(len(index), dtype=bool)
    for d in diwali_dates().values():
        mask |= np.asarray((day >= d - pd.Timedelta(days=before)) & (day <= d + pd.Timedelta(days=after)))
    return mask


def bootstrap_ci(values, stat=np.mean, n: int = 5000, alpha: float = 0.05, seed: int = 2026):
    values = np.asarray([v for v in values if np.isfinite(v)])
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, len(values), replace=True)) for _ in range(n)]
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def save_json(obj, name: str) -> None:
    path = config.results_dir() / name

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else round(float(o), 4)
        if isinstance(o, pd.Timestamp):
            return o.strftime("%Y-%m-%d")
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def load_json(name: str):
    return json.loads((config.results_dir() / name).read_text(encoding="utf-8"))
