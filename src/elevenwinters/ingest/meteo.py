"""ERA5-based hourly weather from the Open-Meteo archive API (free, no key).

Open-Meteo counts a request as several calls when it asks for more than 10
variables or more than two weeks. We pace requests to stay under the hourly
budget in settings.yaml, and cache one Parquet file per city.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from elevenwinters import config


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _weight(n_vars: int, n_days: int) -> float:
    return max(1.0, n_vars / 10) * max(1.0, n_days / 14)


def city_start_dates() -> dict[str, str]:
    """Start weather at the first year any selected station in the city reports."""
    from elevenwinters.ingest.openaq import load_stations

    period_start = pd.Timestamp(config.settings()["period"]["start"])
    st = load_stations()
    first = st.groupby("city")["first_year"].min()
    return {
        city: max(period_start, pd.Timestamp(f"{int(y)}-01-01")).strftime("%Y-%m-%d")
        for city, y in first.items()
    }


def fetch_city(city: dict, start: str, end: str) -> pd.DataFrame:
    cfg = config.settings()["meteo"]
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(cfg["hourly"]),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
    }
    for attempt in range(6):
        r = requests.get(cfg["api"], params=params, timeout=300)
        if r.status_code == 429:
            reason = r.text[:200]
            if "Daily" in reason or "daily" in reason:
                raise RuntimeError(f"Open-Meteo daily limit reached: {reason}")
            print(f"    rate limited ({reason}); waiting 10 min", flush=True)
            time.sleep(600)
            continue
        if r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        hourly = r.json()["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        return df.rename(columns={"time": "datetime"})
    raise RuntimeError(f"failed weather download for {city['name']}")


def download_all() -> None:
    cfg = config.settings()
    end = cfg["period"]["end"]
    starts = city_start_dates()
    # Fetch the cities the core analyses need first: the Punjab-Delhi wind corridor and
    # the cities with US mission monitors. Everything else follows by region.
    priority = ["Delhi", "Panipat", "Patiala", "Ludhiana", "Mumbai", "Hyderabad", "Chennai", "Kolkata"]
    order = {"delhi_ncr": 0, "source": 1, "igp": 2, "rest": 3}
    cities = sorted(
        [c for c in config.cities() if c["name"] in starts and not c.get("met_proxy")],
        key=lambda c: (priority.index(c["name"]) if c["name"] in priority else len(priority),
                       order[c["region"]], c["name"]),
    )
    budget = cfg["meteo"]["max_weighted_calls_per_hour"]
    out_dir = config.raw_dir("meteo")
    for city in cities:
        target = out_dir / f"{slug(city['name'])}.parquet"
        if target.exists():
            continue
        start = starts[city["name"]]
        n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        w = _weight(len(cfg["meteo"]["hourly"]), n_days)
        t0 = time.time()
        df = fetch_city(city, start, end)
        df.to_parquet(target, index=False)
        print(f"  {city['name']}: {len(df):,} hours from {start} (weight {w:.0f})", flush=True)
        # pace so the rolling hourly budget is never exceeded
        wait = w / budget * 3600 - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
