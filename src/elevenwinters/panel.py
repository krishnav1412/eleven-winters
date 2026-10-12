"""Build the analysis tables from raw downloads.

Outputs (processed/):
- site_hourly.parquet   one row per site-hour, cleaned pollutants. A site merges the
                        OpenAQ locations that are the same physical monitor.
- city_hourly.parquet   median across CPCB-network sites per city-hour (+ site count)
- meteo_hourly.parquet  ERA5 weather per city-hour
- qc_report.csv         flag counts per site and pollutant (also copied to results/)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from elevenwinters import config, qc
from elevenwinters.ingest.meteo import slug
from elevenwinters.ingest.openaq import load_stations

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co"]


def _read_location(location_id: int) -> pd.DataFrame | None:
    folder = config.raw_dir("openaq", f"location_id={location_id}")
    files = sorted(folder.glob("year=*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def site_table() -> pd.DataFrame:
    st = load_stations()
    return (
        st.groupby("site")
        .agg(city=("city", "first"), state=("state", "first"), region=("region", "first"),
             network=("network", "first"), name=("name", "first"),
             lat=("lat", "mean"), lon=("lon", "mean"), n_locations=("location_id", "size"))
        .reset_index()
    )


def build_site_hourly() -> pd.DataFrame:
    stations = load_stations()
    start = pd.Timestamp(config.settings()["period"]["start"])
    end = pd.Timestamp(config.settings()["period"]["end"]) + pd.Timedelta(hours=23)
    frames, reports = [], []
    for site, members in stations.groupby("site"):
        raws = [r for r in (_read_location(i) for i in members["location_id"]) if r is not None]
        if not raws:
            continue
        hourly = qc.to_hourly(pd.concat(raws, ignore_index=True))
        hourly = hourly.loc[(hourly.index >= start) & (hourly.index <= end)]
        if hourly.empty:
            continue
        cleaned, rep = qc.clean(hourly)
        first = members.iloc[0]
        for p, counts in rep.items():
            reports.append({"site": site, "city": first.city, "network": first.network,
                            "parameter": p, **counts})
        cleaned = cleaned.reindex(columns=POLLUTANTS).dropna(how="all").reset_index()
        cleaned["site"] = site
        frames.append(cleaned)
    df = pd.concat(frames, ignore_index=True)
    meta = site_table()[["site", "city", "region", "network"]]
    df = df.merge(meta, on="site", how="left")
    for c in POLLUTANTS:
        df[c] = df[c].astype("float32")
    df.to_parquet(config.processed_dir() / "site_hourly.parquet", index=False)

    rep = pd.DataFrame(reports).fillna(0)
    flag_cols = [c for c in ["invalid", "flatline", "stall", "spike", "pm_ratio"] if c in rep]
    rep["flagged_pct"] = (100 * rep[flag_cols].sum(axis=1) / rep["n"].replace(0, np.nan)).round(2)
    rep.to_csv(config.processed_dir() / "qc_report.csv", index=False)
    rep.to_csv(config.results_dir() / "qc_report.csv", index=False)
    site_table().to_csv(config.results_dir() / "sites.csv", index=False)
    return df


def build_city_hourly(site_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    sh = site_hourly if site_hourly is not None else load_site_hourly()
    sh = sh[sh["network"] == "cpcb"]
    agg = sh.groupby(["city", "datetime"])[POLLUTANTS].median()
    n = sh.dropna(subset=["pm25"]).groupby(["city", "datetime"]).size().rename("n_sites")
    city = agg.join(n).reset_index()
    regions = {c["name"]: c["region"] for c in config.cities()}
    city["region"] = city["city"].map(regions)
    city.to_parquet(config.processed_dir() / "city_hourly.parquet", index=False)
    return city


def build_meteo_hourly() -> pd.DataFrame:
    frames = []
    for c in config.cities():
        source = c.get("met_proxy", c["name"])   # neighbouring city's weather where configured
        path = config.raw_dir("meteo") / f"{slug(source)}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path).assign(city=c["name"]))
    met = pd.concat(frames, ignore_index=True)
    met.to_parquet(config.processed_dir() / "meteo_hourly.parquet", index=False)
    return met


def load_site_hourly() -> pd.DataFrame:
    return pd.read_parquet(config.processed_dir() / "site_hourly.parquet")


def load_meteo_hourly() -> pd.DataFrame:
    return pd.read_parquet(config.processed_dir() / "meteo_hourly.parquet")


def build_all() -> None:
    sh = build_site_hourly()
    print(f"  site_hourly: {len(sh):,} rows, {sh['site'].nunique()} sites")
    ch = build_city_hourly(sh)
    print(f"  city_hourly: {len(ch):,} rows, {ch['city'].nunique()} cities")
    met = build_meteo_hourly()
    print(f"  meteo_hourly: {len(met):,} rows, {met['city'].nunique()} cities")
