"""OpenAQ: station index (API, needs a free key) and hourly data (public S3 archive, no key).

The archive stores one gzipped CSV per location per day:
    records/csv.gz/locationid={id}/year={yyyy}/month={mm}/location-{id}-{yyyymmdd}.csv.gz
We download a station-year at a time and keep it as one Parquet file, so the
data folder never holds hundreds of thousands of tiny files.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from elevenwinters import config
from elevenwinters.geo import haversine_km

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _session(pool: int = 64) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=3)
    s.mount("https://", adapter)
    return s


# --------------------------------------------------------------------------- station index


def fetch_locations(refresh: bool = False) -> pd.DataFrame:
    """All OpenAQ locations in India, flattened to one row per location."""
    out = config.raw_dir("openaq") / "locations_IN.json"
    if out.exists() and not refresh:
        records = json.loads(out.read_text(encoding="utf-8"))
    else:
        cfg = config.settings()["openaq"]
        s = _session(4)
        s.headers.update({"X-API-Key": config.openaq_key(), "Accept": "application/json"})
        records, page = [], 1
        while True:
            r = s.get(
                f"{cfg['api']}/locations",
                params={"countries_id": cfg["country_id"], "limit": 1000, "page": page},
                timeout=120,
            )
            r.raise_for_status()
            batch = r.json()["results"]
            records.extend(batch)
            if len(batch) < 1000:
                break
            page += 1
            time.sleep(1.2)  # stay well under 60 requests/minute
        out.write_text(json.dumps(records), encoding="utf-8")

    rows = []
    for rec in records:
        params = sorted({s["parameter"]["name"] for s in rec.get("sensors", [])})
        rows.append(
            {
                "location_id": rec["id"],
                "name": rec.get("name"),
                "locality": rec.get("locality"),
                "provider": (rec.get("provider") or {}).get("name"),
                "owner": (rec.get("owner") or {}).get("name"),
                "lat": (rec.get("coordinates") or {}).get("latitude"),
                "lon": (rec.get("coordinates") or {}).get("longitude"),
                "is_mobile": rec.get("isMobile"),
                "is_monitor": rec.get("isMonitor"),
                "params": ",".join(params),
            }
        )
    return pd.DataFrame(rows)


def archive_years(location_id: int, session: requests.Session | None = None) -> list[int]:
    """Calendar years with at least one archive file for this location."""
    s = session or _session(4)
    base = config.settings()["openaq"]["archive"]
    r = s.get(
        f"{base}/",
        params={
            "list-type": "2",
            "prefix": f"records/csv.gz/locationid={location_id}/",
            "delimiter": "/",
        },
        timeout=60,
    )
    r.raise_for_status()
    return sorted(int(y) for y in re.findall(r"year=(\d{4})/", r.text))


US_MISSION = r"AirNow|StateAir"
AGENCIES = ("dpcc", "cpcb", "imd", "iitm", "hspcb", "uppcb", "ppcb", "pspcb", "mpcb", "kspcb",
            "tnpcb", "tspcb", "rspcb", "bspcb", "wbpcb", "gpcb", "appcb", "kspcb", "spcb", "crrcb")


def site_key(name: str, network: str, city: str) -> str:
    """One physical site can appear as several OpenAQ locations (old 'caaqm' feed vs new
    'CPCB' feed, renamed sites). Key = place name before the first comma + agency suffix."""
    if network == "us_mission":
        return f"us_{city.lower()}"
    low = str(name).lower()
    place = re.sub(r"[^a-z0-9]", "", low.split(",")[0])
    agency = next((a for a in AGENCIES if re.search(rf"\b{a}\b", low)), "")
    return f"{place}_{agency}" if agency else place


def select_stations(min_years: int | None = None) -> pd.DataFrame:
    """Match reference monitors to study cities, merge duplicate locations into sites,
    and keep the sites with the longest records.

    Writes config/stations.csv (one row per OpenAQ location, with its site id), which is
    committed so the station list is fixed.
    """
    cfg = config.settings()["openaq"]
    min_years = min_years or cfg["min_years"]
    locs = fetch_locations()
    locs = locs[(locs["is_monitor"] == True) & (locs["is_mobile"] != True)]  # noqa: E712
    locs = locs[locs["params"].str.contains(r"\bpm25\b", regex=True)]
    locs = locs.dropna(subset=["lat", "lon"])

    s = _session(16)
    chosen = []
    for city in config.cities():
        d = haversine_km(city["lat"], city["lon"], locs["lat"].values, locs["lon"].values)
        cand = locs.assign(dist_km=d)
        cand = cand[cand["dist_km"] <= city["radius_km"]].copy()
        if cand.empty:
            print(f"  {city['name']}: no monitors within {city['radius_km']} km")
            continue
        with ThreadPoolExecutor(8) as ex:
            years = list(ex.map(lambda i: archive_years(i, s), cand["location_id"]))
        cand["years"] = years
        cand = cand[cand["years"].map(len) > 0].copy()
        cand["network"] = np.where(
            cand["provider"].fillna("").str.contains(US_MISSION, case=False), "us_mission", "cpcb"
        )
        cand["site"] = [site_key(n, w, city["name"]) for n, w in zip(cand["name"], cand["network"])]
        sites = (
            cand.groupby("site")
            .agg(years=("years", lambda ys: sorted(set().union(*map(set, ys)))),
                 network=("network", "first"), dist_km=("dist_km", "min"))
            .assign(n_years=lambda t: t["years"].map(len), first_year=lambda t: t["years"].map(min))
        )
        us_sites = sites[sites["network"] == "us_mission"]
        cp_sites = sites[(sites["network"] == "cpcb") & (sites["n_years"] >= min_years)]
        cp_sites = cp_sites.sort_values(["n_years", "first_year", "dist_km"],
                                        ascending=[False, True, True]).head(city["max_stations"])
        keep = set(cp_sites.index) | set(us_sites.index)
        picked = cand[cand["site"].isin(keep)]
        for _, row in picked.iterrows():
            chosen.append(
                {
                    "location_id": int(row["location_id"]),
                    "site": row["site"],
                    "name": row["name"],
                    "city": city["name"],
                    "state": city["state"],
                    "region": city["region"],
                    "network": row["network"],
                    "provider": row["provider"],
                    "lat": round(row["lat"], 5),
                    "lon": round(row["lon"], 5),
                    "dist_km": round(row["dist_km"], 1),
                    "first_year": int(min(row["years"])),
                    "last_year": int(max(row["years"])),
                    "years": " ".join(map(str, row["years"])),
                    "params": row["params"],
                }
            )
        print(f"  {city['name']}: {len(cp_sites)} CPCB-network sites + {len(us_sites)} US-mission "
              f"({len(picked)} locations, {len(cand)} nearby)")
    out = pd.DataFrame(chosen).drop_duplicates("location_id")
    out.to_csv(config.CONFIG_DIR / "stations.csv", index=False)
    return out


def load_stations() -> pd.DataFrame:
    return pd.read_csv(config.CONFIG_DIR / "stations.csv")


# --------------------------------------------------------------------------- archive download


def _list_keys(session: requests.Session, prefix: str) -> list[str]:
    base = config.settings()["openaq"]["archive"]
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(f"{base}/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", S3_NS)]
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        else:
            return keys


def _fetch_day(session: requests.Session, key: str, wanted: set[str]) -> pd.DataFrame | None:
    base = config.settings()["openaq"]["archive"]
    for attempt in range(4):
        try:
            r = session.get(f"{base}/{key}", timeout=60)
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            df = pd.read_csv(io.BytesIO(raw), usecols=["sensors_id", "datetime", "parameter", "units", "value"])
            return df[df["parameter"].isin(wanted)]
        except Exception:  # network hiccup or truncated file: retry, then give up on this day
            time.sleep(1.5 * (attempt + 1))
    return None


def download_station(location_id: int, years: list[int], max_workers: int | None = None,
                     session: requests.Session | None = None) -> dict:
    """Download every daily file for the given years; one Parquet per station-year."""
    cfg = config.settings()
    wanted = set(cfg["pollutants"])
    max_workers = max_workers or cfg["openaq"]["max_workers"]
    s = session or _session(max_workers)
    out_dir = config.raw_dir("openaq", f"location_id={location_id}")
    report = {}
    for year in years:
        target = out_dir / f"year={year}.parquet"
        if target.exists():
            report[year] = "cached"
            continue
        keys = _list_keys(s, f"records/csv.gz/locationid={location_id}/year={year}/")
        if not keys:
            report[year] = "empty"
            continue
        frames, failed = [], 0
        with ThreadPoolExecutor(max_workers) as ex:
            futs = [ex.submit(_fetch_day, s, k, wanted) for k in keys]
            for f in as_completed(futs):
                df = f.result()
                if df is None:
                    failed += 1
                elif not df.empty:
                    frames.append(df)
        if not frames:
            report[year] = f"no pollutant rows ({len(keys)} files, {failed} failed)"
            continue
        df = pd.concat(frames, ignore_index=True)
        # Archive timestamps carry the +05:30 offset; store naive IST for readability.
        ts = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.assign(datetime=ts, location_id=location_id)
        df = df.sort_values(["parameter", "datetime"]).reset_index(drop=True)
        df.to_parquet(target, index=False)
        report[year] = f"{len(keys)} files, {len(df):,} rows, {failed} failed"
    return report


def download_all(max_workers: int | None = None) -> None:
    stations = load_stations()
    start = int(config.settings()["period"]["start"][:4])
    end = int(config.settings()["period"]["end"][:4])
    s = _session(max_workers or config.settings()["openaq"]["max_workers"])
    t0 = time.time()
    for i, row in enumerate(stations.itertuples(), 1):
        years = [int(y) for y in str(row.years).split() if start <= int(y) <= end]
        rep = download_station(row.location_id, years, max_workers=max_workers, session=s)
        done = sum(1 for v in rep.values() if v != "empty")
        mins = (time.time() - t0) / 60
        print(f"[{i}/{len(stations)}] {row.city} | {row.name} ({row.location_id}): "
              f"{done} years | {mins:.1f} min elapsed", flush=True)
        for y, v in rep.items():
            if "failed" in v and not v.endswith(" 0 failed"):
                print(f"      {y}: {v}", flush=True)
