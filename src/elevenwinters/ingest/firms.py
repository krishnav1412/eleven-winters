"""NASA FIRMS active-fire detections (MODIS and VIIRS S-NPP), India country-year files.

Files are public, no key. We keep the full CSV in raw/ and a north-west India
subset (vegetation fires only) in interim/ for analysis.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from elevenwinters import config


def _download(url: str, target) -> None:
    tmp = target.with_suffix(".part")
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(target)
            return
        except Exception as exc:  # retry on transient errors
            print(f"    retry {attempt + 1} for {url}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def download_all() -> None:
    cfg = config.settings()["firms"]
    raw = config.raw_dir("firms")
    for product in cfg["products"]:
        for year in cfg["years"]:
            name = f"{product}_{year}_India.csv"
            target = raw / name
            if target.exists():
                continue
            url = f"{cfg['base']}/{product}/{year}/{name}"
            t0 = time.time()
            _download(url, target)
            mb = target.stat().st_size / 1e6
            print(f"  {name}: {mb:.0f} MB in {time.time() - t0:.0f} s", flush=True)


def subset_northwest() -> pd.DataFrame:
    """Vegetation-fire detections inside the source-region box, one table for all years."""
    cfg = config.settings()["firms"]
    b = cfg["bbox"]
    frames = []
    for product in cfg["products"]:
        for year in cfg["years"]:
            path = config.raw_dir("firms") / f"{product}_{year}_India.csv"
            if not path.exists():
                print(f"  missing {path.name}")
                continue
            df = pd.read_csv(path, dtype={"acq_time": str, "confidence": str})
            df = df[
                df["latitude"].between(b["lat_min"], b["lat_max"])
                & df["longitude"].between(b["lon_min"], b["lon_max"])
            ]
            if "type" in df.columns:
                df = df[df["type"] == 0]  # 0 = presumed vegetation fire
            hhmm = df["acq_time"].str.zfill(4)
            utc = pd.to_datetime(df["acq_date"] + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True)
            df = df.assign(
                product=product,
                datetime_ist=utc.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None),
            )
            frames.append(
                df[["product", "latitude", "longitude", "datetime_ist", "daynight", "frp",
                    "confidence", "satellite"]]
            )
    fires = pd.concat(frames, ignore_index=True)
    fires["confidence"] = fires["confidence"].astype(str)
    fires.to_parquet(config.interim_dir() / "firms_northwest.parquet", index=False)
    return fires
