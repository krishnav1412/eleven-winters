"""Fire detections in the source region and the wind-aware fire index for Delhi.

Fire index (FI) for receptor Delhi on day d:
    FI_d = sum over VIIRS S-NPP detections on day d of
           upwind(bearing, wind_from_d) * exp(-distance_km / L)
upwind = max(0, cos(bearing from Delhi to the fire - direction the wind blows from)),
so a fire directly upwind counts fully and a fire downwind counts zero. The transport
wind is the daily vector mean of 10 m winds at Delhi, Panipat, Patiala and Ludhiana,
which lie along the Punjab-Delhi corridor.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from elevenwinters import config
from elevenwinters.geo import bearing_deg, haversine_km, uv_to_direction, wind_to_uv

DELHI = (28.6139, 77.2090)
CORRIDOR = ["Delhi", "Panipat", "Patiala", "Ludhiana"]
DECAY_KM = 300.0


# ------------------------------------------------------------------ polygons


def _points_in_ring(lon, lat, ring) -> np.ndarray:
    x, y = np.asarray(ring)[:, 0], np.asarray(ring)[:, 1]
    inside = np.zeros(len(lon), dtype=bool)
    j = len(x) - 1
    for i in range(len(x)):
        cross = ((y[i] > lat) != (y[j] > lat)) & (
            lon < (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i] + 1e-12) + x[i]
        )
        inside ^= cross
        j = i
    return inside


def _points_in_geometry(lon, lat, geom) -> np.ndarray:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    inside = np.zeros(len(lon), dtype=bool)
    for poly in polys:
        shell = _points_in_ring(lon, lat, poly[0])
        for hole in poly[1:]:
            shell &= ~_points_in_ring(lon, lat, hole)
        inside |= shell
    return inside


def assign_state(df: pd.DataFrame) -> pd.Series:
    path = config.raw_dir("boundaries") / "nw_states.geojson"
    gj = json.loads(path.read_text(encoding="utf-8"))
    state = pd.Series("Other", index=df.index, dtype="object")
    lon, lat = df["longitude"].to_numpy(), df["latitude"].to_numpy()
    for feat in gj["features"]:
        mask = _points_in_geometry(lon, lat, feat["geometry"])
        state[mask] = feat["properties"]["name"]
    return state


# ------------------------------------------------------------------ detections


def load_fires() -> pd.DataFrame:
    path = config.interim_dir() / "firms_northwest.parquet"
    fires = pd.read_parquet(path)
    if "state" not in fires.columns:
        fires["state"] = assign_state(fires)
        fires.to_parquet(path, index=False)
    fires["date"] = fires["datetime_ist"].dt.normalize()
    return fires


def persistent_cells(fires: pd.DataFrame, cell_deg: float = 0.02, min_months: int = 4) -> set:
    """Grid cells that light up in many months OUTSIDE the two crop-burning seasons
    (Apr-May wheat, Oct-Nov paddy). These are likely kilns, flares or other fixed heat
    sources, so a robustness check drops them."""
    f = fires[~fires["datetime_ist"].dt.month.isin([4, 5, 10, 11])]
    cell = (f["latitude"] / cell_deg).round().astype(int).astype(str) + "_" + \
           (f["longitude"] / cell_deg).round().astype(int).astype(str)
    months = f["datetime_ist"].dt.to_period("M")
    n = pd.DataFrame({"cell": cell, "m": months}).drop_duplicates().groupby("cell").size()
    return set(n[n >= min_months].index)


def _cell(f: pd.DataFrame, cell_deg: float = 0.02) -> pd.Series:
    return (f["latitude"] / cell_deg).round().astype(int).astype(str) + "_" + \
           (f["longitude"] / cell_deg).round().astype(int).astype(str)


def season_summary(fires: pd.DataFrame | None = None) -> pd.DataFrame:
    """Harvest-season (1 Oct - 30 Nov) detections per state, sensor and overpass.
    `detections_clean` excludes persistent non-crop heat sources."""
    f = fires if fires is not None else load_fires()
    persistent = {p: persistent_cells(g) for p, g in f.groupby("product")}
    f = f[f["state"].isin(["Punjab", "Haryana"]) & f["datetime_ist"].dt.month.isin([10, 11])].copy()
    f["season"] = f["datetime_ist"].dt.year
    f["persistent"] = [c in persistent[p] for c, p in zip(_cell(f), f["product"])]
    out = (
        f.groupby(["product", "season", "state", "daynight"])
        .agg(detections=("frp", "size"), frp_sum=("frp", "sum"),
             detections_clean=("persistent", lambda s: int((~s).sum())))
        .reset_index()
    )
    out.to_csv(config.results_dir() / "fire_season_summary.csv", index=False)
    return out


def corridor_wind() -> pd.DataFrame:
    """Daily vector-mean 10 m wind along the Punjab-Delhi corridor."""
    met = pd.read_parquet(config.processed_dir() / "meteo_hourly.parquet")
    met = met[met["city"].isin(CORRIDOR)].copy()
    u, v = wind_to_uv(met["wind_speed_10m"].to_numpy(), met["wind_direction_10m"].to_numpy())
    met["u"], met["v"] = u, v
    met["date"] = met["datetime"].dt.normalize()
    daily = met.groupby("date")[["u", "v"]].mean()
    daily["wind_from"] = uv_to_direction(daily["u"].to_numpy(), daily["v"].to_numpy())
    daily["wind_speed"] = np.hypot(daily["u"], daily["v"])
    return daily


def fire_index(product: str = "viirs-snpp") -> pd.DataFrame:
    """Daily fire counts and the wind-aware fire index for Delhi, with 0-3 day lags."""
    f = load_fires()
    f = f[(f["product"] == product) & f["state"].isin(["Punjab", "Haryana"])]
    wind = corridor_wind()
    f = f.join(wind[["wind_from"]], on="date", how="left")
    brg = bearing_deg(DELHI[0], DELHI[1], f["latitude"].to_numpy(), f["longitude"].to_numpy())
    dist = haversine_km(DELHI[0], DELHI[1], f["latitude"].to_numpy(), f["longitude"].to_numpy())
    upwind = np.clip(np.cos(np.radians(brg - f["wind_from"].to_numpy())), 0, None)
    f = f.assign(w=np.nan_to_num(upwind) * np.exp(-dist / DECAY_KM))
    daily = f.groupby("date").agg(fires=("w", "size"), fi=("w", "sum"))
    full = pd.date_range(config.settings()["period"]["start"], config.settings()["period"]["end"], freq="D")
    daily = daily.reindex(full, fill_value=0.0)
    daily.index.name = "date"
    for lag in (1, 2, 3):
        daily[f"fi_lag{lag}"] = daily["fi"].shift(lag).fillna(0)
        daily[f"fires_lag{lag}"] = daily["fires"].shift(lag).fillna(0)
    daily.to_parquet(config.processed_dir() / "fire_index_delhi.parquet")
    return daily


def build_all() -> None:
    f = load_fires()
    print(f"  fires: {len(f):,} detections, by state: {f['state'].value_counts().to_dict()}")
    s = season_summary(f)
    print(s[s["product"] == "viirs-snpp"].groupby("season")["detections"].sum().to_string())
    fi = fire_index()
    print(f"  fire index: {len(fi)} days, max {fi['fi'].max():.1f}")
