"""Weather adjustment with gradient-boosted trees (LightGBM).

Two uses:
1. Weather normalisation (Grange et al., 2018). Fit log(PM2.5) on weather + time,
   then predict many times with the weather columns resampled from the whole record
   and the time columns kept. The average is the PM2.5 you would expect under
   typical weather; what is left reflects emissions.
2. Business-as-usual (BAU) counterfactual for events. Fit on ordinary days only,
   predict out of fold, and read the event's excess as observed minus predicted.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from elevenwinters.geo import wind_to_uv

T0 = pd.Timestamp("2016-01-01")

PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    n_estimators=500,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
)

MET_DAILY = ["t_mean", "t_range", "rh_mean", "precip_sum", "sp_mean", "ws_mean", "u_mean",
             "v_mean", "blh_mean", "blh_min", "blh_max", "cloud_mean", "sw_sum"]
MET_HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
              "wind_speed_10m", "u10", "v10", "boundary_layer_height", "cloud_cover",
              "shortwave_radiation", "ws_24h", "blh_24h", "precip_24h", "blh_6h"]


# ------------------------------------------------------------------ features


def met_hourly(met_city: pd.DataFrame) -> pd.DataFrame:
    m = met_city.set_index("datetime").sort_index()
    m = m[~m.index.duplicated()]
    u, v = wind_to_uv(m["wind_speed_10m"].to_numpy(), m["wind_direction_10m"].to_numpy())
    m["u10"], m["v10"] = u, v
    m["ws_24h"] = m["wind_speed_10m"].rolling(24, min_periods=12).mean()
    m["blh_24h"] = m["boundary_layer_height"].rolling(24, min_periods=12).mean()
    m["blh_6h"] = m["boundary_layer_height"].rolling(6, min_periods=3).mean()
    m["precip_24h"] = m["precipitation"].rolling(24, min_periods=12).sum()
    return m[MET_HOURLY]


def met_daily(met_city: pd.DataFrame) -> pd.DataFrame:
    m = met_city.set_index("datetime").sort_index()
    u, v = wind_to_uv(m["wind_speed_10m"].to_numpy(), m["wind_direction_10m"].to_numpy())
    m = m.assign(u=u, v=v)
    g = m.groupby(m.index.normalize())
    d = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(),
        "t_range": g["temperature_2m"].max() - g["temperature_2m"].min(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "precip_sum": g["precipitation"].sum(),
        "sp_mean": g["surface_pressure"].mean(),
        "ws_mean": g["wind_speed_10m"].mean(),
        "u_mean": g["u"].mean(),
        "v_mean": g["v"].mean(),
        "blh_mean": g["boundary_layer_height"].mean(),
        "blh_min": g["boundary_layer_height"].min(),
        "blh_max": g["boundary_layer_height"].max(),
        "cloud_mean": g["cloud_cover"].mean(),
        "sw_sum": g["shortwave_radiation"].sum(),
    })
    d.index.name = "date"
    return d


def time_features(index: pd.DatetimeIndex, hourly: bool, trend: bool = True) -> pd.DataFrame:
    tf = pd.DataFrame(index=index)
    if trend:
        tf["trend"] = (index - T0).days.astype(float)
    tf["doy"] = index.dayofyear
    tf["weekday"] = index.dayofweek
    if hourly:
        tf["hour"] = index.hour
    return tf


# ------------------------------------------------------------------ fitting


def fit(X: pd.DataFrame, y: pd.Series, **overrides) -> tuple[lgb.LGBMRegressor, float]:
    """Fit on log(y). Returns the model and Duan's smearing factor for back-transform."""
    params = {**PARAMS, **overrides}
    model = lgb.LGBMRegressor(**params)
    ly = np.log(y.clip(lower=1.0))
    model.fit(X, ly)
    resid = ly - model.predict(X)
    smear = float(np.mean(np.exp(resid)))
    return model, smear


def predict(model, smear: float, X: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(X)) * smear


def grouped_oof(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, train_mask: np.ndarray,
                n_folds: int = 5, seed: int = 2026, **overrides) -> np.ndarray:
    """Out-of-fold predictions for every row. Rows outside train_mask are never used for
    training; groups (e.g. ISO weeks) are assigned to folds at random so whole blocks of
    time are held out together."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # random fold per group; this is the assignment the locked Diwali 2026 forecast was built with
    fold_of = dict(zip(uniq, rng.integers(0, n_folds, len(uniq))))
    folds = np.array([fold_of[g] for g in groups])
    pred = np.full(len(X), np.nan)
    for k in range(n_folds):
        tr = (folds != k) & train_mask
        te = folds == k
        if te.sum() == 0 or tr.sum() < 50:
            continue
        model, smear = fit(X[tr], y[tr], **overrides)
        pred[te] = predict(model, smear, X[te])
    return pred


def r2(obs, pred) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    ss_res = np.sum((obs[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((obs[ok] - obs[ok].mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def normalise(model, smear: float, X: pd.DataFrame, met_cols: list[str],
              n_samples: int = 200, seed: int = 2026) -> np.ndarray:
    """Weather-normalised prediction: average over weather resampled from the record."""
    rng = np.random.default_rng(seed)
    met_pool = X[met_cols].to_numpy()
    Xs = X.copy()
    total = np.zeros(len(X))
    for _ in range(n_samples):
        Xs[met_cols] = met_pool[rng.integers(0, len(met_pool), len(X))]
        total += predict(model, smear, Xs)
    return total / n_samples
