"""Part A: how much of the winter problem is weather? (H1) Did the 2020 drop last? (H6)

Per site, a daily LightGBM model of PM2.5 on weather and time is used to build a
weather-normalised series. Skill is reported from 5-fold cross-validation grouped
by calendar year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

from elevenwinters import config, models
from elevenwinters.analysis.common import US_DELHI, daily_mean, save_json
from elevenwinters.panel import load_meteo_hourly, load_site_hourly, site_table

MIN_DAYS = 500
WINTER = [11, 12, 1]
MONSOON = [7, 8, 9]


def deweather_sites(n_samples: int | None = None) -> pd.DataFrame:
    n_samples = n_samples or config.settings()["deweather"]["n_samples"]
    sh = load_site_hourly()
    met = load_meteo_hourly()
    sites = site_table().set_index("site")
    met_by_city = {c: models.met_daily(g) for c, g in met.groupby("city")}
    rows, skill = [], []
    for site, g in sh.groupby("site"):
        city = sites.loc[site, "city"]
        if city not in met_by_city:
            continue
        pm = daily_mean(g.set_index("datetime")["pm25"].sort_index())
        pm = pm.dropna()
        if len(pm) < MIN_DAYS:
            continue
        md = met_by_city[city]
        X = md.join(models.time_features(md.index, hourly=False)).dropna()
        X = X.loc[X.index.intersection(pm.index)]
        y = pm.loc[X.index]
        years = X.index.year.to_numpy()
        oof = models.grouped_oof(X, y, years, np.ones(len(X), bool), n_folds=5, n_estimators=400)
        model, smear = models.fit(X, y, n_estimators=400)
        norm = models.normalise(model, smear, X, models.MET_DAILY, n_samples=n_samples)
        rows.append(pd.DataFrame({"site": site, "date": X.index, "observed": y.values,
                                  "predicted_cv": oof, "normalised": norm}))
        skill.append({"site": site, "city": city, "n_days": len(y),
                      "r2_cv": round(models.r2(np.log(y), np.log(oof)), 3),
                      "r2_cv_linear": round(models.r2(y, oof), 3)})
        print(f"  {site:<32} {city:<16} days={len(y):5d}  CV R2(log)={skill[-1]['r2_cv']:.2f}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(config.processed_dir() / "deweathered_daily.parquet", index=False)
    pd.DataFrame(skill).to_csv(config.results_dir() / "model_skill_daily.csv", index=False)
    return out


def seasonal_gap(dw: pd.DataFrame | None = None) -> pd.DataFrame:
    """H1: winter (Nov-Jan) minus monsoon (Jul-Sep) means, observed vs normalised.
    weather_share = fraction of the observed gap that disappears once weather is removed."""
    dw = dw if dw is not None else pd.read_parquet(config.processed_dir() / "deweathered_daily.parquet")
    sites = site_table().set_index("site")
    m = dw["date"].dt.month
    rows = []
    for site, g in dw.assign(m=m).groupby("site"):
        w, s = g[g["m"].isin(WINTER)], g[g["m"].isin(MONSOON)]
        if len(w) < 60 or len(s) < 60:
            continue
        obs_gap = w["observed"].mean() - s["observed"].mean()
        norm_gap = w["normalised"].mean() - s["normalised"].mean()
        rows.append({
            "site": site, "city": sites.loc[site, "city"], "region": sites.loc[site, "region"],
            "network": sites.loc[site, "network"],
            "winter_obs": w["observed"].mean(), "monsoon_obs": s["observed"].mean(),
            "winter_norm": w["normalised"].mean(), "monsoon_norm": s["normalised"].mean(),
            "obs_gap": obs_gap, "norm_gap": norm_gap,
            "weather_share": 1 - norm_gap / obs_gap if obs_gap > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.round(3).to_csv(config.results_dir() / "seasonal_gap_by_site.csv", index=False)
    city = (out.groupby(["city", "region"])[["winter_obs", "monsoon_obs", "winter_norm",
                                             "monsoon_norm", "obs_gap", "norm_gap", "weather_share"]]
            .median().reset_index())
    city.round(3).to_csv(config.results_dir() / "seasonal_gap_by_city.csv", index=False)
    north = city["region"].isin(["delhi_ncr", "source", "igp"])
    summary = {
        "n_sites": int(len(out)), "n_cities": int(len(city)),
        "weather_share_median_north": float(city.loc[north, "weather_share"].median()),
        "weather_share_median_rest": float(city.loc[~north, "weather_share"].median()),
        "winter_obs_north": float(city.loc[north, "winter_obs"].median()),
        "winter_obs_rest": float(city.loc[~north, "winter_obs"].median()),
        "winter_norm_north": float(city.loc[north, "winter_norm"].median()),
        "winter_norm_rest": float(city.loc[~north, "winter_norm"].median()),
    }
    summary["ns_gap_obs"] = summary["winter_obs_north"] - summary["winter_obs_rest"]
    summary["ns_gap_norm"] = summary["winter_norm_north"] - summary["winter_norm_rest"]
    summary["ns_gap_weather_share"] = 1 - summary["ns_gap_norm"] / summary["ns_gap_obs"]
    save_json(summary, "parta_seasonal_summary.json")
    return out


def trend_delhi(dw: pd.DataFrame | None = None, site: str = US_DELHI) -> dict:
    """H6: Theil-Sen trend, Mann-Kendall test and change points on the weather-normalised
    monthly series of the Delhi US Embassy monitor (the only gap-free Delhi record)."""
    dw = dw if dw is not None else pd.read_parquet(config.processed_dir() / "deweathered_daily.parquet")
    g = dw[dw["site"] == site].set_index("date").sort_index()
    monthly = g[["observed", "normalised"]].resample("MS").mean()
    counts = g["observed"].resample("MS").count()
    monthly = monthly[counts >= 15]
    clim = monthly["normalised"].groupby(monthly.index.month).transform("mean")
    anom = monthly["normalised"] - clim
    t_years = (monthly.index - monthly.index[0]).days / 365.25
    slope, intercept, lo, hi = stats.theilslopes(anom.values, t_years)
    tau, p = stats.kendalltau(t_years, anom.values)
    algo = rpt.Pelt(model="l2", min_size=6).fit(anom.values.reshape(-1, 1))
    pen = 2 * np.log(len(anom)) * np.var(anom.values)
    bkps = [monthly.index[i].strftime("%Y-%m") for i in algo.predict(pen=pen)[:-1]]
    annual = g["normalised"].resample("YS").mean()
    ann_n = g["normalised"].resample("YS").count()
    annual = annual[ann_n >= 300]
    out = {
        "site": site,
        "months": int(len(monthly)),
        "theil_sen_slope_per_year": slope, "slope_ci95": [lo, hi],
        "mean_level": float(monthly["normalised"].mean()),
        "slope_pct_per_year": 100 * slope / float(monthly["normalised"].mean()),
        "kendall_tau": tau, "kendall_p": p,
        "change_points": bkps,
        "annual_normalised": {str(k.year): round(float(v), 1) for k, v in annual.items()},
    }
    if "2019" in out["annual_normalised"]:
        post = [v for k, v in out["annual_normalised"].items() if 2021 <= int(k) <= 2025]
        out["mean_2021_2025_vs_2019_pct"] = 100 * (np.mean(post) / out["annual_normalised"]["2019"] - 1)
    monthly.assign(anomaly=anom).to_csv(config.results_dir() / "delhi_monthly_normalised.csv")
    save_json(out, "parta_trend_delhi.json")
    return out


def run() -> None:
    dw = deweather_sites()
    gap = seasonal_gap(dw)
    print(f"  seasonal gap computed for {len(gap)} sites")
    tr = trend_delhi(dw)
    print(f"  Delhi trend: {tr['slope_pct_per_year']:.1f}%/yr (p={tr['kendall_p']:.3f}), "
          f"change points {tr['change_points']}")
