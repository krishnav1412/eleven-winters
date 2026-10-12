"""Part C: the fire-count paradox.

C1  Season totals of VIIRS S-NPP detections in Punjab and Haryana, day vs night overpass.
C2  Lag: on which day after a fire does Delhi's weather-adjusted PM2.5 respond most?
C3  Coupling (H5): per season, how much does Delhi's weather-adjusted PM2.5 rise per unit of
    the wind-aware fire index? If detections undercount burning after 2021, the slope rises.
C4  Ground test (H4): in Punjab and Haryana, did the evening share of harvest-season PM2.5
    rise after 2021, relative to a post-harvest reference window in the same year?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from elevenwinters import config, models
from elevenwinters.analysis.common import US_DELHI, daily_mean, festival_mask, save_json
from elevenwinters.fires import season_summary
from elevenwinters.panel import load_meteo_hourly, load_site_hourly

SEASON = ("10-10", "11-30")
SEEDS = [2026, 2027, 2028, 2029, 2030]
EVENING = range(18, 24)
MIDDAY = range(10, 16)


def delhi_daily_residual(site: str = US_DELHI) -> pd.DataFrame:
    """Daily log-residual of Delhi PM2.5 from a weather model WITHOUT fire or trend terms.
    Fire smoke and slow change stay in the residual; weather is taken out."""
    sh = load_site_hourly()
    y = daily_mean(sh[sh["site"] == site].set_index("datetime")["pm25"].sort_index()).dropna()
    met = load_meteo_hourly()
    md = models.met_daily(met[met["city"] == "Delhi"])
    X = md.join(models.time_features(md.index, hourly=False, trend=False)).dropna()
    X = X.loc[X.index.intersection(y.index)]
    y = y.loc[X.index]
    train = ~festival_mask(X.index, before=2, after=3)
    weeks = (X.index.isocalendar().year.astype(str) + "-" + X.index.isocalendar().week.astype(str)).to_numpy()
    # repeated cross-fitting: average out-of-fold predictions over five random fold assignments,
    # so results do not hinge on one particular split
    preds = [models.grouped_oof(X, y, weeks, train, n_folds=5, seed=s, n_estimators=400) for s in SEEDS]
    pred = np.exp(np.nanmean(np.log(np.vstack(preds)), axis=0))
    out = pd.DataFrame({"observed": y, "predicted": pred}, index=X.index)
    out["resid_log"] = np.log(out["observed"]) - np.log(out["predicted"])
    out["festival"] = ~train
    return out


def lag_profile(resid: pd.DataFrame, fi: pd.DataFrame) -> pd.DataFrame:
    md = resid.index.strftime("%m-%d")
    r = resid[(md >= SEASON[0]) & (md <= SEASON[1]) & ~resid["festival"]].copy()
    r["season"] = r.index.year
    r["resid_dm"] = r["resid_log"] - r.groupby("season")["resid_log"].transform("mean")
    rows = []
    for lag in range(0, 6):
        for col in ["fires", "fi"]:
            x = fi[col].shift(lag).reindex(r.index)
            x_dm = x - x.groupby(r["season"]).transform("mean")
            rho, p = stats.spearmanr(x_dm, r["resid_dm"], nan_policy="omit")
            rows.append({"lag_days": lag, "predictor": col, "spearman_rho": rho, "p": p, "n": int(x.notna().sum())})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(config.results_dir() / "fire_lag_profile.csv", index=False)
    return out


def coupling_by_season(resid: pd.DataFrame, fi: pd.DataFrame, best_lag: int) -> pd.DataFrame:
    md = resid.index.strftime("%m-%d")
    r = resid[(md >= SEASON[0]) & (md <= SEASON[1]) & ~resid["festival"]].copy()
    # exposure = index on the best lag day and the day either side (smoke lingers)
    lags = sorted({max(0, best_lag - 1), best_lag, best_lag + 1})
    expo = sum(fi["fi"].shift(L) for L in lags) / len(lags)
    r["exposure"] = expo.reindex(r.index)
    r["fires"] = fi["fires"].reindex(r.index)
    rows = []
    for season, g in r.groupby(r.index.year):
        g = g.dropna(subset=["exposure", "resid_log"])
        if len(g) < 25 or g["exposure"].sum() == 0:
            continue
        X = sm.add_constant(g["exposure"] / 100.0)   # per 100 index units
        fit = sm.OLS(g["resid_log"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        b, se = fit.params.iloc[1], fit.bse.iloc[1]
        rows.append({"season": season, "n_days": len(g), "detections": int(g["fires"].sum()),
                     "mean_exposure": g["exposure"].mean(),
                     "slope_log_per_100": b, "se": se,
                     "pct_per_100": 100 * (np.exp(b) - 1),
                     "pct_lo": 100 * (np.exp(b - 1.96 * se) - 1), "pct_hi": 100 * (np.exp(b + 1.96 * se) - 1),
                     "r2": fit.rsquared})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(config.results_dir() / "fire_coupling_by_season.csv", index=False)
    return out


def evening_shift() -> pd.DataFrame:
    """H4 ground test at source-region sites.
    EI = mean PM2.5 18-23h / mean PM2.5 10-15h. R = EI(harvest 15 Oct-30 Nov) / EI(reference 5-31 Dec).
    Using a same-year reference window cancels each site's normal day-night pattern."""
    sh = load_site_hourly()
    sh = sh[(sh["region"] == "source") & (sh["network"] == "cpcb")]
    rows = []
    for site, g in sh.groupby("site"):
        s = g.set_index("datetime")["pm25"].sort_index()
        for year in sorted(s.index.year.unique()):
            h = s[(s.index >= f"{year}-10-15") & (s.index <= f"{year}-11-30 23:00")]
            ref = s[(s.index >= f"{year}-12-05") & (s.index <= f"{year}-12-31 23:00")]

            def ei(x, days):
                ev, md = x[x.index.hour.isin(EVENING)], x[x.index.hour.isin(MIDDAY)]
                need = 0.6 * days * 6   # at least 60% of the possible evening and midday hours
                if ev.count() < need or md.count() < need:
                    return np.nan
                return ev.mean() / md.mean()

            e_h, e_r = ei(h, 47), ei(ref, 27)
            ok = np.isfinite(e_h)
            rows.append({"site": site, "city": g["city"].iloc[0], "season": year,
                         "ei_harvest": e_h, "ei_reference": e_r,
                         "ratio": e_h / e_r if ok and np.isfinite(e_r) else np.nan,
                         "harvest_hours": int(h.count()), "reference_hours": int(ref.count())})
    out = pd.DataFrame(rows).dropna(subset=["ei_harvest"])
    out["era"] = np.where(out["season"] <= 2021, "2016-2021", np.where(out["season"] >= 2022, "2022 on", ""))
    out.round(4).to_csv(config.results_dir() / "evening_shift_by_site.csv", index=False)
    return out


def run() -> None:
    fs = season_summary()
    viirs = fs[fs["product"] == "viirs-snpp"]
    # shares use detections with fixed heat sources removed, as in Fig. 11; totals use all detections
    by_season = viirs.pivot_table(index="season", columns="daynight", values="detections_clean", aggfunc="sum").fillna(0)
    by_season["total"] = by_season.sum(axis=1)
    by_season["night_share"] = by_season.get("N", 0) / by_season["total"]
    punjab = viirs[viirs["state"] == "Punjab"].groupby("season")["detections"].sum()
    fi = pd.read_parquet(config.processed_dir() / "fire_index_delhi.parquet")

    resid = delhi_daily_residual()
    lags = lag_profile(resid, fi)
    best = lags[lags["predictor"] == "fi"].sort_values("spearman_rho", ascending=False).iloc[0]
    coup = coupling_by_season(resid, fi, int(best["lag_days"]))
    ev = evening_shift()

    all_det = viirs.groupby("season")["detections"].sum()
    summary = {
        "viirs_detections_by_season": all_det.astype(int).to_dict(),
        "viirs_clean_detections_by_season": by_season["total"].astype(int).to_dict(),
        "viirs_night_share_by_season": by_season["night_share"].round(4).to_dict(),
        "viirs_punjab_by_season": punjab.astype(int).to_dict(),
        "best_lag_days": int(best["lag_days"]), "best_lag_rho": float(best["spearman_rho"]),
        "coupling": coup.set_index("season")[["pct_per_100", "pct_lo", "pct_hi", "detections"]].round(3).to_dict("index"),
    }
    early = coup[coup["season"] <= 2021]["pct_per_100"]
    late = coup[coup["season"] >= 2022]["pct_per_100"]
    if len(early) and len(late):
        summary["coupling_mean_2016_2021"] = early.mean()
        summary["coupling_mean_2022_on"] = late.mean()
    e1 = ev[ev["era"] == "2016-2021"]["ratio"].dropna()
    e2 = ev[ev["era"] == "2022 on"]["ratio"].dropna()
    if len(e1) >= 3 and len(e2) >= 3:
        u = stats.mannwhitneyu(e2, e1, alternative="greater")
        summary["evening_ratio"] = {"early_median": e1.median(), "late_median": e2.median(),
                                    "n_early": int(len(e1)), "n_late": int(len(e2)),
                                    "late_seasons": sorted(ev[ev["era"] == "2022 on"]["season"].unique().tolist()),
                                    "mannwhitney_p_late_greater": float(u.pvalue)}
        both = ev.dropna(subset=["ratio"]).pivot_table(index="site", columns="era", values="ratio", aggfunc="mean").dropna()
        if len(both) >= 5:
            w = stats.wilcoxon(both["2022 on"], both["2016-2021"], alternative="greater")
            summary["evening_ratio"]["paired_sites"] = int(len(both))
            summary["evening_ratio"]["paired_share_higher"] = float((both["2022 on"] > both["2016-2021"]).mean())
            summary["evening_ratio"]["paired_wilcoxon_p"] = float(w.pvalue)
    save_json(summary, "partc_summary.json")
    print(f"  best lag {summary['best_lag_days']} d; coupling seasons {len(coup)}; evening rows {len(ev)}")
