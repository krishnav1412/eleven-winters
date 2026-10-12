"""Part B: festival vs fields (H2) and the firecracker fingerprint (H3).

For each receptor, an hourly business-as-usual (BAU) model predicts PM2.5 from weather,
time and (for northern cities) the wind-aware fire index. Diwali windows are never used
for training, and every hour is predicted out of fold (ISO weeks held out together), so
Diwali nights and ordinary nights are judged by the same standard.

Censoring. Monitors often stop reporting (or freeze, then stop) during the worst hours of
Diwali night. A mean over the hours that survive is biased low. So we report:
  evening excess  mean(observed - BAU), 18:00-21:59 on Diwali. Complete in every year at
                  the US Embassy; covers the legal 8-10 pm window. PRIMARY measure.
  night excess    18:00 to 02:59, available hours only (biased low when hours are missing)
  night excess LB the same window with missing hours filled by the last valid reading;
                  a rough lower bound because the monitor fails while concentrations rise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from elevenwinters import config, models
from elevenwinters.analysis.common import (US_DELHI, bootstrap_ci, diwali_dates, festival_mask,
                                           regimes, save_json)
from elevenwinters.panel import load_meteo_hourly, load_site_hourly, site_table

EVENING = [(0, h) for h in range(18, 22)]
NIGHT = [(0, h) for h in range(18, 24)] + [(1, h) for h in range(0, 3)]
LATE = [(0, h) for h in range(22, 24)] + [(1, h) for h in range(0, 3)]
NEXT_DAY = [(1, h) for h in range(6, 18)]
US_SITES = ["us_delhi", "us_mumbai", "us_hyderabad", "us_chennai", "us_kolkata"]
CPCB_DELHI = "cpcb_delhi"
NORTH = {"delhi_ncr", "source", "igp"}


def _hours(day: pd.Timestamp, spec) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([day + pd.Timedelta(days=d, hours=h) for d, h in spec])


def receptor_series(receptor: str) -> tuple[pd.Series, str, str]:
    """Hourly PM2.5 for a receptor id, with its city and region."""
    if receptor == CPCB_DELHI:
        ch = pd.read_parquet(config.processed_dir() / "city_hourly.parquet")
        g = ch[(ch["city"] == "Delhi") & (ch["n_sites"] >= 3)].set_index("datetime")["pm25"]
        return g.sort_index(), "Delhi", "delhi_ncr"
    sites = site_table().set_index("site")
    sh = load_site_hourly()
    y = sh[sh["site"] == receptor].set_index("datetime")["pm25"].sort_index()
    return y[~y.index.duplicated()], sites.loc[receptor, "city"], sites.loc[receptor, "region"]


def bau_hourly(receptor: str, with_fire: bool = True) -> pd.DataFrame:
    y, city, region = receptor_series(receptor)
    met = load_meteo_hourly()
    mh = models.met_hourly(met[met["city"] == city])
    X = mh.join(models.time_features(mh.index, hourly=True))
    use_fire = with_fire and region in NORTH
    if use_fire:
        fi = pd.read_parquet(config.processed_dir() / "fire_index_delhi.parquet")
        cols = ["fi", "fi_lag1", "fi_lag2", "fi_lag3"]
        X = X.join(fi[cols].reindex(X.index.normalize()).set_axis(X.index))
    X = X.dropna()
    obs_idx = X.index.intersection(y.dropna().index)
    train = ~festival_mask(obs_idx, before=3, after=3)
    weeks = (obs_idx.isocalendar().year.astype(str) + "-" + obs_idx.isocalendar().week.astype(str)).to_numpy()
    rng_folds = models.grouped_oof(X.loc[obs_idx], y.loc[obs_idx], weeks, train, n_folds=5)
    # BAU for every hour with weather (also hours where the monitor is missing): full-data model
    model, smear = models.fit(X.loc[obs_idx[train]], y.loc[obs_idx[train]])
    bau_all = pd.Series(models.predict(model, smear, X), index=X.index)
    bau_all.loc[obs_idx] = rng_folds  # out-of-fold where we have observations
    out = pd.DataFrame({"observed": y.reindex(X.index), "bau": bau_all})
    out["excess"] = out["observed"] - out["bau"]
    out.attrs.update(receptor=receptor, with_fire=use_fire,
                     r2_log_ordinary=models.r2(np.log(y.loc[obs_idx[train]]), np.log(rng_folds[train])))
    return out


def _window(df: pd.DataFrame, idx: pd.DatetimeIndex, min_hours: int, fill: bool = False) -> dict:
    sub = df.reindex(idx)
    n_obs = int(sub["observed"].notna().sum())
    res = {"n_obs": n_obs, "n_missing": len(idx) - n_obs}
    if n_obs < min_hours:
        return {**res, "obs": np.nan, "bau": np.nan, "excess": np.nan}
    if fill:
        o = sub["observed"].ffill()
        ok = o.notna() & sub["bau"].notna()
        return {**res, "obs": o[ok].mean(), "bau": sub["bau"][ok].mean(), "excess": (o - sub["bau"])[ok].mean()}
    ok = sub["observed"].notna()
    return {**res, "obs": sub["observed"][ok].mean(), "bau": sub["bau"][ok].mean(), "excess": sub["excess"][ok].mean()}


def night_metrics(bau: pd.DataFrame, day: pd.Timestamp) -> dict:
    ev = _window(bau, _hours(day, EVENING), 3)
    ni = _window(bau, _hours(day, NIGHT), 6)
    lb = _window(bau, _hours(day, NIGHT), 3, fill=True)
    late = _window(bau, _hours(day, LATE), 0)
    nd = _window(bau, _hours(day, NEXT_DAY), 8)
    return {"evening_obs": ev["obs"], "evening_bau": ev["bau"], "evening_excess": ev["excess"],
            "night_obs": ni["obs"], "night_bau": ni["bau"], "night_excess": ni["excess"],
            "night_excess_lb": lb["excess"], "late_hours_missing": late["n_missing"],
            "nextday_excess": nd["excess"]}


def diwali_table(bau: pd.DataFrame) -> pd.DataFrame:
    reg = regimes()
    return pd.DataFrame([{"year": y, "date": d, "regime": reg.get(y), **night_metrics(bau, d)}
                         for y, d in diwali_dates().items()])


def placebo_nights(bau: pd.DataFrame) -> pd.DataFrame:
    """Ordinary nights between 1 Oct and 15 Dec, away from festivals (Dussehra about 20 days
    before Diwali; Chhath and Gurpurab up to 16 days after)."""
    rows = []
    for year, d in diwali_dates().items():
        for day in pd.date_range(f"{year}-10-01", f"{year}-12-15", freq="D"):
            if d - pd.Timedelta(days=22) <= day <= d + pd.Timedelta(days=16):
                continue
            m = night_metrics(bau, day)
            if np.isfinite(m["evening_excess"]) or np.isfinite(m["night_excess"]):
                rows.append({"year": year, "date": day, **m})
    return pd.DataFrame(rows)


def event_curves(bau: pd.DataFrame, receptor: str) -> pd.DataFrame:
    rows = []
    for year, d in diwali_dates().items():
        idx = pd.date_range(d - pd.Timedelta(days=2), d + pd.Timedelta(days=3) - pd.Timedelta(hours=1), freq="h")
        sub = bau.reindex(idx)
        rows.append(pd.DataFrame({"site": receptor, "year": year,
                                  "hours_from_diwali_6pm": ((idx - (d + pd.Timedelta(hours=18))) / pd.Timedelta(hours=1)).astype(int),
                                  "datetime": idx, "observed": sub["observed"].values,
                                  "bau": sub["bau"].values, "excess": sub["excess"].values}))
    return pd.concat(rows, ignore_index=True)


def fingerprint(site_hourly: pd.DataFrame) -> pd.DataFrame:
    """H3: Diwali evening-to-midnight vs the same hours on the five reference nights D-7..D-3,
    at CPCB Delhi sites. Ratios and gases do not need a weather model because both windows
    sit in the same week."""
    spec = [(0, h) for h in range(18, 24)]
    sh = site_hourly[(site_hourly["city"] == "Delhi") & (site_hourly["network"] == "cpcb")]
    rows = []
    for site, g in sh.groupby("site"):
        g = g.set_index("datetime").sort_index()
        g = g[~g.index.duplicated()]
        for year, d in diwali_dates().items():
            night = g.reindex(_hours(d, spec))
            ref = pd.concat([g.reindex(_hours(d - pd.Timedelta(days=k), spec)) for k in range(3, 8)])
            if night["pm25"].count() < 4 or ref["pm25"].count() < 15:
                continue
            row = {"site": site, "year": year}
            for p in ["pm25", "pm10", "so2", "no2", "co"]:
                row[f"{p}_diwali"] = night[p].mean() if night[p].count() >= 4 else np.nan
                row[f"{p}_ref"] = ref[p].mean() if ref[p].count() >= 15 else np.nan
            both = night[["pm25", "pm10"]].dropna()
            bref = ref[["pm25", "pm10"]].dropna()
            if len(both) >= 4 and len(bref) >= 15:
                row["ratio_diwali"] = both["pm25"].sum() / both["pm10"].sum()
                row["ratio_ref"] = bref["pm25"].sum() / bref["pm10"].sum()
            rows.append(row)
    fp = pd.DataFrame(rows)
    fp.round(3).to_csv(config.results_dir() / "fingerprint_delhi_sites.csv", index=False)
    tests = {"n_site_years": int(len(fp)), "n_sites": int(fp["site"].nunique()),
             "years": sorted(int(y) for y in fp["year"].unique())}
    for p in ["pm25", "so2", "no2", "co", "ratio"]:
        a, b = f"{p}_diwali", f"{p}_ref"
        if a not in fp:
            continue
        pair = fp[[a, b]].dropna()
        if len(pair) < 5:
            continue
        diff = pair[a] - pair[b]
        w = stats.wilcoxon(pair[a], pair[b])
        tests[p] = {"n": int(len(pair)), "median_diwali": pair[a].median(), "median_ref": pair[b].median(),
                    "median_diff": diff.median(), "median_pct_change": 100 * (pair[a] / pair[b] - 1).median(),
                    "share_higher": float((diff > 0).mean()), "wilcoxon_p": float(w.pvalue)}
    save_json(tests, "partb_fingerprint.json")
    return fp


def _regime_summary(t: pd.DataFrame, col: str) -> dict:
    out = {}
    for reg, g in t.dropna(subset=[col]).groupby("regime"):
        if reg not in ("permissive", "green_window", "full_ban"):
            continue
        lo, hi = bootstrap_ci(g[col].values)
        out[reg] = {"years": [int(y) for y in g["year"]], "mean": g[col].mean(), "ci95": [lo, hi],
                    "values": [round(float(v), 1) for v in g[col]]}
    return out


def run() -> None:
    results = {}
    tables, curves, placebos = [], [], []
    plan = [(US_DELHI, True), (US_DELHI, False), (CPCB_DELHI, True)] + [(s, False) for s in US_SITES[1:]]
    for receptor, wf in plan:
        bau = bau_hourly(receptor, with_fire=wf)
        tag = "fire" if bau.attrs["with_fire"] else "nofire"
        t = diwali_table(bau).assign(site=receptor, model=tag)
        pl = placebo_nights(bau).assign(site=receptor, model=tag)
        for col in ["evening_excess", "night_excess"]:
            ref = pl[col].dropna()
            t[f"{col}_placebo_pct"] = [100 * (ref < v).mean() if np.isfinite(v) else np.nan for v in t[col]]
        tables.append(t)
        placebos.append(pl)
        if wf or receptor not in (US_DELHI,):
            curves.append(event_curves(bau, receptor))
        results[f"r2_{receptor}_{tag}"] = bau.attrs["r2_log_ordinary"]
        print(f"  {receptor} ({tag}): out-of-fold R2(log) on ordinary hours = {bau.attrs['r2_log_ordinary']:.2f}", flush=True)
        if receptor == US_DELHI and tag == "fire":
            bau.to_parquet(config.processed_dir() / "bau_hourly_us_delhi.parquet")
    table = pd.concat(tables, ignore_index=True)
    table.round(2).to_csv(config.results_dir() / "diwali_excess.csv", index=False)
    pd.concat(curves).round(2).to_parquet(config.processed_dir() / "diwali_event_curves.parquet", index=False)
    pl_all = pd.concat(placebos, ignore_index=True)
    pl_all.round(2).to_csv(config.results_dir() / "placebo_nights.csv", index=False)

    def sel(site, model):
        return table[(table["site"] == site) & (table["model"] == model) & (table["year"] <= 2025)]

    def plsel(site, model):
        return pl_all[(pl_all["site"] == site) & (pl_all["model"] == model)]

    # Main Diwali measure: CPCB Delhi network median, whole night (missing hours carried forward)
    net, netpl = sel(CPCB_DELHI, "fire"), plsel(CPCB_DELHI, "fire")
    results["main"] = {
        "receptor": "CPCB Delhi network median (3+ sites)", "measure": "night_excess_lb",
        "by_regime": _regime_summary(net, "night_excess_lb"),
        "n_diwalis": int(net["night_excess_lb"].notna().sum()),
        "years": [int(y) for y in net.dropna(subset=["night_excess_lb"])["year"]],
        "mean": net["night_excess_lb"].mean(), "min": net["night_excess_lb"].min(), "max": net["night_excess_lb"].max(),
        "placebo": {"n_nights": int(netpl["night_excess_lb"].notna().sum()), "mean": netpl["night_excess_lb"].mean(),
                    "sd": netpl["night_excess_lb"].std(), "p95": netpl["night_excess_lb"].quantile(0.95),
                    "p99": netpl["night_excess_lb"].quantile(0.99)},
    }
    results["main"]["n_above_p99"] = int((net["night_excess_lb"] > results["main"]["placebo"]["p99"]).sum())
    # US Embassy: complete evening window (6-10 pm) and the censoring story
    us, uspl = sel(US_DELHI, "fire"), plsel(US_DELHI, "fire")
    results["us_embassy"] = {
        "evening_by_regime": _regime_summary(us, "evening_excess"),
        "night_lb_by_regime": _regime_summary(us, "night_excess_lb"),
        "evening_mean": us["evening_excess"].mean(), "n_evening": int(us["evening_excess"].notna().sum()),
        "placebo_evening": {"mean": uspl["evening_excess"].mean(), "sd": uspl["evening_excess"].std(),
                            "p95": uspl["evening_excess"].quantile(0.95)},
    }
    results["censoring"] = {
        "diwali_late_hours_missing_share": us["late_hours_missing"].sum() / (5 * len(us)),
        "ordinary_late_hours_missing_share": uspl["late_hours_missing"].sum() / (5 * len(uspl)),
        "diwali_nights_with_missing_late_hours": int((us["late_hours_missing"] > 0).sum()),
        "n_diwalis": int(len(us)),
        "network_diwali_late_missing_share": net["late_hours_missing"].sum() / (5 * len(net)),
    }
    # disentanglement: does the Diwali excess still track farm-fire exposure?
    fi = pd.read_parquet(config.processed_dir() / "fire_index_delhi.parquet")
    for site, col in [(US_DELHI, "evening_excess"), (US_DELHI, "night_excess_lb")]:
        for tag in ["fire", "nofire"]:
            g = sel(site, tag).dropna(subset=[col])
            exposure = [fi["fi"].reindex([d - pd.Timedelta(days=1), d]).sum() for d in g["date"]]
            rho, p = stats.spearmanr(exposure, g[col])
            results[f"vs_fire_{site}_{col}_{tag}"] = {"spearman_rho": rho, "p": p, "n": int(len(g))}
    nofire_minus_fire = (sel(US_DELHI, "nofire").set_index("year")["night_excess_lb"]
                         - sel(US_DELHI, "fire").set_index("year")["night_excess_lb"]).dropna()
    results["fire_term_effect_on_diwali_excess"] = {"median_change": nofire_minus_fire.median(),
                                                    "by_year": nofire_minus_fire.round(1).to_dict()}
    # agreement between the two Delhi records where both have a full night
    a = us.set_index("year")["night_excess_lb"]
    b = net.set_index("year")["night_excess_lb"]
    both = pd.concat([a, b], axis=1, keys=["us", "cpcb"]).dropna()
    if len(both) >= 4:
        r, p = stats.pearsonr(both["us"], both["cpcb"])
        results["us_vs_cpcb_night"] = {"n_years": int(len(both)), "pearson_r": r, "p": p,
                                       "years": [int(y) for y in both.index]}
    # regime contrast on the main measure (Kruskal-Wallis, tiny samples: descriptive)
    groups = [g["night_excess_lb"].dropna().values for _, g in net.groupby("regime") if g["night_excess_lb"].notna().sum() >= 2]
    if len(groups) >= 2:
        kw = stats.kruskal(*groups)
        results["main"]["kruskal_p"] = float(kw.pvalue)
    fp = fingerprint(load_site_hourly())
    print(f"  fingerprint: {len(fp)} site-years", flush=True)
    save_json(results, "partb_summary.json")
