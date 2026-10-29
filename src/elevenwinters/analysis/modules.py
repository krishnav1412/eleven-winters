"""Course modules 2-7 (BCSE331L) applied to the Eleven Winters data.

Each function works on the cleaned tables built by `ew panel` and writes its numbers to
results/modules/ and its figures to paper/figures/modules/.

  m2  Data transformation      deduplication, replacing values, binning, missing data (incl. EM / MLE)
  m3  Correlation and time series   univariate, bivariate, multivariate; indexing, resampling, STL
  m4  Summarisation and visualisation   summary measures, contingency table, scatter / dot / bar plots
  m5  Clustering               hierarchical, spectral, MST, Gaussian mixture (EM), outlier detection
  m6  Dimensionality reduction PCA, SVD, factor analysis, intrinsic dimensionality, MDS, Isomap, SOM
  m7  Model development        linear regression, evaluation, accuracy
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from elevenwinters import config, models  # noqa: E402
from elevenwinters.analysis.common import US_DELHI, daily_mean, diwali_dates  # noqa: E402
from elevenwinters.figures.theme import NAQI  # noqa: E402
from elevenwinters.panel import load_meteo_hourly, load_site_hourly  # noqa: E402

OUT = config.results_dir("modules")
FIG = config.ROOT / "paper" / "figures" / "modules"
FIG.mkdir(parents=True, exist_ok=True)
RES: dict = {}
SEED = 2026

INK, MUTED, RULE, ACCENT, EMBER = "#1B1F1D", "#59605C", "#CBD1CB", "#B88A12", "#A9491B"
REGION_COL = {"delhi_ncr": "#1B1F1D", "source": "#A9491B", "igp": "#B88A12", "rest": "#8A918C"}
REGION_NAME = {"delhi_ncr": "Delhi-NCR", "source": "Punjab and Haryana", "igp": "Rest of Gangetic Plain",
               "rest": "Outside the plain"}
NAQI_NAMES = [n for _, _, n in NAQI]
NAQI_BINS = [-0.1] + [hi for _, hi, _ in NAQI]


def _style() -> None:
    for f in (config.ROOT / "assets" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "IBM Plex Sans Condensed", "font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E3E7E2", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left"})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def put(key: str, value) -> None:
    RES[key] = value


# ------------------------------------------------------------------ shared tables


def daily_city() -> pd.DataFrame:
    ch = pd.read_parquet(config.processed_dir() / "city_hourly.parquet")
    ch["date"] = ch["datetime"].dt.normalize()
    g = ch.groupby(["city", "region", "date"])
    d = g[["pm25", "pm10", "no2", "so2", "co"]].mean()
    d["hours"] = g["pm25"].count()
    d = d[d["hours"] >= 18].reset_index()
    return d


def delhi_daily_with_weather() -> pd.DataFrame:
    dc = daily_city()
    d = dc[dc["city"] == "Delhi"].set_index("date")
    met = load_meteo_hourly()
    md = models.met_daily(met[met["city"] == "Delhi"])
    return d.join(md, how="inner")


def us_delhi_daily() -> pd.Series:
    sh = load_site_hourly()
    s = sh[sh["site"] == US_DELHI].set_index("datetime")["pm25"].sort_index()
    return daily_mean(s).dropna()


# ------------------------------------------------------------------ Module 2


def m2() -> None:
    # deduplication: one physical monitor often appears under two archive ids
    st = pd.read_csv(config.CONFIG_DIR / "stations.csv")
    per_site = st.groupby("site").size()
    put("m2_locations", int(len(st)))
    put("m2_sites", int(st["site"].nunique()))
    put("m2_sites_with_duplicates", int((per_site > 1).sum()))
    # stack every feed of every site and look for repeated (site, hour) rows before merging
    from elevenwinters.panel import _read_location
    frames = []
    for site, members in st.groupby("site"):
        for i in members["location_id"]:
            r = _read_location(i)
            if r is not None:
                frames.append(r.loc[r["parameter"] == "pm25", ["datetime", "value"]].assign(site=site))
    raw = pd.concat(frames, ignore_index=True)
    dedup = raw.drop_duplicates(subset=["site", "datetime"])
    put("m2_raw_pm25_rows", int(len(raw)))
    put("m2_duplicate_site_hours", int(len(raw) - len(dedup)))

    # replacing values: monitor error codes and impossible readings become missing
    us_raw = pd.concat([_read_location(i) for i in (2597, 8118)])
    us_raw = us_raw[us_raw["parameter"] == "pm25"]
    put("m2_us_minus999", int((us_raw["value"] <= -999).sum()))
    put("m2_us_ge999", int((us_raw["value"] >= 999).sum()))
    qc = pd.read_csv(config.results_dir() / "qc_report.csv")
    pm = qc[qc["parameter"] == "pm25"]
    put("m2_qc_flags", {c: int(pm[c].sum()) for c in ["invalid", "flatline", "stall", "spike", "pm_ratio"] if c in pm})

    # discretisation and binning
    dc = daily_city()
    dc["naqi"] = pd.cut(dc["pm25"], NAQI_BINS, labels=NAQI_NAMES)
    tab = pd.crosstab(dc["region"].map(REGION_NAME), dc["naqi"], normalize="index").round(3)
    tab.to_csv(OUT / "m2_naqi_by_region.csv")
    put("m2_naqi_by_region", tab.to_dict("index"))
    s = dc["pm25"]
    _, ew_edges = np.histogram(s, bins=5)
    put("m2_equal_width_edges", [round(float(x), 1) for x in ew_edges])
    put("m2_equal_freq_edges", [round(float(x), 1) for x in s.quantile([0, .2, .4, .6, .8, 1])])

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    left = np.zeros(len(tab))
    colors = ["#E6E8E3", "#C9CCC5", "#A3A69F", "#7C7F78", "#4E514B", "#1F211D"]
    for name, c in zip(NAQI_NAMES, colors):
        vals = tab[name].values if name in tab else np.zeros(len(tab))
        ax.barh(tab.index, vals, left=left, color=c, edgecolor="white", linewidth=0.5, label=name)
        left += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Share of days in each National AQI category", pad=24)
    ax.grid(axis="y", visible=False)
    save(fig, "m2_binning")

    # missing data: how much is missing, and how well do imputation methods recover it?
    us = us_delhi_daily()
    full = pd.date_range(us.index.min(), us.index.max(), freq="D")
    put("m2_us_daily_missing_share", round(1 - len(us) / len(full), 3))
    rng = np.random.default_rng(SEED)
    s_full = us.reindex(full)
    obs = s_full.dropna()

    def score(masked: pd.Series, truth: pd.Series, idx) -> dict:
        out = {}
        out["Deletion (mean of the rest)"] = np.full(len(idx), masked.mean())
        out["Mean imputation"] = masked.fillna(masked.mean())[idx].values
        out["Median imputation"] = masked.fillna(masked.median())[idx].values
        out["Last value carried forward"] = masked.ffill().bfill()[idx].values
        out["Linear interpolation"] = masked.interpolate(limit_direction="both")[idx].values
        return {k: float(np.sqrt(np.mean((v - truth[idx].values) ** 2))) for k, v in out.items()}

    idx_mcar = rng.choice(obs.index, size=int(0.1 * len(obs)), replace=False)
    m = s_full.copy()
    m[idx_mcar] = np.nan
    rmse_mcar = score(m, s_full, idx_mcar)
    idx_mnar = obs[obs > obs.quantile(0.9)].index   # the worst days go missing, as on Diwali
    m2_ = s_full.copy()
    m2_[idx_mnar] = np.nan
    rmse_mnar = score(m2_, s_full, idx_mnar)

    # maximum likelihood: EM for a multivariate normal on log PM2.5, PM10, NO2 (Delhi network)
    dd = dc[dc["city"] == "Delhi"].set_index("date")[["pm25", "pm10", "no2"]].dropna()
    X = np.log(dd.values)
    miss = rng.random(len(X)) < 0.15
    truth = X[miss, 0].copy()
    Xm = X.copy()
    Xm[miss, 0] = np.nan
    mu, cov = em_mvn(Xm)
    imp = Xm.copy()
    for i in np.where(miss)[0]:
        o = ~np.isnan(Xm[i])
        imp[i, 0] = mu[0] + cov[0, o] @ np.linalg.solve(cov[np.ix_(o, o)], Xm[i, o] - mu[o])
    em_rmse = float(np.sqrt(np.mean((np.exp(imp[miss, 0]) - np.exp(truth)) ** 2)))
    mean_rmse = float(np.sqrt(np.mean((np.exp(np.nanmean(Xm[:, 0])) - np.exp(truth)) ** 2)))
    put("m2_impute_rmse_mcar", {k: round(v, 1) for k, v in rmse_mcar.items()})
    put("m2_impute_rmse_mnar", {k: round(v, 1) for k, v in rmse_mnar.items()})
    put("m2_em", {"rmse_em": round(em_rmse, 1), "rmse_mean": round(mean_rmse, 1), "n_masked": int(miss.sum()),
                  "mu": [round(float(x), 3) for x in mu], "corr_pm25_pm10":
                  round(float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), 3)})


def em_mvn(X: np.ndarray, iters: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood mean and covariance of a multivariate normal with missing values (EM)."""
    n, p = X.shape
    mu = np.nanmean(X, axis=0)
    cov = np.diag(np.nanvar(X, axis=0))
    for _ in range(iters):
        Xf = np.where(np.isnan(X), 0, X)
        C = np.zeros((p, p))
        for i in range(n):
            m = np.isnan(X[i])
            if m.any():
                o = ~m
                B = cov[np.ix_(m, o)] @ np.linalg.inv(cov[np.ix_(o, o)])
                Xf[i, m] = mu[m] + B @ (X[i, o] - mu[o])
                C[np.ix_(m, m)] += cov[np.ix_(m, m)] - B @ cov[np.ix_(o, m)]
        mu_new = Xf.mean(axis=0)
        d = Xf - mu_new
        cov_new = (d.T @ d + C) / n
        if np.abs(mu_new - mu).max() < tol:
            return mu_new, cov_new
        mu, cov = mu_new, cov_new
    return mu, cov


# ------------------------------------------------------------------ Module 3


def m3() -> None:
    us = us_delhi_daily()
    put("m3_univariate", {"n": int(len(us)), "mean": round(float(us.mean()), 1), "median": round(float(us.median()), 1),
                          "sd": round(float(us.std()), 1), "skew": round(float(stats.skew(us)), 2),
                          "kurtosis": round(float(stats.kurtosis(us)), 2),
                          "skew_log": round(float(stats.skew(np.log(us))), 2)})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axes[0].hist(us, bins=60, color=INK)
    axes[0].set_title("Daily PM2.5, US Embassy")
    axes[0].set_xlabel("µg/m³")
    axes[1].hist(np.log(us), bins=60, color=ACCENT)
    axes[1].set_title("After a log transform")
    axes[1].set_xlabel("log µg/m³")
    save(fig, "m3_univariate")

    d = delhi_daily_with_weather()
    cols = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂", "co": "CO", "t_mean": "Temperature",
            "rh_mean": "Humidity", "ws_mean": "Wind speed", "blh_mean": "Mixing height", "precip_sum": "Rain"}
    sub = d[list(cols)].dropna().rename(columns=cols)
    pear = sub.corr(method="pearson")
    spear = sub.corr(method="spearman")
    pear.round(3).to_csv(OUT / "m3_pearson.csv")
    spear.round(3).to_csv(OUT / "m3_spearman.csv")
    put("m3_bivariate", {k: {"pearson": round(float(pear.loc["PM2.5", k]), 2), "spearman": round(float(spear.loc["PM2.5", k]), 2)}
                         for k in ["Temperature", "Humidity", "Wind speed", "Mixing height", "Rain", "PM10"]})
    # partial correlation PM2.5 vs mixing height controlling for temperature
    r_pb, r_pt, r_bt = spear.loc["PM2.5", "Mixing height"], spear.loc["PM2.5", "Temperature"], spear.loc["Mixing height", "Temperature"]
    put("m3_partial_pm25_blh_given_t", round(float((r_pb - r_pt * r_bt) / np.sqrt((1 - r_pt ** 2) * (1 - r_bt ** 2))), 3))

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(spear.values, cmap="RdGy_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(spear)), spear.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(spear)), spear.index)
    ax.grid(False)
    for i in range(len(spear)):
        for j in range(len(spear)):
            ax.text(j, i, f"{spear.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(spear.values[i, j]) > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlations, Delhi daily (CPCB network + ERA5)")
    save(fig, "m3_corr")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(sub["Mixing height"], sub["PM2.5"], s=4, alpha=0.35, color=INK)
    ax.set_xlabel("Daily mean mixing height, m")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.set_title("Bivariate: a shallow boundary layer traps pollution")
    save(fig, "m3_scatter_blh")

    # time series: time-based indexing, resampling, grouping, rolling, decomposition
    sh = load_site_hourly()
    h = sh[sh["site"] == US_DELHI].set_index("datetime")["pm25"].sort_index()
    put("m3_indexing_nov2020_mean", round(float(h.loc["2020-11"].mean()), 1))
    put("m3_indexing_diwali2024_peak", round(float(h.loc["2024-10-31 18:00":"2024-11-01 06:00"].max()), 1))
    monthly = h.resample("MS").mean()
    by_month = h.groupby(h.index.month).mean()
    by_hour = h.groupby(h.index.hour).mean()
    put("m3_by_month", {int(k): round(float(v), 1) for k, v in by_month.items()})
    put("m3_by_hour_min_max", {"min_hour": int(by_hour.idxmin()), "min": round(float(by_hour.min()), 1),
                               "max_hour": int(by_hour.idxmax()), "max": round(float(by_hour.max()), 1)})
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf
    mfill = monthly.interpolate(limit_direction="both")
    stl = STL(mfill, period=12, robust=True).fit()
    put("m3_stl_seasonal_amplitude", round(float(stl.seasonal.max() - stl.seasonal.min()), 1))
    put("m3_stl_trend_start_end", [round(float(stl.trend.iloc[0]), 1), round(float(stl.trend.iloc[-1]), 1)])
    put("m3_acf_daily", {str(k): round(float(v), 2) for k, v in zip([1, 7, 30, 182, 365], acf(us.asfreq("D").interpolate(), nlags=365)[[1, 7, 30, 182, 365]])})

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, (y, lab) in zip(axes, [(mfill, "Monthly mean (resampled)"), (stl.trend, "Trend"),
                                   (stl.seasonal, "Seasonal"), (stl.resid, "Residual")]):
        ax.plot(y.index, y.values, color=INK, lw=1.2)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("STL decomposition of Delhi PM2.5 (US Embassy), µg/m³", x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    save(fig, "m3_stl")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axes[0].bar(by_month.index, by_month.values, color=INK)
    axes[0].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axes[0].set_title("Grouped by month")
    axes[1].plot(by_hour.index, by_hour.values, color=EMBER, marker="o", ms=3)
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].set_title("Grouped by hour of day")
    save(fig, "m3_grouping")


# ------------------------------------------------------------------ Module 4


def m4() -> None:
    dc = daily_city()
    g = dc.groupby(["city", "region"])["pm25"]
    summ = pd.DataFrame({"days": g.count(), "mean": g.mean(), "median": g.median(), "sd": g.std(),
                         "iqr": g.quantile(0.75) - g.quantile(0.25), "p95": g.quantile(0.95)})
    summ["cv"] = summ["sd"] / summ["mean"]
    summ = summ[summ["days"] >= 365].round(1).sort_values("mean", ascending=False).reset_index()
    summ.to_csv(OUT / "m4_summary_by_city.csv", index=False)
    put("m4_summary_top5", summ.head(5)[["city", "days", "mean", "median", "sd", "iqr", "p95"]].to_dict("records"))
    put("m4_summary_bottom3", summ.tail(3)[["city", "days", "mean", "median", "sd", "iqr", "p95"]].to_dict("records"))

    # 2-D: PM2.5 vs PM10
    two = dc.dropna(subset=["pm25", "pm10"])
    slope, inter, r, p, se = stats.linregress(two["pm10"], two["pm25"])
    put("m4_pm25_vs_pm10", {"slope": round(slope, 3), "intercept": round(inter, 1), "r": round(r, 3), "n": int(len(two))})

    # contingency table: AQI category by season, Delhi, and a chi-square test of independence
    d = dc[dc["city"] == "Delhi"].copy()
    season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer", 6: "Monsoon",
              7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
    d["season"] = d["date"].dt.month.map(season)
    d["naqi"] = pd.cut(d["pm25"], NAQI_BINS, labels=NAQI_NAMES)
    ct = pd.crosstab(d["season"], d["naqi"]).reindex(["Winter", "Summer", "Monsoon", "Post-monsoon"])
    ct.to_csv(OUT / "m4_contingency_delhi.csv")
    chi2, pval, dof, _ = stats.chi2_contingency(ct.loc[:, ct.sum() > 0])
    n = ct.values.sum()
    cramer = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    put("m4_contingency", {"table": ct.to_dict("index"), "chi2": round(float(chi2), 1), "dof": int(dof),
                           "p": float(pval), "cramers_v": round(float(cramer), 3)})
    # n-D: region x season x pollutant
    dc["season"] = dc["date"].dt.month.map(season)
    nd = dc.groupby([dc["region"].map(REGION_NAME), "season"])[["pm25", "pm10", "no2"]].mean().round(1)
    nd.to_csv(OUT / "m4_nd_region_season.csv")
    put("m4_nd_winter", nd.xs("Winter", level="season").to_dict("index"))

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    samp = two.sample(min(6000, len(two)), random_state=SEED)
    ax.scatter(samp["pm10"], samp["pm25"], s=3, alpha=0.25, color=INK)
    xs = np.linspace(0, samp["pm10"].quantile(0.995), 50)
    ax.plot(xs, inter + slope * xs, color=EMBER, lw=1.6)
    ax.set_xlabel("PM10, µg/m³")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.set_title(f"Scatter plot: PM2.5 vs PM10 (r = {r:.2f})")
    save(fig, "m4_scatter")

    winter = dc[dc["season"] == "Winter"].groupby(["city", "region"])["pm25"].mean().sort_values().reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.scatter(winter["pm25"], winter["city"], c=winter["region"].map(REGION_COL), s=26, zorder=3)
    ax.hlines(winter["city"], 0, winter["pm25"], color=RULE, lw=0.8)
    ax.set_xlabel("Mean winter PM2.5, µg/m³")
    ax.set_title("Dot chart: winter PM2.5 by city")
    ax.grid(axis="y", visible=False)
    for k, c in REGION_COL.items():
        ax.scatter([], [], color=c, label=REGION_NAME[k], s=26)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "m4_dot")

    reg = dc.groupby([dc["region"].map(REGION_NAME), "season"])["pm25"].mean().unstack()
    reg = reg[["Winter", "Post-monsoon", "Summer", "Monsoon"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    reg.plot.bar(ax=ax, color=["#1F211D", "#7C7F78", "#B88A12", "#C9CCC5"], width=0.8, rot=0)
    ax.set_ylabel("Mean PM2.5, µg/m³")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.set_title("Bar plot: mean PM2.5 by region and season")
    save(fig, "m4_bar")


# ------------------------------------------------------------------ Modules 5 and 6 (shared features)


def city_profiles() -> tuple[pd.DataFrame, pd.Series]:
    """Each city's mean PM2.5 in each calendar month, divided by the city's own annual mean,
    so cities are compared by the SHAPE of their year rather than by how polluted they are."""
    dc = daily_city()
    dc["month"] = dc["date"].dt.month
    prof = dc.groupby(["city", "month"])["pm25"].mean().unstack()
    cnt = dc.groupby(["city", "month"])["pm25"].count().unstack()
    prof = prof[(cnt >= 20).all(axis=1)]
    shape = prof.div(prof.mean(axis=1), axis=0)
    region = dc.drop_duplicates("city").set_index("city")["region"].reindex(shape.index)
    return shape, region


def daily_matrix() -> pd.DataFrame:
    d = delhi_daily_with_weather()
    cols = ["pm25", "pm10", "no2", "so2", "co", "t_mean", "rh_mean", "ws_mean", "blh_mean", "precip_sum"]
    X = d[cols].dropna()
    for c in ["pm25", "pm10", "no2", "so2", "co"]:
        X[c] = np.log(X[c].clip(lower=0.1))
    X["precip_sum"] = np.log1p(X["precip_sum"])
    return X


def m5() -> None:
    from sklearn.cluster import SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    shape, region = city_profiles()
    X = shape.values
    put("m5_n_cities", int(len(shape)))
    k = 3

    # hierarchical agglomerative (Ward)
    Z = linkage(X, method="ward")
    hier = fcluster(Z, k, criterion="maxclust")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    dendrogram(Z, labels=shape.index.tolist(), ax=ax, color_threshold=Z[-(k - 1), 2], above_threshold_color=MUTED,
               leaf_font_size=7.5)
    ax.set_title("Hierarchical agglomerative clustering of cities by the shape of their year (Ward)")
    ax.grid(False)
    save(fig, "m5_dendrogram")

    # spectral clustering
    spec = SpectralClustering(n_clusters=k, affinity="rbf", gamma=2.0, random_state=SEED).fit_predict(X)

    # minimum spanning tree clustering: cut the k-1 longest edges
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).toarray()
    edges = np.argwhere(mst > 0)
    w = mst[mst > 0]
    for idx in np.argsort(w)[-(k - 1):]:
        i, j = edges[idx]
        mst[i, j] = 0
    _, mstlab = connected_components(mst + mst.T > 0, directed=False)

    # model-based: Gaussian mixture fitted by EM, k chosen by BIC
    Xs = StandardScaler().fit_transform(X)
    bics = {kk: GaussianMixture(kk, covariance_type="diag", random_state=SEED, n_init=5).fit(Xs).bic(Xs) for kk in range(1, 6)}
    kbest = min(bics, key=bics.get)
    gmm = GaussianMixture(k, covariance_type="diag", random_state=SEED, n_init=5).fit(Xs)
    gmmlab = gmm.predict(Xs)

    from sklearn.metrics import adjusted_rand_score
    labels = pd.DataFrame({"region": region, "hierarchical": hier, "spectral": spec, "mst": mstlab,
                           "gmm_em": gmmlab}, index=shape.index)
    labels.to_csv(OUT / "m5_city_clusters.csv")
    put("m5_mst_sizes", sorted(np.bincount(mstlab).tolist(), reverse=True))
    put("m5_bic", {str(k_): round(float(v), 1) for k_, v in bics.items()})
    put("m5_bic_best_k", int(kbest))
    put("m5_agreement_ari", {
        "hierarchical_vs_spectral": round(adjusted_rand_score(hier, spec), 2),
        "hierarchical_vs_mst": round(adjusted_rand_score(hier, mstlab), 2),
        "hierarchical_vs_gmm": round(adjusted_rand_score(hier, gmmlab), 2),
        "hierarchical_vs_region": round(adjusted_rand_score(hier, region.astype("category").cat.codes), 2)})
    groups = {int(c): sorted(shape.index[hier == c].tolist()) for c in np.unique(hier)}
    put("m5_hier_groups", groups)
    amp = (shape.max(axis=1) - shape.min(axis=1))
    put("m5_group_amplitude", {int(c): round(float(amp[hier == c].mean()), 2) for c in np.unique(hier)})

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    pal = ["#1F211D", "#A9491B", "#B88A12", "#7C7F78"]
    for c in np.unique(hier):
        for row in X[hier == c]:
            ax.plot(range(1, 13), row, color=pal[(c - 1) % 4], lw=0.7, alpha=0.35)
        ax.plot(range(1, 13), X[hier == c].mean(axis=0), color=pal[(c - 1) % 4], lw=2.4, label=f"Cluster {c} ({(hier == c).sum()} cities)")
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylabel("PM2.5 ÷ city's annual mean")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("The three seasonal shapes found by clustering")
    save(fig, "m5_profiles")

    # outlier detection using clustering: lowest-likelihood days under a Gaussian mixture
    Xd = daily_matrix()
    pol = Xd[["pm25", "pm10", "no2", "so2", "co"]]
    Zd = StandardScaler().fit_transform(pol)
    g = GaussianMixture(4, covariance_type="full", random_state=SEED, n_init=3).fit(Zd)
    ll = pd.Series(g.score_samples(Zd), index=pol.index)
    cut = ll.quantile(0.01)
    out = ll[ll <= cut].index
    dv = set(diwali_dates().values())
    near = [d for d in out if any(abs((d - x).days) <= 1 for x in dv)]
    put("m5_outliers", {"n_days": int(len(pol)), "n_outliers": int(len(out)), "n_near_diwali": int(len(near)),
                        "diwali_days_available": int(sum(1 for x in dv if x in pol.index)),
                        "examples": [d.strftime("%Y-%m-%d") for d in out[:12]]})


def m6() -> None:
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.manifold import MDS, Isomap
    from sklearn.preprocessing import StandardScaler

    Xd = daily_matrix()
    names = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂", "co": "CO", "t_mean": "Temperature",
             "rh_mean": "Humidity", "ws_mean": "Wind", "blh_mean": "Mixing height", "precip_sum": "Rain"}
    Z = StandardScaler().fit_transform(Xd)
    pca = PCA().fit(Z)
    ev = pca.explained_variance_ratio_
    put("m6_pca_explained", [round(float(x), 3) for x in ev])
    put("m6_intrinsic_dim_90", int(np.searchsorted(np.cumsum(ev), 0.90) + 1))
    put("m6_intrinsic_dim_kaiser", int((pca.explained_variance_ > 1).sum()))
    load = pd.DataFrame(pca.components_[:3].T, index=[names[c] for c in Xd.columns], columns=["PC1", "PC2", "PC3"]).round(2)
    load.to_csv(OUT / "m6_pca_loadings.csv")
    put("m6_pca_loadings", load.to_dict("index"))

    # SVD: the same decomposition, by hand
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    put("m6_svd_matches_pca", bool(np.allclose((S ** 2) / (S ** 2).sum(), ev, atol=1e-8)))
    put("m6_svd_singular_values", [round(float(x), 1) for x in S[:5]])

    fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=SEED).fit(Z)
    fal = pd.DataFrame(fa.components_.T, index=[names[c] for c in Xd.columns], columns=["Factor 1", "Factor 2"]).round(2)
    fal.to_csv(OUT / "m6_factor_loadings.csv")
    put("m6_factor_loadings", fal.to_dict("index"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].bar(range(1, len(ev) + 1), ev, color=INK)
    axes[0].plot(range(1, len(ev) + 1), np.cumsum(ev), color=EMBER, marker="o", ms=3)
    axes[0].axhline(0.9, color=RULE, ls="--", lw=0.8)
    axes[0].set_title("Scree plot and cumulative variance")
    axes[0].set_xlabel("Component")
    pcs = pca.transform(Z)
    sc = axes[1].scatter(pcs[:, 0], pcs[:, 1], s=3, c=Xd.index.month, cmap="twilight", alpha=0.6)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Days projected on PC1–PC2 (colour = month)")
    fig.colorbar(sc, ax=axes[1], ticks=[1, 4, 7, 10])
    save(fig, "m6_pca")

    # non-linear methods on city profiles
    shape, region = city_profiles()
    Xc = shape.values
    mds = MDS(n_components=2, random_state=SEED, n_init=4, normalized_stress="auto").fit(Xc)
    iso = Isomap(n_neighbors=6, n_components=2).fit_transform(Xc)
    put("m6_mds_stress", round(float(mds.stress_), 3))
    som_units, som_grid = som(Xc, rows=4, cols=4, iters=4000)
    put("m6_som_units", {city: [int(u[0]), int(u[1])] for city, u in zip(shape.index, som_units)})

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, emb, title in [(axes[0], mds.embedding_, "MDS"), (axes[1], iso, "Isomap (manifold learning)")]:
        ax.scatter(emb[:, 0], emb[:, 1], c=region.map(REGION_COL), s=22)
        for (x, y), c in zip(emb, shape.index):
            if c in ("Delhi", "Chennai", "Mumbai", "Kolkata", "Bathinda", "Bengaluru", "Lucknow"):
                ax.annotate(c, (x, y), fontsize=6.5, xytext=(3, 2), textcoords="offset points", color=MUTED)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[2]
    jit = np.random.default_rng(SEED).uniform(-0.25, 0.25, (len(som_units), 2))
    ax.scatter(som_units[:, 1] + jit[:, 1], som_units[:, 0] + jit[:, 0], c=region.map(REGION_COL), s=22)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(3.6, -0.6)
    ax.set_title("Self-organising map (4×4)")
    for k, c in REGION_COL.items():
        axes[0].scatter([], [], color=c, label=REGION_NAME[k], s=22)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    save(fig, "m6_nonlinear")


def som(X: np.ndarray, rows: int, cols: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """A small self-organising map in numpy: returns each sample's best-matching unit."""
    rng = np.random.default_rng(SEED)
    W = X[rng.integers(0, len(X), rows * cols)].astype(float).copy()
    grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
    for t in range(iters):
        frac = t / iters
        lr = 0.5 * (1 - frac) + 0.01
        sigma = max(rows, cols) / 2 * (1 - frac) + 0.5
        x = X[rng.integers(0, len(X))]
        bmu = np.argmin(((W - x) ** 2).sum(axis=1))
        h = np.exp(-((grid - grid[bmu]) ** 2).sum(axis=1) / (2 * sigma ** 2))
        W += lr * h[:, None] * (x - W)
    bmus = np.array([grid[np.argmin(((W - x) ** 2).sum(axis=1))] for x in X])
    return bmus, W


# ------------------------------------------------------------------ Module 7


def m7() -> None:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score

    us = us_delhi_daily()
    met = load_meteo_hourly()
    md = models.met_daily(met[met["city"] == "Delhi"])
    d = md.join(us.rename("pm25"), how="inner").dropna()
    feats = ["t_mean", "t_range", "rh_mean", "precip_sum", "ws_mean", "u_mean", "v_mean", "blh_mean", "blh_min", "sw_sum"]
    X = d[feats].copy()
    X["precip_sum"] = np.log1p(X["precip_sum"])
    month = pd.get_dummies(d.index.month, prefix="m", drop_first=True).set_index(d.index).astype(float)
    X = X.join(month)
    y = np.log(d["pm25"])
    train = d.index.year <= 2022
    test = d.index.year >= 2023
    lr = LinearRegression().fit(X[train], y[train])
    pred = np.exp(lr.predict(X[test]))
    obs = d["pm25"][test]
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    coefs = pd.Series(lr.coef_, index=X.columns)
    put("m7_linear", {"n_train": int(train.sum()), "n_test": int(test.sum()),
                      "r2_log_test": round(float(r2_score(y[test], lr.predict(X[test]))), 3),
                      "r2_log_train": round(float(r2_score(y[train], lr.predict(X[train]))), 3),
                      "rmse": round(rmse, 1), "mae": round(float(mean_absolute_error(obs, pred)), 1),
                      "coef_blh_mean_per_100m_pct": round(float(100 * (np.exp(coefs["blh_mean"] * 100) - 1)), 1),
                      "coef_ws_per_ms_pct": round(float(100 * (np.exp(coefs["ws_mean"]) - 1)), 1)})
    # gradient boosting for comparison (same features, same split)
    gb, smear = models.fit(X[train], d["pm25"][train], n_estimators=400)
    pg = models.predict(gb, smear, X[test])
    put("m7_lightgbm", {"r2_log_test": round(float(r2_score(y[test], np.log(pg))), 3),
                        "rmse": round(float(np.sqrt(np.mean((pg - obs) ** 2))), 1),
                        "mae": round(float(mean_absolute_error(obs, pg)), 1)})
    # accuracy: does the predicted AQI category match the observed one?
    cat_o = pd.cut(obs, NAQI_BINS, labels=False)
    cat_p = pd.cut(pd.Series(pred, index=obs.index), NAQI_BINS, labels=False)
    cat_g = pd.cut(pd.Series(pg, index=obs.index), NAQI_BINS, labels=False)
    acc = float((cat_o == cat_p).mean())
    within1 = float((abs(cat_o - cat_p) <= 1).mean())
    majority = float((cat_o == cat_o.mode()[0]).mean())
    cm = confusion_matrix(cat_o, cat_p, labels=range(6))
    pd.DataFrame(cm, index=NAQI_NAMES, columns=NAQI_NAMES).to_csv(OUT / "m7_confusion.csv")
    put("m7_accuracy", {"linear_exact": round(acc, 3), "linear_within_one": round(within1, 3),
                        "lightgbm_exact": round(float((cat_o == cat_g).mean()), 3),
                        "majority_baseline": round(majority, 3)})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].scatter(obs, pred, s=5, alpha=0.4, color=INK)
    lim = [0, max(obs.max(), pred.max()) * 1.02]
    axes[0].plot(lim, lim, color=EMBER, lw=1)
    axes[0].set_xlabel("Observed PM2.5, µg/m³")
    axes[0].set_ylabel("Predicted (linear model)")
    axes[0].set_title("Test years 2023–2026")
    im = axes[1].imshow(cm, cmap="Greys")
    axes[1].set_xticks(range(6), ["Good", "Satis.", "Mod.", "Poor", "V.poor", "Severe"], rotation=40, ha="right")
    axes[1].set_yticks(range(6), ["Good", "Satis.", "Mod.", "Poor", "V.poor", "Severe"])
    axes[1].set_xlabel("Predicted category")
    axes[1].set_ylabel("Observed category")
    axes[1].grid(False)
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                             color="white" if cm[i, j] > cm.max() * 0.5 else INK)
    axes[1].set_title(f"AQI category accuracy {acc:.0%}")
    save(fig, "m7_regression")


def run() -> None:
    _style()
    for f in (m2, m3, m4, m5, m6, m7):
        f()
        print(f"  {f.__name__} done", flush=True)
    (OUT / "modules.json").write_text(json.dumps(RES, indent=1, default=str), encoding="utf-8")
    print(f"  {len(RES)} results written to results/modules/modules.json")


if __name__ == "__main__":
    run()
