"""Every number quoted in the paper and on the website comes from here.

numbers() reads the result files written by the analysis stages and returns a flat dict.
The paper template and the site templates both render from this dict, so a number is
never typed by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from elevenwinters import config
from elevenwinters.analysis.common import load_json

R = config.results_dir


def fmt(x, digits: int = 0, pct: bool = False, sign: bool = False) -> str:
    """Format a number for prose: thousands separators, fixed digits, optional % or sign."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if pct:
        x = 100 * x
    s = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    s = s.replace("-", "−")
    return s + ("%" if pct else "")


def _safe(fn, default=None):
    try:
        return fn()
    except (FileNotFoundError, KeyError, ValueError):
        return default


def numbers() -> dict:
    n: dict = {}
    s = config.settings()
    n["period_start"], n["period_end"] = s["period"]["start"], s["period"]["end"]

    sites = _safe(lambda: pd.read_csv(R() / "sites.csv"))
    if sites is not None:
        n["n_sites"] = len(sites)
        n["n_sites_cpcb"] = int((sites["network"] == "cpcb").sum())
        n["n_sites_us"] = int((sites["network"] == "us_mission").sum())
        n["n_cities"] = sites["city"].nunique()
        n["n_locations"] = int(sites["n_locations"].sum())
    qc = _safe(lambda: pd.read_csv(R() / "qc_report.csv"))
    if qc is not None:
        pm = qc[qc["parameter"] == "pm25"]
        n["pm25_hours"] = int(pm["n"].sum())
        flags = [c for c in ["invalid", "flatline", "stall", "spike", "pm_ratio"] if c in pm]
        n["pm25_flagged_share"] = pm[flags].sum().sum() / pm["n"].sum()
        for c in flags:
            n[f"pm25_flag_{c}"] = int(pm[c].sum())

    skill = _safe(lambda: pd.read_csv(R() / "model_skill_daily.csv"))
    if skill is not None:
        n["daily_models"] = len(skill)
        n["daily_r2_median"] = skill["r2_cv"].median()
        n["daily_r2_q1"], n["daily_r2_q3"] = skill["r2_cv"].quantile([0.25, 0.75])
        us = skill[skill["site"] == "us_delhi"]
        n["daily_r2_us_delhi"] = float(us["r2_cv"].iloc[0]) if len(us) else np.nan

    gap = _safe(lambda: load_json("parta_seasonal_summary.json"))
    if gap:
        n.update({f"gap_{k}": v for k, v in gap.items()})
    city_gap = _safe(lambda: pd.read_csv(R() / "seasonal_gap_by_city.csv"))
    if city_gap is not None:
        d = city_gap[city_gap["city"] == "Delhi"]
        if len(d):
            n["gap_delhi_obs"] = float(d["obs_gap"].iloc[0])
            n["gap_delhi_norm"] = float(d["norm_gap"].iloc[0])
            n["gap_delhi_weather_share"] = float(d["weather_share"].iloc[0])
        n["gap_cities"] = len(city_gap)
        n["gap_weather_share_range"] = (city_gap["weather_share"].min(), city_gap["weather_share"].max())

    tr = _safe(lambda: load_json("parta_trend_delhi.json"))
    if tr:
        n["trend_pct_per_year"] = tr["slope_pct_per_year"]
        n["trend_slope"] = tr["theil_sen_slope_per_year"]
        n["trend_ci"] = tr["slope_ci95"]
        n["trend_p"] = tr["kendall_p"]
        n["trend_change_points"] = tr["change_points"]
        n["trend_annual"] = tr["annual_normalised"]
        n["trend_post2020_vs_2019_pct"] = tr.get("mean_2021_2025_vs_2019_pct")

    b = _safe(lambda: load_json("partb_summary.json"))
    if b:
        n["b"] = b
        m = b["main"]
        n["diwali_main_mean"], n["diwali_main_min"], n["diwali_main_max"] = m["mean"], m["min"], m["max"]
        n["diwali_main_n"] = m["n_diwalis"]
        n["diwali_main_years"] = m["years"]
        n["diwali_placebo_p95"] = m["placebo"]["p95"]
        n["diwali_placebo_p99"] = m["placebo"]["p99"]
        n["diwali_placebo_mean"] = m["placebo"]["mean"]
        n["diwali_placebo_n"] = m["placebo"]["n_nights"]
        n["diwali_above_p99"] = m["n_above_p99"]
        n["diwali_kruskal_p"] = m.get("kruskal_p")
        for reg, v in m["by_regime"].items():
            n[f"diwali_{reg}_mean"] = v["mean"]
            n[f"diwali_{reg}_years"] = v["years"]
            n[f"diwali_{reg}_values"] = v["values"]
        c = b["censoring"]
        n["cens_diwali"] = c["diwali_late_hours_missing_share"]
        n["cens_ordinary"] = c["ordinary_late_hours_missing_share"]
        n["cens_ratio"] = c["diwali_late_hours_missing_share"] / max(c["ordinary_late_hours_missing_share"], 1e-9)
        n["cens_nights"] = c["diwali_nights_with_missing_late_hours"]
        n["cens_n"] = c["n_diwalis"]
        n["cens_network"] = c["network_diwali_late_missing_share"]
        n["us_evening_mean"] = b["us_embassy"]["evening_mean"]
        n["us_evening_p95"] = b["us_embassy"]["placebo_evening"]["p95"]
        n["fire_term_median_change"] = b["fire_term_effect_on_diwali_excess"]["median_change"]
        for tag in ["fire", "nofire"]:
            k = f"vs_fire_us_delhi_night_excess_lb_{tag}"
            if k in b:
                n[f"diwali_fire_rho_{tag}"] = b[k]["spearman_rho"]
                n[f"diwali_fire_p_{tag}"] = b[k]["p"]
        if "us_vs_cpcb_night" in b:
            n["us_cpcb_r"] = b["us_vs_cpcb_night"]["pearson_r"]
            n["us_cpcb_years"] = b["us_vs_cpcb_night"]["n_years"]
        n["r2_hourly_us_delhi"] = b.get("r2_us_delhi_fire")
        n["r2_hourly_cpcb_delhi"] = b.get("r2_cpcb_delhi_fire")
    fp = _safe(lambda: load_json("partb_fingerprint.json"))
    if fp:
        n["fp"] = fp

    c = _safe(lambda: load_json("partc_summary.json"))
    if c:
        n["c"] = c
        det = {int(k): v for k, v in c["viirs_detections_by_season"].items()}
        ns = {int(k): v for k, v in c["viirs_night_share_by_season"].items()}
        n["viirs_2021"], n["viirs_2025"] = det.get(2021), det.get(2025)
        n["viirs_peak_season"] = max(det, key=det.get)
        n["viirs_peak"] = max(det.values())
        n["viirs_drop_2021_2025"] = 1 - det[2025] / det[2021]
        n["night_share_2016_2021"] = float(np.mean([ns[y] for y in range(2016, 2022)]))
        n["night_share_2024"], n["night_share_2025"] = ns.get(2024), ns.get(2025)
        n["best_lag"] = c["best_lag_days"]
        n["coupling_early"] = c.get("coupling_mean_2016_2021")
        n["coupling_2025"] = c["coupling"].get("2025", {}).get("pct_per_100")
        n["coupling_2025_lo"] = c["coupling"].get("2025", {}).get("pct_lo")
        n["coupling_2025_hi"] = c["coupling"].get("2025", {}).get("pct_hi")
        ev = c.get("evening_ratio")
        if ev:
            n["evening"] = ev
    fs = _safe(lambda: pd.read_csv(R() / "fire_season_summary.csv"))
    if fs is not None:
        v = fs[fs["product"] == "viirs-snpp"]
        nn = v[v["daynight"] == "N"].groupby(["state", "season"])["detections_clean"].sum()
        n["night_punjab"] = {int(k): int(x) for k, x in nn.get("Punjab", pd.Series(dtype=int)).items()}
        n["night_haryana"] = {int(k): int(x) for k, x in nn.get("Haryana", pd.Series(dtype=int)).items()}
        n["night_punjab_2016_2023_max"] = max(n["night_punjab"][y] for y in range(2016, 2024))
        n["night_haryana_2016_2023_max"] = max(n["night_haryana"][y] for y in range(2016, 2024))
    lag = _safe(lambda: pd.read_csv(R() / "fire_lag_profile.csv"))
    if lag is not None:
        li = lag[lag["predictor"] == "fi"].set_index("lag_days")["spearman_rho"]
        n["lag_rho"] = li.round(2).to_dict()

    prereg = config.ROOT / "prereg" / "diwali-2026-forecast.json"
    if prereg.exists():
        rec = json.loads(prereg.read_text(encoding="utf-8"))
        n["forecast"] = rec
        n["forecast_sha"] = (config.ROOT / "prereg" / "SHA256SUMS").read_text().split()[0]
    return n


FIG_META = [
    ("fig01_design", "Diwali moves through the burning season, and the rules keep changing",
     "Diwali (Lakshmi Puja) dates, 2016–2026, and the firecracker rules in force in Delhi-NCR. The shaded band marks "
     "1–15 November, commonly reported as the peak of paddy-residue burning.",
     "Supreme Court, NGT and Delhi government orders as reported; panchang dates."),
    ("fig02_coverage", "The open archive has a two-year hole in India's official network",
     "Share of hours with a valid PM2.5 reading, by city and month. CPCB-network data are missing from the OpenAQ "
     "archive from late 2022 to early 2025; the US mission monitors run almost without a break.",
     "CPCB and US Department of State monitors via the OpenAQ archive."),
    ("fig03_tape", "Delhi's decade, drawn as filter tape",
     "Daily mean PM2.5 at the US Embassy, one strip per season from July to June, shaded by India's National AQI "
     "category. Marigold ticks mark Diwali.",
     "US Embassy New Delhi monitor via OpenAQ; NAQI breakpoints from CPCB."),
    ("fig04_gap", "Weather explains part of the winter gap, not all of it",
     "Winter (Nov–Jan) minus monsoon (Jul–Sep) PM2.5 by city, as observed (open circle) and after removing weather "
     "(filled). The number is the share of the gap that weather accounts for.",
     "CPCB and US mission monitors; ERA5 via Open-Meteo; LightGBM weather normalisation."),
    ("fig05_trend", "In Delhi, the 2020 dip did not become a lasting break",
     "Monthly PM2.5 at the US Embassy, observed and weather-normalised. Dashed lines mark change points found in the "
     "normalised series.",
     "US Embassy New Delhi; ERA5 via Open-Meteo; PELT change-point detection."),
    ("fig06_event", "Every Diwali night rises far above the weather-and-fire baseline",
     "Hourly PM2.5 above the business-as-usual baseline (weather, calendar and wind-aware fire index) around Diwali "
     "night, CPCB Delhi network median, one line per year coloured by the firecracker rules. Shading marks 6 pm–3 am.",
     "CPCB Delhi network via OpenAQ; NASA FIRMS; ERA5."),
    ("fig07_nights", "Bans did not shrink the Diwali spike",
     "Diwali-night excess PM2.5 by year. Filled points: CPCB network median, coloured by regime. Open points: US "
     "Embassy, with missing hours carried forward. The band covers 90% of ordinary October–December nights.",
     "CPCB and US Embassy monitors via OpenAQ; NASA FIRMS; ERA5."),
    ("fig08_censoring", "The reference monitor goes dark when the air is worst",
     "Share of hours with no valid reading at the US Embassy monitor, Diwali nights against ordinary autumn nights.",
     "US Embassy New Delhi via OpenAQ."),
    ("fig09_cities", "The timing follows local custom",
     "Median hourly excess PM2.5 around Diwali in five cities. Dotted lines mark the midnights before and after "
     "Diwali day.",
     "CPCB Delhi network and US mission monitors via OpenAQ; ERA5."),
    ("fig10_fingerprint", "Diwali night carries a combustion fingerprint",
     "Diwali evening (6 pm–midnight) against the same hours on the five nights a week earlier, at CPCB Delhi sites. "
     "Thin lines are site-years; the thick line joins the medians.",
     "CPCB Delhi monitors via OpenAQ."),
    ("fig11_fires", "Daytime detections collapsed while night detections rose",
     "VIIRS S-NPP fire detections in Punjab and Haryana, October–November, split by overpass, after removing fixed "
     "heat sources. Diamonds mark official Punjab season counts.",
     "NASA FIRMS VIIRS S-NPP; official counts as reported by PTI and the Government of India."),
    ("fig12_coupling", "Each detected fire now goes with more smoke in Delhi",
     "Percent change in Delhi's weather-adjusted PM2.5 per 100 units of the wind-aware fire index, by season, with "
     "95% confidence intervals (HAC standard errors).",
     "US Embassy New Delhi; NASA FIRMS; ERA5."),
    ("fig13_evening", "Ground monitors in Punjab and Haryana do not yet show an evening shift",
     "Evening-to-midday PM2.5 ratio in the harvest window divided by the same ratio in December, per monitor and "
     "season. Ticks mark the median. Values above 1 mean the evening share rises at harvest.",
     "CPCB monitors in Punjab, Haryana and Chandigarh via OpenAQ."),
    ("fig14_forecast", "Our locked forecast for Diwali 2026",
     "Forecast Diwali-night excess PM2.5 for 8 November 2026 under each possible firecracker regime, with 80% "
     "prediction intervals. Open circles are past Diwalis under the same rules.",
     "This study, prereg/diwali-2026-forecast.json."),
]


def figure_meta() -> dict:
    return {name: {"n": i + 1, "title": title, "caption": cap, "source": src}
            for i, (name, title, cap, src) in enumerate(FIG_META)}


def write_numbers() -> None:
    n = numbers()
    keep = {k: v for k, v in n.items() if not isinstance(v, (dict, list, tuple))}
    (R() / "numbers.json").write_text(json.dumps(keep, indent=2, default=float), encoding="utf-8")
    print(f"  {len(keep)} scalar numbers written to results/numbers.json")
