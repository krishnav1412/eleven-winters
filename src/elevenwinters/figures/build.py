"""Every figure, defined once in Altair.

Each spec is exported three ways: a vector PDF and a PNG for the paper (light, white
background), and Vega-Lite JSON in light and dark versions for the website.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

from elevenwinters import config
from elevenwinters.analysis.common import US_DELHI, daily_mean, diwali_dates, load_json, regimes
from elevenwinters.figures.theme import (FONT_LABEL, FONT_NUM, NAQI, REGIME_LABEL, SOOT, TAPE, pal,
                                         vl_config)

PAPER_DIR = config.ROOT / "paper" / "figures"
WEB_DIR = config.ROOT / "site" / "static" / "figs"
FONT_DIR = config.ROOT / "assets" / "fonts"
W = 640  # full-width figure, px

alt.data_transformers.disable_max_rows()

REGIME_ORDER = ["permissive", "green_window", "full_ban"]
US_NAMES = {"us_delhi": "New Delhi", "us_mumbai": "Mumbai", "us_kolkata": "Kolkata",
            "us_hyderabad": "Hyderabad", "us_chennai": "Chennai"}


def _regime_scale(mode: str) -> alt.Scale:
    p = pal(mode)
    return alt.Scale(domain=[REGIME_LABEL[r] for r in REGIME_ORDER], range=[p["regime"][r] for r in REGIME_ORDER])


def _season_md(dates: pd.Series) -> pd.Series:
    """Map dates to a common dummy year so different years share one axis."""
    return pd.to_datetime("2000-" + dates.dt.strftime("%m-%d"))


# ------------------------------------------------------------------ figures


def fig_design(mode: str):
    """Fig 1. Diwali dates and the firecracker rules in Delhi-NCR."""
    p = pal(mode)
    rules = {r["year"]: r for r in config.events()["cracker_rules_delhi"]}
    df = pd.DataFrame([{"year": y, "date": d, "regime": rules[y]["regime"]} for y, d in diwali_dates().items()])
    df["md"] = _season_md(df["date"])
    df["label"] = df["date"].dt.strftime("%-d %b") if hasattr(df["date"].dt, "strftime") else df["date"]
    df["label"] = df["date"].dt.day.astype(str) + " " + df["date"].dt.strftime("%b")
    df["regime_label"] = df["regime"].map(REGIME_LABEL)
    df["year_s"] = df["year"].astype(str)
    y = alt.Y("year_s:O", title=None, sort=df["year_s"].tolist(),
              axis=alt.Axis(labelFont=FONT_NUM, labelColor=p["ink"], labelFontSize=11.5))
    x = alt.X("md:T", title=None, scale=alt.Scale(domain=["2000-10-15", "2000-11-20"]),
              axis=alt.Axis(format="%e %b", values=[f"2000-{m}" for m in
                                                    ["10-15", "10-22", "10-29", "11-05", "11-12", "11-19"]],
                            grid=True, gridColor=p["grid"], gridDash=[2, 3]))
    band = alt.Chart(pd.DataFrame({"a": ["2000-11-01"], "b": ["2000-11-15"]})).mark_rect(
        color=p["ember"], opacity=0.13).encode(x="a:T", x2="b:T")
    band_lbl = alt.Chart(pd.DataFrame({"a": ["2000-11-01"], "t": ["1-15 Nov: commonly reported burning peak"]})).mark_text(
        align="left", baseline="bottom", dx=4, dy=-4, color=p["ember"], fontSize=11).encode(x="a:T", y=alt.value(0), text="t")
    pts = alt.Chart(df).mark_point(filled=True, size=90, color=p["marigold"], stroke=None).encode(
        x=x, y=y, shape=alt.condition(alt.datum.year == 2026, alt.value("diamond"), alt.value("circle")),
        tooltip=["year", "label", "regime_label"])
    lbl = alt.Chart(df).mark_text(align="left", dx=9, font=FONT_NUM, fontSize=10.5, color=p["muted"]).encode(
        x=x, y=y, text="label")
    left = (band + band_lbl + pts + lbl).properties(width=420, height=330)
    right = alt.Chart(df).mark_text(align="left", fontSize=12, color=p["ink"]).encode(
        x=alt.value(0), y=alt.Y("year_s:O", sort=df["year_s"].tolist(), axis=None), text="regime_label",
        color=alt.condition(alt.datum.regime == "unknown", alt.value(p["muted"]), alt.value(p["ink"]))
    ).properties(width=200, height=330, title=alt.TitleParams("Firecracker rules, Delhi-NCR", fontSize=11,
                                                                 color=p["muted"], font=FONT_LABEL, fontWeight=500))
    return alt.hconcat(left, right, spacing=10)


def fig_coverage(mode: str):
    """Fig 2. Share of hours with PM2.5 data, by city and month."""
    p = pal(mode)
    sh = pd.read_parquet(config.processed_dir() / "site_hourly.parquet", columns=["datetime", "site", "city", "region", "network", "pm25"])
    sh = sh.dropna(subset=["pm25"])
    sh["month"] = sh["datetime"].dt.to_period("M").dt.to_timestamp()
    sh["row"] = np.where(sh["network"] == "us_mission", "US mission: " + sh["city"], sh["city"])
    cov = sh.groupby(["row", "region", "month"])["datetime"].nunique().reset_index(name="hours")
    cov["share"] = (cov["hours"] / (cov["month"].dt.days_in_month * 24)).clip(upper=1)
    order_region = {"delhi_ncr": 0, "source": 1, "igp": 2, "rest": 3}
    rows = (cov.assign(us=cov["row"].str.startswith("US"), ro=cov["region"].map(order_region))
            .drop_duplicates("row").sort_values(["us", "ro", "row"])["row"].tolist())
    return alt.Chart(cov).mark_rect().encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", labelFont=FONT_NUM)),
        y=alt.Y("row:N", sort=rows, title=None, axis=alt.Axis(labelFont=FONT_LABEL, labelFontSize=10, labelColor=p["ink"])),
        color=alt.Color("share:Q", title="Share of hours with data",
                        scale=alt.Scale(domain=[0, 1], range=[p["band"], p["ink"]]),
                        legend=alt.Legend(format="%", gradientLength=160, orient="top")),
        tooltip=["row", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("share:Q", format=".0%")],
    ).properties(width=W, height=alt.Step(11))


def fig_tape(mode: str):
    """Fig 3. Delhi's decade as filter tape: daily PM2.5 at the US Embassy, one strip per season."""
    p = pal(mode)
    sh = pd.read_parquet(config.processed_dir() / "site_hourly.parquet", columns=["datetime", "site", "pm25"])
    s = sh[sh["site"] == US_DELHI].set_index("datetime")["pm25"].sort_index()
    d = daily_mean(s, min_hours=16).dropna().reset_index(name="pm25")
    d["season_start"] = np.where(d["date"].dt.month >= 7, d["date"].dt.year, d["date"].dt.year - 1)
    d = d[d["season_start"] >= 2016]
    d["season"] = d["season_start"].astype(str) + "–" + (d["season_start"] + 1).astype(str).str[2:]
    d["day"] = (d["date"] - pd.to_datetime(d["season_start"].astype(str) + "-07-01")).dt.days
    d["cat"] = pd.cut(d["pm25"], bins=[-1] + [b[1] for b in NAQI], labels=[b[2] for b in NAQI]).astype(str)
    dv = pd.DataFrame([{"date": dt} for dt in diwali_dates().values() if dt <= pd.Timestamp("2026-09-30")])
    dv["season_start"] = np.where(dv["date"].dt.month >= 7, dv["date"].dt.year, dv["date"].dt.year - 1)
    dv["season"] = dv["season_start"].astype(str) + "–" + (dv["season_start"] + 1).astype(str).str[2:]
    dv["day"] = (dv["date"] - pd.to_datetime(dv["season_start"].astype(str) + "-07-01")).dt.days
    seasons = sorted(d["season"].unique())
    month_starts = [0, 31, 62, 92, 123, 153, 184, 215, 243, 274, 304, 335]
    month_names = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    label_expr = "{" + ",".join(f"'{a}':'{b}'" for a, b in zip(month_starts, month_names)) + "}[datum.value]"
    x = alt.X("day:Q", title=None, scale=alt.Scale(domain=[0, 366], nice=False),
              axis=alt.Axis(values=month_starts, labelExpr=label_expr, labelAlign="left", labelFont=FONT_LABEL,
                            grid=False, domain=False, ticks=False))
    y = alt.Y("season:O", sort=seasons, title=None, axis=alt.Axis(labelFont=FONT_NUM, labelColor=p["ink"],
                                                                   domain=False, ticks=False, grid=False))
    # one strip of white filter tape per season; it stays white in dark mode, like the object
    tape = alt.Chart(pd.DataFrame({"season": seasons, "a": 0, "b": 366})).mark_rect(
        height=16, color=TAPE, stroke="#C9CCC5", strokeWidth=0.5).encode(x="a:Q", x2="b:Q", y=y)
    cells = alt.Chart(d).mark_rect(width=2.6, height=12).encode(
        x=x, y=y,
        color=alt.Color("cat:N", title="Daily PM2.5 (µg/m³) by National AQI category",
                        scale=alt.Scale(domain=[b[2] for b in NAQI], range=SOOT),
                        legend=alt.Legend(orient="top", symbolType="circle", symbolSize=120, symbolStrokeColor="#A3A69F",
                                          titleLimit=600, labelLimit=200,
                                          symbolStrokeWidth=0.5,
                                          labelExpr="{'Good':'0–30 Good','Satisfactory':'31–60','Moderate':'61–90','Poor':'91–120','Very poor':'121–250','Severe':'250+ Severe'}[datum.label]")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("pm25:Q", format=".0f", title="PM2.5")],
    )
    marks = alt.Chart(dv).mark_tick(color="#B88A12", thickness=2.5, size=22).encode(
        x="day:Q", y=alt.Y("season:O", sort=seasons), tooltip=[alt.Tooltip("date:T", format="Diwali, %d %b %Y")])
    return (tape + cells + marks).properties(width=W + 120, height=alt.Step(21))


def fig_gap(mode: str):
    """Fig 4. Winter minus monsoon PM2.5, observed vs weather-normalised, by city."""
    p = pal(mode)
    g = pd.read_csv(config.results_dir() / "seasonal_gap_by_city.csv")
    names = {"delhi_ncr": "Delhi-NCR", "source": "Punjab and Haryana", "igp": "Rest of the Gangetic Plain",
             "rest": "Outside the plain"}
    g["group"] = g["region"].map(names)
    g = g.sort_values("obs_gap", ascending=False)
    long = g.melt(id_vars=["city", "group", "weather_share"], value_vars=["obs_gap", "norm_gap"],
                  var_name="kind", value_name="gap")
    long["kind"] = long["kind"].map({"obs_gap": "Observed", "norm_gap": "Weather removed"})
    y = alt.Y("city:N", sort=g["city"].tolist(), title=None, axis=alt.Axis(labelColor=p["ink"], labelFont=FONT_LABEL))
    rule = alt.Chart(g).mark_rule(color=p["faint"], strokeWidth=1.2).encode(y=y, x="norm_gap:Q", x2="obs_gap:Q")
    pts = alt.Chart(long).mark_point(size=55, strokeWidth=1.6).encode(
        y=y, x=alt.X("gap:Q", title="Winter (Nov–Jan) minus monsoon (Jul–Sep) PM2.5, µg/m³"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "Weather removed"], range=[p["muted"], p["ink"]]),
                        title=None),
        fill=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "Weather removed"], range=["transparent", p["ink"]]),
                       legend=None),
        tooltip=["city", "kind", alt.Tooltip("gap:Q", format=".0f")])
    share = alt.Chart(g).mark_text(align="left", dx=8, font=FONT_NUM, fontSize=10, color=p["slate"]).encode(
        y=y, x="obs_gap:Q", text=alt.Text("weather_share:Q", format=".0%"))
    return (rule + pts + share).properties(width=W - 120, height=alt.Step(15)).facet(
        row=alt.Row("group:N", sort=list(names.values()), title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelAnchor="start", labelOrient="top")),
    ).resolve_scale(y="independent")


def fig_trend(mode: str):
    """Fig 5. Delhi (US Embassy) monthly PM2.5, observed and weather-normalised, with change points."""
    p = pal(mode)
    m = pd.read_csv(config.results_dir() / "delhi_monthly_normalised.csv", parse_dates=["date"])
    tr = load_json("parta_trend_delhi.json")
    long = m.melt(id_vars="date", value_vars=["observed", "normalised"], var_name="kind", value_name="pm25")
    long["kind"] = long["kind"].map({"observed": "Observed", "normalised": "Weather removed"})
    lock = alt.Chart(pd.DataFrame({"a": ["2020-03-25"], "b": ["2020-05-31"]})).mark_rect(color=p["band"]).encode(x="a:T", x2="b:T")
    lock_t = alt.Chart(pd.DataFrame({"a": ["2020-03-25"], "t": ["2020 lockdown"]})).mark_text(
        align="left", dx=3, dy=6, baseline="top", color=p["muted"], fontSize=10.5).encode(x="a:T", y=alt.value(0), text="t")
    line = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("pm25:Q", title="Monthly mean PM2.5, µg/m³"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "Weather removed"], range=[p["faint"], p["ink"]]), title=None),
        strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(domain=["Observed", "Weather removed"], range=[[3, 2], [1, 0]]), legend=None),
        strokeWidth=alt.StrokeWidth("kind:N", scale=alt.Scale(domain=["Observed", "Weather removed"], range=[1.2, 2.2]), legend=None),
    )
    layers = [lock, lock_t, line]
    if tr.get("change_points"):
        cp = pd.DataFrame({"date": pd.to_datetime([c + "-01" for c in tr["change_points"]])})
        layers.append(alt.Chart(cp).mark_rule(color=p["slate"], strokeDash=[4, 3], strokeWidth=1.2).encode(x="date:T"))
    return alt.layer(*layers).properties(width=W, height=250)


def fig_event(mode: str):
    """Fig 6. Hour-by-hour excess PM2.5 around Diwali night, CPCB Delhi network, by firecracker regime."""
    p = pal(mode)
    c = pd.read_parquet(config.processed_dir() / "diwali_event_curves.parquet")
    c = c[(c["site"] == "cpcb_delhi") & c["hours_from_diwali_6pm"].between(-24, 42)].copy()
    reg = regimes()
    c["regime"] = c["year"].map(reg)
    c = c[c["regime"].isin(REGIME_ORDER)]
    has = c.groupby("year")["excess"].transform(lambda s: s.notna().sum() >= 12)
    c = c[has].copy()
    c["regime_label"] = c["regime"].map(REGIME_LABEL)
    c["year_s"] = c["year"].astype(str)
    peak = c.loc[c.dropna(subset=["excess"]).groupby("year")["excess"].idxmax()]
    base = alt.Chart(c)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["rule"]).encode(y="y:Q")
    night = alt.Chart(pd.DataFrame({"a": [0], "b": [9]})).mark_rect(color=p["band"]).encode(x="a:Q", x2="b:Q")
    lines = base.mark_line(strokeWidth=1.6).encode(
        x=alt.X("hours_from_diwali_6pm:Q", title="Hours from 6 pm on Diwali",
                scale=alt.Scale(domain=[-24, 42], nice=False), axis=alt.Axis(values=list(range(-24, 43, 6)))),
        y=alt.Y("excess:Q", title="PM2.5 above weather-and-fire baseline, µg/m³"),
        detail="year_s:N",
        color=alt.Color("regime_label:N", scale=_regime_scale(mode), title=None,
                        sort=[REGIME_LABEL[r] for r in REGIME_ORDER]),
        tooltip=["year", "datetime:T", alt.Tooltip("excess:Q", format=".0f")])
    labels = alt.Chart(peak).mark_text(dy=-8, font=FONT_NUM, fontSize=10, color=p["muted"]).encode(
        x="hours_from_diwali_6pm:Q", y="excess:Q", text="year_s:N")
    return (night + zero + lines + labels).properties(width=W, height=290)


def fig_nights(mode: str):
    """Fig 7. Diwali-night excess by year, two Delhi records, against ordinary autumn nights."""
    p = pal(mode)
    t = pd.read_csv(config.results_dir() / "diwali_excess.csv", parse_dates=["date"])
    pl = pd.read_csv(config.results_dir() / "placebo_nights.csv")
    t = t[(t["model"] == "fire") & t["site"].isin(["cpcb_delhi", US_DELHI]) & (t["year"] <= 2025)]
    t = t.dropna(subset=["night_excess_lb"]).copy()
    pld = pl[(pl["site"] == "cpcb_delhi") & (pl["model"] == "fire")]
    lo, hi = pld["night_excess_lb"].quantile([0.05, 0.95])
    band = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(color=p["band"]).encode(y="lo:Q", y2="hi:Q")
    band_t = alt.Chart(pd.DataFrame({"y": [lo], "t": ["Band: 90% of ordinary Oct–Dec nights"]})).mark_text(
        align="right", x=W - 4, dy=-6, color=p["muted"], fontSize=10.5).encode(y="y:Q", text="t")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["rule"]).encode(y="y:Q")
    t["regime_label"] = t["regime"].map(REGIME_LABEL)
    t["year_s"] = t["year"].astype(str)
    t["record"] = t["site"].map({"cpcb_delhi": "CPCB network median", US_DELHI: "US Embassy (gaps filled forward)"})
    x = alt.X("year_s:O", title=None, axis=alt.Axis(labelAngle=0, labelFont=FONT_NUM),
              scale=alt.Scale(domain=[str(y) for y in range(2016, 2026)]))
    us = alt.Chart(t[t["site"] == US_DELHI]).mark_point(filled=False, size=70, strokeWidth=1.5, color=p["muted"]).encode(
        x=x, y="night_excess_lb:Q", tooltip=["year", "record", alt.Tooltip("night_excess_lb:Q", format=".0f", title="excess"),
                                             alt.Tooltip("late_hours_missing:Q", title="late hours missing (of 5)")])
    net = alt.Chart(t[t["site"] == "cpcb_delhi"]).mark_point(filled=True, size=110).encode(
        x=x, y=alt.Y("night_excess_lb:Q", title="Diwali-night excess PM2.5 (6 pm–3 am), µg/m³"),
        color=alt.Color("regime_label:N", scale=_regime_scale(mode), title=None, sort=[REGIME_LABEL[r] for r in REGIME_ORDER]),
        tooltip=["year", "record", "regime_label", alt.Tooltip("night_excess_lb:Q", format=".0f", title="excess"),
                 alt.Tooltip("night_bau:Q", format=".0f", title="baseline")])
    return (band + band_t + zero + us + net).properties(width=W, height=290)


def fig_censoring(mode: str):
    """Fig 8. Share of hours missing at the US Embassy monitor, Diwali nights vs ordinary autumn nights."""
    p = pal(mode)
    b = pd.read_parquet(config.processed_dir() / "bau_hourly_us_delhi.parquet")
    obs = b["observed"]
    rows = []
    dates = diwali_dates()
    for year, d in dates.items():
        if year > 2025:
            continue
        for day in pd.date_range(f"{year}-10-01", f"{year}-12-15", freq="D"):
            diwali = day == d
            if not diwali and d - pd.Timedelta(days=22) <= day <= d + pd.Timedelta(days=16):
                continue
            # same population as the analysis: nights where the monitor was running that evening
            evening = [day + pd.Timedelta(hours=h) for h in range(18, 22)]
            if sum(pd.notna(obs.get(ts, np.nan)) for ts in evening) < 3:
                continue
            for k in range(0, 14):   # 17:00 to 06:00
                ts = day + pd.Timedelta(hours=17 + k)
                rows.append({"kind": "Diwali night" if diwali else "Ordinary Oct–Dec nights", "k": k,
                             "missing": int(pd.isna(obs.get(ts, np.nan)))})
    df = pd.DataFrame(rows).groupby(["kind", "k"])["missing"].mean().reset_index()
    labels = {k: f"{(17 + k) % 24:02d}:00" for k in range(14)}
    df["hour"] = df["k"].map(labels)
    return alt.Chart(df).mark_line(point=alt.OverlayMarkDef(filled=True, size=40)).encode(
        x=alt.X("hour:O", sort=[labels[k] for k in range(14)], title=None, axis=alt.Axis(labelAngle=0, labelFont=FONT_NUM)),
        y=alt.Y("missing:Q", title="Share of hours with no valid reading", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Diwali night", "Ordinary Oct–Dec nights"],
                                                  range=[p["marigold"], p["faint"]]), title=None),
        tooltip=["kind", "hour", alt.Tooltip("missing:Q", format=".0%")],
    ).properties(width=W, height=230)


def fig_cities(mode: str):
    """Fig 8. When does the Diwali excess happen? Five cities, hours around Diwali night."""
    p = pal(mode)
    c = pd.read_parquet(config.processed_dir() / "diwali_event_curves.parquet")
    c = c[c["hours_from_diwali_6pm"].between(-42, 30)]
    c["clock_from_diwali_midnight"] = c["hours_from_diwali_6pm"] - 6 + 0  # hours from 00:00 of Diwali day + 18 - 24
    c["h"] = c["hours_from_diwali_6pm"] + 18  # hours since 00:00 on Diwali day
    prof = c.groupby(["site", "h"])["excess"].median().reset_index()
    names = {"cpcb_delhi": "Delhi (CPCB network)", "us_kolkata": "Kolkata", "us_mumbai": "Mumbai",
             "us_hyderabad": "Hyderabad", "us_chennai": "Chennai"}
    prof["city"] = prof["site"].map(names)
    order = list(names.values())
    prof = prof.dropna(subset=["city"])
    ticks = list(range(-24, 49, 6))
    lbl = "{" + ",".join(f"'{t}':'{['00','06','12','18'][(t % 24)//6]}:00'" for t in ticks) + "}[datum.value]"
    panels = []
    for i, city in enumerate(order):
        d = prof[prof["city"] == city].sort_values("h")
        last = i == len(order) - 1
        x = alt.X("h:Q", title="Clock time; dotted lines mark the midnights before and after Diwali day" if last else None,
                  scale=alt.Scale(domain=[-24, 48], nice=False),
                  axis=alt.Axis(values=ticks, labelExpr=lbl, labels=last, ticks=last))
        day_rules = alt.Chart(pd.DataFrame({"h": [0, 24]})).mark_rule(color=p["faint"], strokeDash=[2, 3]).encode(x=x)
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["rule"]).encode(y="y:Q")
        area = alt.Chart(d).mark_area(opacity=0.35, color=p["marigold"]).encode(
            x=x, y=alt.Y("excess:Q", title=None, axis=alt.Axis(tickCount=3)))
        line = alt.Chart(d).mark_line(color=p["marigold"], strokeWidth=1.8).encode(
            x=x, y="excess:Q", tooltip=[alt.Tooltip("h:Q", title="hours from 00:00 on Diwali"),
                                        alt.Tooltip("excess:Q", format=".0f", title="median excess")])
        panels.append(alt.layer(area, line, zero, day_rules).properties(
            width=W, height=64, title=alt.TitleParams(city, fontSize=12.5, fontWeight=600, color=p["ink"], anchor="start")))
    return alt.vconcat(*panels, spacing=10).resolve_scale(y="independent")


def fig_fingerprint(mode: str):
    """Fig 9. Diwali night vs reference nights at CPCB Delhi sites."""
    p = pal(mode)
    fp = pd.read_csv(config.results_dir() / "fingerprint_delhi_sites.csv")
    specs = [("ratio", "PM2.5 / PM10 ratio"), ("so2", "SO₂, µg/m³"), ("co", "CO, mg/m³"), ("no2", "NO₂, µg/m³")]
    rows = []
    for key, label in specs:
        a, b = f"{key}_diwali", f"{key}_ref"
        if a not in fp:
            continue
        sub = fp[["site", "year", a, b]].dropna()
        for _, r in sub.iterrows():
            rows += [{"panel": label, "id": f"{r.site}{r.year}", "when": "Week before", "value": r[b]},
                     {"panel": label, "id": f"{r.site}{r.year}", "when": "Diwali", "value": r[a]}]
    df = pd.DataFrame(rows)
    med = df.groupby(["panel", "when"])["value"].median().reset_index()
    x = alt.X("when:N", sort=["Week before", "Diwali"], title=None, scale=alt.Scale(padding=0.3),
              axis=alt.Axis(labelAngle=0, labelFont=FONT_LABEL, labelColor=p["ink"]))
    return alt.hconcat(*[
        alt.layer(
            alt.Chart(df[df["panel"] == lab]).mark_line(strokeWidth=0.8, opacity=0.35, color=p["faint"]).encode(
                x=x, y=alt.Y("value:Q", title=lab), detail="id:N"),
            alt.Chart(med[med["panel"] == lab]).mark_line(strokeWidth=3, color=p["marigold"]).encode(x=x, y="value:Q"),
            alt.Chart(med[med["panel"] == lab]).mark_point(filled=True, size=70, color=p["marigold"]).encode(x=x, y="value:Q"),
        ).properties(width=120, height=220)
        for _, lab in specs if lab in set(df["panel"])
    ], spacing=26)


def fig_fires(mode: str):
    """Fig 10. VIIRS fire detections in Punjab and Haryana, October-November, by overpass."""
    p = pal(mode)
    fs = pd.read_csv(config.results_dir() / "fire_season_summary.csv")
    v = fs[fs["product"] == "viirs-snpp"].groupby(["season", "daynight"])["detections_clean"].sum().unstack().fillna(0)
    v = v.rename(columns={"D": "day", "N": "night"}).reset_index()
    v["night_share"] = v["night"] / (v["day"] + v["night"])
    off = pd.DataFrame(config.events()["punjab_fire_counts_official"])
    x = alt.X("season:O", title=None, axis=alt.Axis(labelAngle=0, labelFont=FONT_NUM))
    bars = alt.Chart(v).mark_bar(width=22, color=p["ember"]).encode(
        x=x, y=alt.Y("day:Q", title="Afternoon-pass detections (about 1:30 pm)", axis=alt.Axis(format="~s")),
        tooltip=["season", alt.Tooltip("day:Q", format=","), alt.Tooltip("night:Q", format=",")])
    lbl = alt.Chart(v).mark_text(dy=-6, font=FONT_NUM, fontSize=9.5, color=p["muted"]).encode(
        x=x, y="day:Q", text=alt.Text("day:Q", format=".2~s"))
    offp = alt.Chart(off).mark_point(shape="diamond", size=60, filled=True, color=p["ink"]).encode(
        x="season:O", xOffset=alt.value(20), y="events:Q", tooltip=["season", alt.Tooltip("events:Q", format=","), "source"])
    left = (bars + lbl + offp).properties(width=W // 2 - 20, height=250)
    line = alt.Chart(v).mark_line(color=p["ember"], strokeWidth=2.2, point=alt.OverlayMarkDef(filled=True, size=46, color=p["ember"])).encode(
        x=x, y=alt.Y("night_share:Q", title="Night-pass share of detections (about 1:30 am)", axis=alt.Axis(format="%")),
        tooltip=["season", alt.Tooltip("night_share:Q", format=".2%"), alt.Tooltip("night:Q", format=",")])
    nlbl = alt.Chart(v[v["season"] >= 2023]).mark_text(dx=-4, dy=-10, align="right", font=FONT_NUM, fontSize=9.5, color=p["muted"]).encode(
        x=x, y="night_share:Q", text=alt.Text("night_share:Q", format=".1%"))
    right = (line + nlbl).properties(width=W // 2 - 20, height=250)
    return alt.hconcat(left, right, spacing=44).resolve_scale(y="independent")


def fig_coupling(mode: str):
    """Fig 11. Delhi's weather-adjusted PM2.5 response per unit of the wind-aware fire index, by season."""
    p = pal(mode)
    c = pd.read_csv(config.results_dir() / "fire_coupling_by_season.csv")
    x = alt.X("season:O", title=None, axis=alt.Axis(labelAngle=0, labelFont=FONT_NUM))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=p["rule"]).encode(y="y:Q")
    ci = alt.Chart(c).mark_rule(strokeWidth=2, color=p["ember"]).encode(x=x, y="pct_lo:Q", y2="pct_hi:Q")
    pt = alt.Chart(c).mark_point(filled=True, size=85, color=p["ember"]).encode(
        x=x, y=alt.Y("pct_per_100:Q", title="% rise in Delhi PM2.5 per 100 index units"),
        tooltip=["season", alt.Tooltip("pct_per_100:Q", format=".1f"), "detections", "n_days"])
    return (zero + ci + pt).properties(width=W, height=260)


def fig_evening(mode: str):
    """Fig 12. Evening-to-midday PM2.5 ratio in harvest season vs December, Punjab and Haryana monitors."""
    p = pal(mode)
    e = pd.read_csv(config.results_dir() / "evening_shift_by_site.csv").dropna(subset=["ratio"]).copy()
    rng = np.random.default_rng(7)
    e["xj"] = e["season"] + rng.uniform(-0.18, 0.18, len(e))
    med = e.groupby("season")["ratio"].median().reset_index()
    seasons = list(range(int(e["season"].min()), int(e["season"].max()) + 1))
    xs = alt.Scale(domain=[seasons[0] - 0.6, seasons[-1] + 0.6], nice=False)
    xa = alt.Axis(values=seasons, format="d", labelFont=FONT_NUM, grid=False)
    one = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color=p["rule"], strokeDash=[3, 3]).encode(y="y:Q")
    pts = alt.Chart(e).mark_point(filled=True, size=36, opacity=0.6, color=p["faint"]).encode(
        x=alt.X("xj:Q", title=None, scale=xs, axis=xa),
        y=alt.Y("ratio:Q", title="Harvest evening ratio ÷ December evening ratio", scale=alt.Scale(zero=False)),
        tooltip=["site", "city", "season", alt.Tooltip("ratio:Q", format=".2f")])
    m = alt.Chart(med).mark_tick(thickness=3, size=40, color=p["ember"]).encode(x=alt.X("season:Q", scale=xs), y="ratio:Q")
    return (one + pts + m).properties(width=W, height=250)


def fig_forecast(mode: str):
    """Fig 13. Locked forecast for Diwali 2026 by firecracker regime."""
    p = pal(mode)
    path = config.ROOT / "prereg" / "diwali-2026-forecast.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rows, past = [], []
    for k, v in rec["scenarios"].items():
        rows.append({"scenario": REGIME_LABEL[k], "point": v["point"], "lo": v["interval_80"][0], "hi": v["interval_80"][1]})
        past += [{"scenario": REGIME_LABEL[k], "value": x, "year": str(y)} for x, y in zip(v["past_values"], v["based_on_years"])]
    f, pv = pd.DataFrame(rows), pd.DataFrame(past)
    y = alt.Y("scenario:N", sort=[REGIME_LABEL[r] for r in REGIME_ORDER], title=None, axis=alt.Axis(labelColor=p["ink"], labelFontSize=12))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=p["rule"]).encode(x="x:Q")
    iv = alt.Chart(f).mark_rule(strokeWidth=10, opacity=0.35, color=p["marigold"]).encode(y=y, x=alt.X("lo:Q", title="Diwali-night excess PM2.5, µg/m³"), x2="hi:Q")
    pt = alt.Chart(f).mark_tick(thickness=3, size=22, color=p["marigold"]).encode(y=y, x="point:Q")
    pp = alt.Chart(pv).mark_point(filled=False, size=50, color=p["ink"], strokeWidth=1.3).encode(
        y=y, x="value:Q", tooltip=["scenario", "year", "value"])
    pl = alt.Chart(pv).mark_text(dy=-11, font=FONT_NUM, fontSize=9.5, color=p["muted"]).encode(y=y, x="value:Q", text="year")
    return (zero + iv + pt + pp + pl).properties(width=W - 80, height=alt.Step(52))


FIGURES = {
    "fig01_design": fig_design, "fig02_coverage": fig_coverage, "fig03_tape": fig_tape,
    "fig04_gap": fig_gap, "fig05_trend": fig_trend, "fig06_event": fig_event, "fig07_nights": fig_nights,
    "fig08_censoring": fig_censoring, "fig09_cities": fig_cities, "fig10_fingerprint": fig_fingerprint,
    "fig11_fires": fig_fires, "fig12_coupling": fig_coupling, "fig13_evening": fig_evening,
    "fig14_forecast": fig_forecast,
}


def _configured(chart, mode: str, web: bool):
    cfg = vl_config(mode, web)
    if cfg["background"] is None:
        cfg["background"] = "transparent"
    return chart.configure(**cfg)


def render(name: str, fn) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        spec = _configured(fn(mode), mode, web=True).to_dict()
        (WEB_DIR / f"{name}.{mode}.json").write_text(json.dumps(spec), encoding="utf-8")
    spec = _configured(fn("light"), "light", web=False).to_dict()
    (PAPER_DIR / f"{name}.pdf").write_bytes(vlc.vegalite_to_pdf(spec))
    (PAPER_DIR / f"{name}.png").write_bytes(vlc.vegalite_to_png(spec, scale=2.5))


def run(only: list[str] | None = None) -> None:
    vlc.register_font_directory(str(FONT_DIR))
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        try:
            render(name, fn)
            print(f"  {name}: ok", flush=True)
        except FileNotFoundError as exc:
            print(f"  {name}: skipped, missing input ({exc.filename})", flush=True)
