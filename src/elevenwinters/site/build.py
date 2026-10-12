"""Build the static companion website into docs/ (served by GitHub Pages from /docs).

Pages are Jinja2 templates filled with the same numbers the paper uses (report.numbers()).
Charts are the Vega-Lite specs written by figures/build.py.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from elevenwinters import config, report
from elevenwinters.figures.theme import pal, vl_config

TEMPLATES = config.ROOT / "src" / "elevenwinters" / "site" / "templates"
STATIC = config.ROOT / "site" / "static"
OUT = config.ROOT / "docs"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _copy_static() -> None:
    for sub in ("css", "js", "figs"):
        src = STATIC / sub
        if src.exists():
            shutil.copytree(src, OUT / sub, dirs_exist_ok=True)
    for mode in ("light", "dark"):
        p = pal(mode)
        flat = {k: v for k, v in p.items() if isinstance(v, str)}
        cfg = vl_config(mode, web=True)
        cfg["background"] = "transparent"
        (OUT / "figs" / f"_theme.{mode}.json").write_text(json.dumps({"config": cfg, "palette": flat}), encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")


def _explorer_data() -> int:
    dw = pd.read_parquet(config.processed_dir() / "deweathered_daily.parquet")
    skill = pd.read_csv(config.results_dir() / "model_skill_daily.csv").set_index("site")
    sites = pd.read_csv(config.results_dir() / "sites.csv").set_index("site")
    out_dir = OUT / "data" / "sites"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for site, g in dw.groupby("site"):
        g = g.set_index("date").sort_index()
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        roll = full[["observed", "normalised"]].rolling(30, center=True, min_periods=15).mean()
        keep = roll.dropna(how="all")
        keep = keep.iloc[::3]  # every third day is plenty for a 30-day mean, and keeps files small
        mon = g[["observed", "normalised"]].groupby(g.index.month).mean()
        monthly = [{"month": MONTHS[m - 1], "kind": k, "pm25": round(float(mon.loc[m, c]), 1)}
                   for m in mon.index for k, c in [("Observed", "observed"), ("Weather removed", "normalised")]]

        def r(a):
            return [None if not np.isfinite(x) else round(float(x), 1) for x in a]

        payload = {"site": site, "date": keep.index.strftime("%Y-%m-%d").tolist(),
                   "observed_30d": r(keep["observed"].values), "normalised_30d": r(keep["normalised"].values),
                   "monthly": monthly}
        (out_dir / f"{site}.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        meta = sites.loc[site]
        index.append({"site": site, "name": str(meta["name"]).split(",")[0] if meta["network"] == "cpcb" else "US Embassy/Consulate",
                      "city": meta["city"], "network": meta["network"], "days": int(len(g)),
                      "first": g.index.min().strftime("%b %Y"), "last": g.index.max().strftime("%b %Y"),
                      "r2": f"{skill.loc[site, 'r2_cv']:.2f}" if site in skill.index else "n/a"})
    index.sort(key=lambda s: (s["city"] != "Delhi", s["city"], s["network"] != "us_mission", s["name"]))
    (OUT / "data" / "sites.json").write_text(json.dumps(index), encoding="utf-8")
    return len(index)


def run() -> None:
    OUT.mkdir(exist_ok=True)
    _copy_static()
    n_sites = _explorer_data()
    n = report.numbers()
    figs = report.figure_meta()
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["fmt"] = report.fmt
    pdf = config.ROOT / "paper" / "eleven-winters.pdf"
    has_pdf = pdf.exists()
    if has_pdf:
        (OUT / "paper").mkdir(exist_ok=True)
        shutil.copy2(pdf, OUT / "paper" / "eleven-winters.pdf")
    for page in ["index", "explorer", "methods", "forecast"]:
        html = env.get_template(f"{page}.html").render(n=n, figs=figs, page=page, has_pdf=has_pdf, n_sites=n_sites)
        (OUT / f"{page}.html").write_text(html, encoding="utf-8")
    print(f"  site built: 4 pages, {n_sites} explorer sites, {len(figs)} figures")
