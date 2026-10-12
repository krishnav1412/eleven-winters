"""Part D: a pre-registered forecast for Diwali 2026 (Sunday 8 November).

Target: the Diwali-night excess in the CPCB Delhi network median (hours with 3+ reporting
sites), i.e. mean(observed - BAU) from 18:00 on 8 Nov to 02:59 on 9 Nov 2026 with missing
hours carried forward, computed with exactly the code in partb.py (`night_excess_lb`).
Because the firecracker rules for 2026 were not announced when the forecast was locked,
we publish one interval per regime. The interval is a standard prediction interval from
past Diwalis under the same regime, with a variance pooled across regimes.

The locked record is written to prereg/ with a SHA-256 checksum, committed and tagged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from elevenwinters import config

PREREG = config.ROOT / "prereg"
LEVEL = 0.80
TARGET_SITE = "cpcb_delhi"
TARGET_COL = "night_excess_lb"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return None


def build_forecast() -> dict:
    t = pd.read_csv(config.results_dir() / "diwali_excess.csv", parse_dates=["date"])
    dl = t[(t["site"] == TARGET_SITE) & (t["model"] == "fire")].dropna(subset=[TARGET_COL])
    dl = dl[dl["regime"].isin(["permissive", "green_window", "full_ban"])]
    groups = {r: g[TARGET_COL].to_numpy() for r, g in dl.groupby("regime")}
    resid = np.concatenate([v - v.mean() for v in groups.values()])
    dof = len(resid) - len(groups)
    s_pooled = float(np.sqrt(np.sum(resid ** 2) / dof))
    tcrit = float(stats.t.ppf(0.5 + LEVEL / 2, dof))
    scenarios = {}
    for regime, v in groups.items():
        half = tcrit * s_pooled * np.sqrt(1 + 1 / len(v))
        scenarios[regime] = {
            "based_on_years": dl[dl["regime"] == regime]["year"].tolist(),
            "past_values": [round(float(x), 1) for x in v],
            "point": round(float(v.mean()), 1),
            "interval_80": [round(float(v.mean() - half), 1), round(float(v.mean() + half), 1)],
        }
    return {
        "title": "Pre-registered forecast: Delhi Diwali-night PM2.5 excess, 8 November 2026",
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": _git_commit(),
        "target": {
            "site": "CPCB Delhi network median, hours with 3 or more reporting sites (config/stations.csv)",
            "quantity": "mean hourly (observed - BAU) PM2.5, 18:00 8 Nov to 02:59 9 Nov 2026 IST, "
                        "missing hours carried forward from the last valid hour",
            "units": "ug/m3",
            "bau_model": "LightGBM on ERA5 weather, calendar and wind-aware fire index; Diwali windows excluded "
                         "from training; out-of-fold prediction by ISO week (src/elevenwinters/analysis/partb.py)",
        },
        "scenarios": scenarios,
        "interval_method": f"{int(LEVEL * 100)}% prediction interval, mean of past Diwalis under the regime "
                           f"+/- t({dof} df) x pooled SD ({s_pooled:.1f}) x sqrt(1 + 1/n)",
        "scoring": "After 8 Nov 2026 the same code is rerun with data through 15 Nov 2026. The forecast for the "
                   "regime actually in force counts as correct if the observed excess falls inside its interval.",
    }


def lock(force: bool = False) -> dict:
    PREREG.mkdir(exist_ok=True)
    path = PREREG / "diwali-2026-forecast.json"
    if path.exists() and not force:
        print("  forecast already locked; refusing to overwrite (pass force=True only before publication)")
        return json.loads(path.read_text(encoding="utf-8"))
    rec = build_forecast()
    # write exact bytes (LF line endings) so the checksum matches the copy in git on every OS
    payload = (json.dumps(rec, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (PREREG / "SHA256SUMS").write_bytes(f"{digest}  diwali-2026-forecast.json\n".encode())
    lines = [
        "# Diwali 2026 forecast (pre-registered)",
        "",
        f"Locked {rec['locked_at_utc']}. SHA-256 of the JSON record: `{digest}`.",
        "",
        f"Target: {rec['target']['quantity']} at the {rec['target']['site']}.",
        "",
        "| Firecracker rules in 2026 | Forecast (µg/m³) | 80% interval | Based on |",
        "|---|---|---|---|",
    ]
    names = {"full_ban": "Full ban", "green_window": "Green crackers in fixed hours", "permissive": "No effective restriction"}
    for k, v in rec["scenarios"].items():
        lines.append(f"| {names.get(k, k)} | {v['point']} | {v['interval_80'][0]} to {v['interval_80'][1]} "
                     f"| {', '.join(map(str, v['based_on_years']))} |")
    lines += ["", rec["interval_method"], "", "Scoring: " + rec["scoring"], "",
              "Verify: `sha256sum -c SHA256SUMS` (Linux/macOS) or "
              "`Get-FileHash diwali-2026-forecast.json` (Windows PowerShell).", ""]
    (PREREG / "README.md").write_bytes("\n".join(lines).encode("utf-8"))
    print(f"  forecast locked, sha256 {digest[:16]}...")
    return rec
