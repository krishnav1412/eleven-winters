"""Command line entry point: `ew <stage>`. `ew all` runs every stage after the downloads."""

from __future__ import annotations

import argparse
import importlib
import time

# stage name -> (module, function, help)
STAGES = {
    "stations": ("elevenwinters.ingest.openaq", "select_stations", "pick monitors for each city"),
    "openaq": ("elevenwinters.ingest.openaq", "download_all", "download hourly station data"),
    "firms": ("elevenwinters.ingest.firms", "download_all", "download fire detections"),
    "meteo": ("elevenwinters.ingest.meteo", "download_all", "download hourly ERA5 weather"),
    "firms-subset": ("elevenwinters.ingest.firms", "subset_northwest", "clip fires to NW India"),
    "panel": ("elevenwinters.panel", "build_all", "clean and merge station and weather data"),
    "fires": ("elevenwinters.fires", "build_all", "fire summaries and the fire index"),
    "parta": ("elevenwinters.analysis.parta", "run", "weather normalisation, seasonal gap, trend"),
    "partb": ("elevenwinters.analysis.partb", "run", "Diwali excess, placebo, fingerprint"),
    "partc": ("elevenwinters.analysis.partc", "run", "fire lag, coupling, evening shift"),
    "forecast": ("elevenwinters.analysis.forecast", "lock", "lock the Diwali 2026 forecast"),
    "figures": ("elevenwinters.figures.build", "run", "render every figure"),
    "numbers": ("elevenwinters.report", "write_numbers", "write results/numbers.json"),
    "paper": ("elevenwinters.paper", "build", "render the paper to HTML, PDF and LaTeX"),
    "site": ("elevenwinters.site.build", "run", "build the static website into docs/"),
}

DOWNLOADS = ["stations", "openaq", "firms", "meteo"]
# `forecast` is deliberately not in the pipeline: the locked record must not be regenerated.
PIPELINE = ["firms-subset", "panel", "fires", "parta", "partb", "partc", "figures", "numbers", "paper", "site"]


def run(stage: str) -> None:
    module, func, _ = STAGES[stage]
    t0 = time.time()
    print(f"== {stage}", flush=True)
    getattr(importlib.import_module(module), func)()
    print(f"== {stage} done in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ew", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all", "download"], help="stage to run")
    args = parser.parse_args()
    if args.stage == "all":
        stages = PIPELINE
    elif args.stage == "download":
        stages = DOWNLOADS
    else:
        stages = [args.stage]
    for stage in stages:
        run(stage)


if __name__ == "__main__":
    main()
