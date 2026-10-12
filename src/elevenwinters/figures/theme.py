"""Visual system shared by the paper and the website.

Colour carries one meaning each, everywhere:
  soot greys  PM2.5 concentration, keyed to India's National AQI breakpoints
  ember       fire detections and the burning season
  marigold    Diwali (a ramp of marigold encodes the firecracker regime)
  slate       weather
  ink         weather-adjusted values; muted grey for observed values
"""

from __future__ import annotations

FONT_LABEL = "IBM Plex Sans Condensed"
FONT_NUM = "IBM Plex Mono"

# 24-hour PM2.5 NAQI breakpoints (ug/m3) and category names
NAQI = [(0, 30, "Good"), (31, 60, "Satisfactory"), (61, 90, "Moderate"), (91, 120, "Poor"),
        (121, 250, "Very poor"), (251, 10_000, "Severe")]
SOOT = ["#E6E8E3", "#C9CCC5", "#A3A69F", "#7C7F78", "#4E514B", "#1F211D"]
TAPE = "#FCFCFA"

PALETTES = {
    "light": {
        "ink": "#1B1F1D", "muted": "#59605C", "faint": "#8A918C", "rule": "#CBD1CB", "grid": "#E3E7E2",
        "ember": "#A9491B", "ember_soft": "#E2B69E", "marigold": "#B88A12", "slate": "#476270",
        "band": "#E9ECE7",
        "regime": {"permissive": "#6B4F05", "green_window": "#B88A12", "full_ban": "#D8BD70"},
    },
    "dark": {
        "ink": "#E3E6E1", "muted": "#9BA29D", "faint": "#6F7772", "rule": "#2E3532", "grid": "#252B28",
        "ember": "#E07A45", "ember_soft": "#7A452B", "marigold": "#DDB13E", "slate": "#8FA8B4",
        "band": "#1F2421",
        "regime": {"permissive": "#F4D27A", "green_window": "#C9A23A", "full_ban": "#7D6A31"},
    },
}

REGIME_LABEL = {"permissive": "No effective restriction", "green_window": "Green crackers, fixed hours",
                "full_ban": "Full ban", "unknown": "Not announced"}


def pal(mode: str) -> dict:
    return PALETTES[mode]


def vl_config(mode: str, web: bool) -> dict:
    p = pal(mode)
    return {
        "background": None if web else "#FFFFFF",
        "padding": 6,
        "font": FONT_LABEL,
        "view": {"stroke": None, "continuousWidth": 640, "continuousHeight": 260},
        "axis": {
            "labelFont": FONT_NUM, "labelFontSize": 10.5, "labelColor": p["muted"],
            "titleFont": FONT_LABEL, "titleFontSize": 12, "titleFontWeight": 500, "titleColor": p["muted"],
            "domainColor": p["rule"], "tickColor": p["rule"], "gridColor": p["grid"],
            "labelPadding": 4, "titlePadding": 8, "tickSize": 4,
        },
        "axisX": {"grid": False},
        "axisY": {"domain": False, "ticks": False, "grid": True},
        "axisBand": {"grid": False},
        "legend": {
            "labelFont": FONT_LABEL, "labelFontSize": 12, "labelColor": p["ink"],
            "titleFont": FONT_LABEL, "titleFontSize": 11, "titleColor": p["muted"], "titleFontWeight": 500,
            "orient": "top", "direction": "horizontal", "symbolStrokeWidth": 2.2, "columnPadding": 14,
        },
        "header": {
            "labelFont": FONT_LABEL, "labelFontSize": 12.5, "labelFontWeight": 600, "labelColor": p["ink"],
            "labelAnchor": "start", "labelPadding": 6, "titleFontSize": 0, "title": None,
        },
        "text": {"font": FONT_LABEL, "fontSize": 11.5, "color": p["ink"]},
        "line": {"strokeWidth": 1.8, "strokeCap": "round"},
        "point": {"filled": True, "size": 42},
        "rule": {"color": p["rule"]},
        "title": {"font": FONT_LABEL, "fontSize": 13, "color": p["ink"], "anchor": "start"},
        "concat": {"spacing": 18},
        "facet": {"spacing": 16},
    }
