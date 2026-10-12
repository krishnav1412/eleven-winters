"""Build the paper: Jinja2 template -> Markdown -> HTML (+ PDF via headless Edge/Chrome) and LaTeX.

The LaTeX file is for journal submission (compile on Overleaf or any TeX install);
the PDF is the preprint and the course submission.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

import pypandoc
from jinja2 import Environment, FileSystemLoader

from elevenwinters import config, report

PAPER = config.ROOT / "paper"
REPO_URL = os.environ.get("EW_REPO_URL", "https://github.com/OWNER/eleven-winters")
SITE_URL = os.environ.get("EW_SITE_URL", "https://OWNER.github.io/eleven-winters/")
AFFILIATION = os.environ.get("EW_AFFILIATION", "Independent researcher")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/chromium", "/usr/bin/google-chrome",
]

FONTS_LINK = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
              "&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;"
              "1,8..60,200..900&display=swap")


def render_markdown() -> Path:
    env = Environment(loader=FileSystemLoader(PAPER), trim_blocks=False, lstrip_blocks=False)
    env.filters["fmt"] = report.fmt
    md = env.get_template("paper.md.j2").render(
        n=report.numbers(), figs=report.figure_meta(), build_date=dt.date.today().strftime("%d %B %Y"),
        repo_url=REPO_URL, site_url=SITE_URL, affiliation=AFFILIATION)
    out = PAPER / "paper.md"
    out.write_text(md, encoding="utf-8")
    return out


def to_html(md: Path) -> Path:
    out = PAPER / "paper.html"
    header = PAPER / "_head.html"
    header.write_text(f'<link rel="stylesheet" href="{FONTS_LINK}">\n', encoding="utf-8")
    pypandoc.convert_file(
        str(md), "html5", outputfile=str(out),
        extra_args=["--standalone", "--citeproc", "--css=paper.css", f"--include-in-header={header}",
                    "--metadata=lang:en-GB", f"--resource-path={PAPER}"],
        cworkdir=str(PAPER))
    return out


def to_latex(md: Path) -> Path:
    out = PAPER / "paper.tex"
    text = md.read_text(encoding="utf-8").replace(".png)", ".pdf)")
    tmp = PAPER / "_paper_tex.md"
    tmp.write_text(text, encoding="utf-8")
    pypandoc.convert_file(str(tmp), "latex", outputfile=str(out),
                          extra_args=["--standalone", "--citeproc", "-V", "geometry:margin=2.2cm",
                                      "-V", "fontsize=11pt"], cworkdir=str(PAPER))
    tmp.unlink()
    return out


def to_pdf(html: Path) -> Path | None:
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chromium")
    if not browser:
        print("  no Edge/Chrome found; skipping PDF")
        return None
    out = PAPER / "eleven-winters.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=20000", f"--print-to-pdf={out}", html.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    return out


def build() -> None:
    md = render_markdown()
    html = to_html(md)
    tex = to_latex(md)
    pdf = to_pdf(html)
    print(f"  paper: {md.name}, {html.name}, {tex.name}" + (f", {pdf.name} ({pdf.stat().st_size // 1024} KB)" if pdf else ""))
