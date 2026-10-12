"""Paths, settings and secrets. Everything else imports from here."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


def data_dir() -> Path:
    raw = os.environ.get("EW_DATA_DIR", "~/eleven-winters-data")
    path = Path(os.path.expanduser(raw))
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(*parts: str) -> Path:
    path = data_dir().joinpath("raw", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def interim_dir(*parts: str) -> Path:
    path = data_dir().joinpath("interim", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(*parts: str) -> Path:
    path = data_dir().joinpath("processed", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(*parts: str) -> Path:
    """Small analysis outputs (tables, numbers) that are committed to the repo."""
    path = ROOT.joinpath("results", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def openaq_key() -> str:
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAQ_API_KEY is missing. Add it to .env (see .env.example).")
    return key


@lru_cache
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def settings() -> dict:
    return load_yaml("settings.yaml")


def cities() -> list[dict]:
    return load_yaml("cities.yaml")["cities"]


def events() -> dict:
    return load_yaml("events.yaml")
