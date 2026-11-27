"""Settings from environment variables or a local .env file. Never commit .env."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _load_env():
    p = ROOT / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
WEATHER_TTL_S = int(os.environ.get("WEATHER_TTL_S", 3 * 3600))
WEATHER_RETRY_S = int(os.environ.get("WEATHER_RETRY_S", 300))  # back-off after a failed Open-Meteo refresh
CHAT_RATE_PER_MIN = int(os.environ.get("CHAT_RATE_PER_MIN", 10))
CHAT_MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", 300))
CHAT_TIMEOUT_S = float(os.environ.get("CHAT_TIMEOUT_S", 20))
CHAT_CACHE_TTL_S = int(os.environ.get("CHAT_CACHE_TTL_S", 300))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
HORIZONS = (2, 3, 4)
DATASET_PATH = ROOT / "data" / "dengue_history.csv"
HISTORICAL_END = pd.Timestamp(pd.read_csv(DATASET_PATH, usecols=["date"])["date"].max()).date()

# Phase 2 operations engine (all local; no network needed)
ONSITE_DATA_PATH = ROOT / "data" / "onsite_dengue_clean.csv"
CASE_REPORTS_PATH = ROOT / "data" / "case_reports" / "reports.csv"
BED_CAPACITY_PATH = ROOT / "data" / "bed_capacity.csv"
BED_CAPACITY_QUANTILE = float(os.environ.get("BED_CAPACITY_QUANTILE", 0.95))
LARVICIDE_EFFECT_PER_CREW = float(os.environ.get("LARVICIDE_EFFECT_PER_CREW", 0.15))
FOGGING_EFFECT_PER_CREW = float(os.environ.get("FOGGING_EFFECT_PER_CREW", 0.10))
MAX_CREWS_PER_WARD = int(os.environ.get("MAX_CREWS_PER_WARD", 5))
CASES_PER_CREW = float(os.environ.get("CASES_PER_CREW", 4))
EXPORTS_DIR = ROOT / "exports"
