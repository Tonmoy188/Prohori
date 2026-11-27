"""DengueWatch configuration.

Core forecasting/allocation/routing/dispatch needs none of these values; only the
optional chat assistant and CORS read them. Importing this module must never crash.
"""

import csv
import os
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_env():
    env_file = ROOT / ".env"

    if not env_file.is_file():
        return

    # utf-8-sig: Windows editors often add a BOM, which would corrupt the first key
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def _num(name, default, cast=int, minimum=None):
    """Read a numeric env var; fall back to the default on bad/out-of-range input."""
    try:
        value = cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return default if minimum is not None and value < minimum else value


_load_env()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = (
    os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    or "gemini-2.5-flash"
)

WEATHER_TTL_S = _num("WEATHER_TTL_S", 10800, minimum=0)
WEATHER_RETRY_S = _num("WEATHER_RETRY_S", 300, minimum=0)

CHAT_RATE_PER_MIN = _num("CHAT_RATE_PER_MIN", 10, minimum=1)
CHAT_MAX_TOKENS = _num("CHAT_MAX_TOKENS", 300, minimum=1)
CHAT_TIMEOUT_S = _num("CHAT_TIMEOUT_S", 20, float, minimum=1)
CHAT_CACHE_TTL_S = _num("CHAT_CACHE_TTL_S", 300, minimum=0)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]

HORIZONS = (2, 3, 4)

DATASET_PATH = ROOT / "data" / "onsite_dengue_clean.csv"


def _dataset_end():
    """Latest dataset date, or None if the file is missing/unreadable (never raises)."""
    try:
        with DATASET_PATH.open(encoding="utf-8-sig", newline="") as f:
            return max(
                date.fromisoformat(row["date"][:10]) for row in csv.DictReader(f)
            )
    except (OSError, KeyError, ValueError):
        return None


HISTORICAL_END = _dataset_end()
