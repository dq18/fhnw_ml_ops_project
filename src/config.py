"""
Centralized configuration for the MLOps Crag Climbability project.
All Feature Group names, Feature View names, model names, and label thresholds
are defined here as the single source of truth.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Hopsworks ────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CRAG_JSON_PATH = DATA_DIR / "crags.json"
CLIMB_LOGS_PATH = DATA_DIR / "climb_logs.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# ── Feature Group names & versions ──────────────────────────────────────────
CRAG_FG_NAME = "crag_static"
CRAG_FG_VERSION = 1
WEATHER_FG_NAME = "weather_daily"
WEATHER_FG_VERSION = 1
CLIMB_LOGS_FG_NAME = "climb_logs"
CLIMB_LOGS_FG_VERSION = 1

# ── Feature View ─────────────────────────────────────────────────────────────
FEATURE_VIEW_NAME = "crag_climbability"
FEATURE_VIEW_VERSION = 2

# ── Model Registry ───────────────────────────────────────────────────────────
MODEL_NAME = "crag_climbability_model"
# MODEL_VERSION is no longer hard-coded; training auto-increments and
# inference loads by the "production" tag.  Keep a default for backward compat.
MODEL_VERSION = None  # None → auto-increment on save / load production tag
MODEL_LOCAL_PATH = MODELS_DIR / "crag_classifier.joblib"

# ── Historical data window ───────────────────────────────────────────────────
ARCHIVE_YEARS = 5
ARCHIVE_START_DATE = (datetime.now() - timedelta(days=ARCHIVE_YEARS * 365)).strftime(
    "%Y-%m-%d"
)
# Open-Meteo archive has ~5-day lag; stop a week before today to be safe
ARCHIVE_END_DATE = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

# ── Label thresholds ────────────────────────────────────────────────────────
# A crag is considered "climbable" (label=1) when ALL conditions are met:
LABEL_RAIN_3D_MAX = 15.0  # mm  — cumulative rain over last 3 days
LABEL_PRECIP_TODAY_MAX = 5.0  # mm  — precipitation today
LABEL_WIND_3D_AVG_MAX = 50.0  # km/h — average max wind last 3 days
LABEL_TEMP_MIN = 3.0  # °C  — daily max temperature
LABEL_TEMP_MAX = 35.0  # °C  — daily max temperature
# Additionally: the current month must be in the crag's climbing seasons.

# ── Season month mapping ────────────────────────────────────────────────────
SEASON_MONTHS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}

# ── Open-Meteo daily variables to fetch ─────────────────────────────────────
ARCHIVE_DAILY_VARIABLES = [
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
]

FORECAST_CURRENT_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
]

# ── Orientations  ──────────────────────────────────────────────────────────
ORIENTATIONS = [
    "north",
    "north east",
    "east",
    "south east",
    "south",
    "south west",
    "west",
    "north west",
]

SEASONS = ["spring", "summer", "autumn", "winter"]
