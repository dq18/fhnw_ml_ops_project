"""
Feature Pipeline — containerized version.

Creates/updates three Feature Groups in Hopsworks:
  A. crag_static   — static crag attributes from data/crags.json
  B. weather_daily — daily weather + rolling aggregates per crag
  C. climb_logs    — ascent logs (climbable label) from data/climb_logs.csv

Run:
    python -m src.pipelines.feature_pipeline
"""

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import hopsworks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# hsfs Kafka engine uses a hardcoded /tmp path for SSL certs.
# On Windows this resolves to \tmp on the current drive — create it if missing.
Path("/tmp").mkdir(exist_ok=True)

from src.config import (
    CRAG_FG_NAME, CRAG_FG_VERSION,
    WEATHER_FG_NAME, WEATHER_FG_VERSION,
    CLIMB_LOGS_FG_NAME, CLIMB_LOGS_FG_VERSION,
    CLIMB_LOGS_PATH,
    ARCHIVE_START_DATE, ARCHIVE_END_DATE,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
)
from src.features.crag_features import load_raw_crags, prepare_crag_df
from src.features.weather_features import add_rolling_features
from src.weather_client import fetch_archive_daily, fetch_recent_daily


def run_feature_pipeline() -> None:
    """Execute the full feature pipeline."""
    print(f"Archive window: {ARCHIVE_START_DATE} → {ARCHIVE_END_DATE}")

    # ── A: Load & prepare crag data (from crags.json) ────────────────────
    raw_crags = load_raw_crags()
    crag_df = prepare_crag_df(raw_crags)
    print(f"Crags loaded: {len(crag_df)} rows")

    # ── B: Connect to Hopsworks ──────────────────────────────────────────
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )
    fs = project.get_feature_store()
    print(f"Connected to project: {project.name}")

    # ── C: Create & populate crag_static FG ──────────────────────────────
    crag_fg = fs.get_or_create_feature_group(
        name=CRAG_FG_NAME,
        version=CRAG_FG_VERSION,
        description="Static crag attributes: location, rock type, orientation, seasons",
        primary_key=["crag_id"],
        online_enabled=True,
    )
    crag_fg.insert(crag_df)
    print(f"Inserted {len(crag_df)} rows into {CRAG_FG_NAME}")

    # ── D: Fetch historical weather for all crags ────────────────────────
    all_weather = []
    for _, crag in crag_df.iterrows():
        crag_id = crag["crag_id"]
        lat, lon = crag["latitude"], crag["longitude"]
        print(f"  Fetching weather for crag {crag_id} ({crag['name']})...", end=" ")

        weather_df = fetch_archive_daily(lat, lon, ARCHIVE_START_DATE, ARCHIVE_END_DATE)
        time.sleep(7)  # respect Open-Meteo free-tier minutely rate limit
        recent_df = fetch_recent_daily(lat, lon, days=14)

        combined = pd.concat([weather_df, recent_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        combined["crag_id"] = crag_id

        all_weather.append(combined)
        print(f"{len(combined)} days")
        time.sleep(7)  # respect Open-Meteo free-tier minutely rate limit

    weather_all_df = pd.concat(all_weather, ignore_index=True)
    print(f"Total weather rows: {len(weather_all_df)}")

    # ── E: Compute rolling features ──────────────────────────────────────
    weather_featured = []
    for crag_id, group in weather_all_df.groupby("crag_id"):
        featured = add_rolling_features(group)
        weather_featured.append(featured)

    weather_featured_df = pd.concat(weather_featured, ignore_index=True)
    print(f"Weather featured rows: {len(weather_featured_df)}")

    # ── F: Create & populate weather_daily FG ────────────────────────────
    weather_featured_df["date"] = pd.to_datetime(weather_featured_df["date"])

    weather_fg = fs.get_or_create_feature_group(
        name=WEATHER_FG_NAME,
        version=WEATHER_FG_VERSION,
        description="Daily weather features + rolling aggregates per crag",
        primary_key=["crag_id", "date"],
        event_time="date",
        online_enabled=True,
    )
    weather_fg.insert(weather_featured_df)
    print(f"Inserted {len(weather_featured_df)} rows into {WEATHER_FG_NAME}")

    # ── G: Load & populate climb_logs FG ─────────────────────────────────
    logs_df = pd.read_csv(CLIMB_LOGS_PATH)
    logs_df["date"] = pd.to_datetime(logs_df["date"])
    logs_df["crag_id"] = logs_df["crag_id"].astype(int)

    logs_fg = fs.get_or_create_feature_group(
        name=CLIMB_LOGS_FG_NAME,
        version=CLIMB_LOGS_FG_VERSION,
        description="Climb logs: ascents logged per crag per day (label source)",
        primary_key=["crag_id", "date"],
        event_time="date",
        online_enabled=True,
    )
    logs_fg.insert(logs_df)
    print(f"Inserted {len(logs_df)} rows into {CLIMB_LOGS_FG_NAME}")

    print("\n✓ Feature pipeline complete.")


if __name__ == "__main__":
    run_feature_pipeline()
