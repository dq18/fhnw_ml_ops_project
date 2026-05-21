"""
Inference Pipeline — containerized version.

Steps:
  A. Connect to Hopsworks, download the trained model
  B. Fetch current weather (RT features) for each crag
  C. Get latest batch features from offline store
  D. Merge RT features + batch features
  E. Run predictions
  F. Save predictions to predictions/{today}.csv

Run:
    python -m src.pipelines.inference_pipeline
"""

import sys
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import hopsworks
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# hsfs Kafka engine uses a hardcoded /tmp path for SSL certs.
# On Windows this resolves to \tmp on the current drive — create it if missing.
Path("/tmp").mkdir(exist_ok=True)

from src.config import (
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    MODEL_NAME, MODEL_VERSION,
    PREDICTIONS_DIR,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
)
from src.weather_client import fetch_forecast_current
from src.features.crag_features import prepare_crag_df


def run_inference_pipeline() -> None:
    """Execute the full inference pipeline."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    print(f"Inference date: {today}")

    # ── A: Connect to Hopsworks, get Feature View & model ────────────────
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )
    fs = project.get_feature_store()
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)

    mr = project.get_model_registry()
    model_hw = mr.get_model(name=MODEL_NAME, version=MODEL_VERSION)
    model_dir = model_hw.download()
    model_pipeline = joblib.load(os.path.join(model_dir, "crag_classifier.joblib"))
    print(f"Model loaded from: {model_dir}")

    # ── B: Fetch current weather (RT features) ───────────────────────────
    crag_df = prepare_crag_df()
    rt_features = []

    for _, crag in crag_df.iterrows():
        crag_id = crag["crag_id"]
        print(f"  Fetching RT weather for {crag['name']}...", end=" ")
        current = fetch_forecast_current(crag["latitude"], crag["longitude"])
        current["crag_id"] = crag_id
        rt_features.append(current)
        print(f"temp={current.get('temperature_2m', '?')}°C")
        time.sleep(0.2)

    rt_df = pd.DataFrame(rt_features)
    print(f"RT features for {len(rt_df)} crags")

    # ── C: Get latest features from offline store ────────────────────────
    batch_df = fv.get_batch_data()
    batch_df["date"] = pd.to_datetime(batch_df["date"])
    latest_df = (
        batch_df
        .sort_values("date")
        .groupby("crag_id")
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"Latest batch features: {len(latest_df)} crags")

    # ── D: Merge RT + batch features ────────────────────────────────────
    rt_renamed = rt_df.rename(columns={
        "temperature_2m": "rt_temperature",
        "wind_speed_10m": "rt_wind_speed",
        "cloud_cover": "rt_cloud_cover",
        "precipitation": "rt_precipitation",
    })
    inference_df = pd.merge(latest_df, rt_renamed, on="crag_id", how="left")

    # ── E: Run predictions ───────────────────────────────────────────────
    drop_cols = [c for c in ["date", "crag_id"] if c in inference_df.columns]
    X_inference = inference_df.drop(columns=drop_cols, errors="ignore")

    if hasattr(model_pipeline, "feature_names_in_"):
        X_inference = X_inference.reindex(columns=model_pipeline.feature_names_in_, fill_value=0)

    predictions = model_pipeline.predict(X_inference)
    try:
        probabilities = model_pipeline.predict_proba(X_inference)[:, 1]
    except Exception:
        probabilities = predictions.astype(float)

    # ── F: Build & save results ──────────────────────────────────────────
    crag_names = crag_df.set_index("crag_id")["name"]
    results = pd.DataFrame({
        "crag_id": inference_df["crag_id"],
        "crag_name": inference_df["crag_id"].map(crag_names),
        "prediction": predictions,
        "probability": probabilities,
        "rt_temp": inference_df.get("rt_temperature"),
        "rt_precip": inference_df.get("rt_precipitation"),
        "rt_wind": inference_df.get("rt_wind_speed"),
        "rain_3d_sum": inference_df.get("rain_3d_sum"),
    })
    results["climbable"] = results["prediction"].map({1: "YES", 0: "NO"})

    output_path = PREDICTIONS_DIR / f"{today}.csv"
    results.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Predictions — {today}")
    print(f"{'='*60}")
    print(results[["crag_name", "climbable", "probability"]].to_string(index=False))
    print(f"\nSaved to {output_path}")
    print("\n✓ Inference pipeline complete.")


if __name__ == "__main__":
    run_inference_pipeline()
