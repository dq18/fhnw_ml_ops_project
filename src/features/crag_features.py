"""
Transform raw crag JSON data into a clean, typed DataFrame
suitable for the crag_static Feature Group.

Key transformations:
  - Explode 'orientations' (comma-separated string) into 8 boolean compass columns.
  - Explode 'seasons' (comma-separated string) into 4 boolean columns.
  - Clean categorical columns (rocks, rain_exposure, sun_exposure) to lowercase.
  - Compute num_climbing_types from 'climbing_types'.
"""

import json

import pandas as pd

from src.config import CRAG_JSON_PATH, ORIENTATIONS, SEASONS


def load_raw_crags() -> pd.DataFrame:
    """Load crags from the JSON file."""
    with open(CRAG_JSON_PATH, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def prepare_crag_df(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Clean and feature-engineer the raw crag DataFrame.

    Parameters
    ----------
    raw_df : pd.DataFrame, optional
        If None, loads from the Excel file automatically.

    Returns
    -------
    pd.DataFrame
        Columns: crag_id, name, latitude, longitude, elevation_m, rocks,
        rain_exposure, sun_exposure,
        orientation_north .. orientation_north_west (8 bools),
        season_spring .. season_winter (4 bools),
        num_climbing_types.
    """
    if raw_df is None:
        raw_df = load_raw_crags()

    df = raw_df.copy()

    # ── Select & rename core columns ─────────────────────────────────────
    keep_cols = [
        "crag_id",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "rocks",
        "rain_exposure",
        "sun_exposure",
        "orientations",
        "climbing_types",
        "seasons",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # ── Clean categorical strings ────────────────────────────────────────
    for col in ["rocks", "rain_exposure", "sun_exposure"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # ── Explode orientations → 8 boolean columns ────────────────────────
    orientations_raw = df["orientations"].astype(str).str.strip().str.lower()
    for orient in ORIENTATIONS:
        col_name = "orientation_" + orient.replace(" ", "_")
        df[col_name] = orientations_raw.str.contains(orient, na=False).astype(int)
    df = df.drop(columns=["orientations"])

    # ── Explode seasons → 4 boolean columns ─────────────────────────────
    seasons_raw = df["seasons"].astype(str).str.strip().str.lower()
    for season in SEASONS:
        col_name = "season_" + season
        df[col_name] = seasons_raw.str.contains(season, na=False).astype(int)
    df = df.drop(columns=["seasons"])

    # ── num_climbing_types ───────────────────────────────────────────────
    df["num_climbing_types"] = (
        df["climbing_types"]
        .astype(str)
        .apply(lambda x: len([s for s in x.split(",") if s.strip()]))
    )
    df = df.drop(columns=["climbing_types"])

    # ── Ensure types ─────────────────────────────────────────────────────
    df["crag_id"] = df["crag_id"].astype(int)
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)
