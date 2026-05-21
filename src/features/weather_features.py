"""
Compute rolling / aggregated weather features from daily weather data.

These are the **aggregated features** required by the assignment:
  - rain_3d_sum   : cumulative precipitation over the last 3 days
  - rain_7d_sum   : cumulative precipitation over the last 7 days
  - wind_3d_avg   : average of daily max wind speed over the last 3 days
  - sun_3d_hours  : cumulative sunshine hours over the last 3 days
  - days_since_rain: number of days since last precipitation > 1 mm
"""

import numpy as np
import pandas as pd


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling / aggregated weather features to a daily weather DataFrame.

    The DataFrame must already be sorted by date and contain columns:
      - precipitation_sum
      - wind_speed_10m_max
      - sunshine_duration  (in hours)

    Parameters
    ----------
    df : pd.DataFrame
        Daily weather data for a single crag, sorted by date ascending.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with additional columns:
        rain_3d_sum, rain_7d_sum, wind_3d_avg, sun_3d_hours, days_since_rain.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ── Rolling sums / averages ──────────────────────────────────────────
    df["rain_3d_sum"] = (
        df["precipitation_sum"].rolling(window=3, min_periods=1).sum()
    )
    df["rain_7d_sum"] = (
        df["precipitation_sum"].rolling(window=7, min_periods=1).sum()
    )
    df["wind_3d_avg"] = (
        df["wind_speed_10m_max"].rolling(window=3, min_periods=1).mean()
    )
    df["sun_3d_hours"] = (
        df["sunshine_duration"].rolling(window=3, min_periods=1).sum()
    )

    # ── Days since last significant rain (>1 mm) ────────────────────────
    rainy = df["precipitation_sum"] > 1.0
    # Create groups that increment each time it rains
    rain_groups = rainy.cumsum()
    # Within each group, count days since the rain event
    df["days_since_rain"] = df.groupby(rain_groups).cumcount()
    # If it never rained, days_since_rain = row index (i.e., all days)
    if not rainy.any():
        df["days_since_rain"] = np.arange(len(df))

    return df
