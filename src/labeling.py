"""
Synthetic label computation for crag climbability.

The target variable `climbable` is derived from weather conditions and
the crag's climbing seasons using a transparent, rule-based heuristic.

**This is NOT ground truth.** It is a documented definition used for this
project. The exact rules are:

  climbable = 1  iff ALL of the following hold:
    1. rain_3d_sum        < 15 mm     (rock had time to dry)
    2. precipitation_sum  < 5 mm      (not heavily raining today)
    3. wind_3d_avg        < 50 km/h   (safe wind conditions)
    4. temperature_2m_max >= 3 °C     (not freezing)
    5. temperature_2m_max <= 35 °C    (not dangerously hot)
    6. The month of the date falls within one of the crag's climbing seasons.

Thresholds are defined in src/config.py and can be tuned.
"""

import pandas as pd

from src.config import (
    LABEL_PRECIP_TODAY_MAX,
    LABEL_RAIN_3D_MAX,
    LABEL_TEMP_MAX,
    LABEL_TEMP_MIN,
    LABEL_WIND_3D_AVG_MAX,
    SEASON_MONTHS,
)


def _get_valid_months(season_flags: dict[str, int]) -> set[int]:
    """
    Given a dict like {'season_spring': 1, 'season_summer': 1, ...},
    return the set of valid month numbers.
    """
    valid = set()
    for season, months in SEASON_MONTHS.items():
        key = f"season_{season}"
        if season_flags.get(key, 0) == 1:
            valid.update(months)
    return valid


def compute_climbable(
    weather_row: pd.Series,
    season_flags: dict[str, int],
) -> int:
    """
    Compute the climbable label for a single day + crag combination.

    Parameters
    ----------
    weather_row : pd.Series
        Must contain: rain_3d_sum, precipitation_sum, wind_3d_avg,
        temperature_2m_max, and 'date' (as datetime.date or pd.Timestamp).
    season_flags : dict
        Boolean flags like {'season_spring': 1, 'season_summer': 0, ...}.

    Returns
    -------
    int
        1 if climbable, 0 otherwise.
    """
    # Weather conditions
    if weather_row["rain_3d_sum"] >= LABEL_RAIN_3D_MAX:
        return 0
    if weather_row["precipitation_sum"] >= LABEL_PRECIP_TODAY_MAX:
        return 0
    if weather_row["wind_3d_avg"] >= LABEL_WIND_3D_AVG_MAX:
        return 0
    if weather_row["temperature_2m_max"] < LABEL_TEMP_MIN:
        return 0
    if weather_row["temperature_2m_max"] > LABEL_TEMP_MAX:
        return 0

    # Season check
    date = weather_row["date"]
    if hasattr(date, "month"):
        month = date.month
    else:
        month = pd.Timestamp(date).month

    valid_months = _get_valid_months(season_flags)
    if month not in valid_months:
        return 0

    return 1


def compute_climbable_vectorized(
    weather_df: pd.DataFrame,
    season_flags: dict[str, int],
) -> pd.Series:
    """
    Vectorized version of compute_climbable for an entire DataFrame.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Must contain: rain_3d_sum, precipitation_sum, wind_3d_avg,
        temperature_2m_max, and 'date'.
    season_flags : dict
        Boolean flags for the crag's seasons.

    Returns
    -------
    pd.Series of int
        1 if climbable, 0 otherwise. Same index as weather_df.
    """
    valid_months = _get_valid_months(season_flags)

    dates = pd.to_datetime(weather_df["date"])
    month_ok = dates.dt.month.isin(valid_months)

    climbable = (
        (weather_df["rain_3d_sum"] < LABEL_RAIN_3D_MAX)
        & (weather_df["precipitation_sum"] < LABEL_PRECIP_TODAY_MAX)
        & (weather_df["wind_3d_avg"] < LABEL_WIND_3D_AVG_MAX)
        & (weather_df["temperature_2m_max"] >= LABEL_TEMP_MIN)
        & (weather_df["temperature_2m_max"] <= LABEL_TEMP_MAX)
        & month_ok
    )

    return climbable.astype(int)
