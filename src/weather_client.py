"""
Open-Meteo weather API client.

Provides two functions:
  - fetch_archive_daily(): historical daily weather for a coordinate pair.
  - fetch_forecast_current(): current weather conditions (the RT feature).

Uses requests-cache + retry-requests to be gentle on the free API.
"""

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.config import ARCHIVE_DAILY_VARIABLES, FORECAST_CURRENT_VARIABLES

# ── Shared session (cached + retry) ─────────────────────────────────────────
_cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.5)
_client = openmeteo_requests.Client(session=_retry_session)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_archive_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily historical weather from Open-Meteo Archive API.

    Parameters
    ----------
    latitude, longitude : float
        Coordinates of the crag.
    start_date, end_date : str
        ISO date strings, e.g. "2021-04-24".

    Returns
    -------
    pd.DataFrame
        Columns: date, precipitation_sum, wind_speed_10m_max,
        sunshine_duration, temperature_2m_max, temperature_2m_min,
        shortwave_radiation_sum.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ARCHIVE_DAILY_VARIABLES,
        "timezone": "Europe/Zurich",
    }

    responses = _client.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(ARCHIVE_DAILY_VARIABLES):
        daily_data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(daily_data)
    df["date"] = df["date"].dt.tz_localize(None).dt.date
    # sunshine_duration comes in seconds → convert to hours
    if "sunshine_duration" in df.columns:
        df["sunshine_duration"] = df["sunshine_duration"] / 3600.0
    return df


def fetch_forecast_current(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather conditions from Open-Meteo Forecast API.
    This is the **real-time (RT) feature** used at inference time.

    Parameters
    ----------
    latitude, longitude : float
        Coordinates of the crag.

    Returns
    -------
    dict
        Keys: temperature_2m, wind_speed_10m, cloud_cover, precipitation.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": FORECAST_CURRENT_VARIABLES,
        "timezone": "Europe/Zurich",
    }

    responses = _client.weather_api(FORECAST_URL, params=params)
    response = responses[0]
    current = response.Current()

    result = {}
    for i, var in enumerate(FORECAST_CURRENT_VARIABLES):
        result[var] = current.Variables(i).Value()

    return result


def fetch_recent_daily(
    latitude: float,
    longitude: float,
    days: int = 14,
) -> pd.DataFrame:
    """
    Fetch the most recent N days of daily weather via the Forecast API's
    past_days parameter. Useful to fill the gap between archive lag and today.

    Returns the same DataFrame schema as fetch_archive_daily().
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": days,
        "forecast_days": 1,
        "daily": ARCHIVE_DAILY_VARIABLES,
        "timezone": "Europe/Zurich",
    }

    responses = _client.weather_api(FORECAST_URL, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(ARCHIVE_DAILY_VARIABLES):
        daily_data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(daily_data)
    df["date"] = df["date"].dt.tz_localize(None).dt.date
    if "sunshine_duration" in df.columns:
        df["sunshine_duration"] = df["sunshine_duration"] / 3600.0
    return df
