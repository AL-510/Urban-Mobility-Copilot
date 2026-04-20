"""Fetch weather data from Open-Meteo API."""

import logging
from datetime import datetime, timedelta

import httpx
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "visibility",
    "weather_code",
]


def fetch_weather_forecast(
    lat: float | None = None,
    lon: float | None = None,
    days: int = 3,
) -> pd.DataFrame:
    """Fetch hourly weather forecast from Open-Meteo."""
    settings = get_settings()
    lat = lat or settings.city_center_lat
    lon = lon or settings.city_center_lon

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "auto",
    }

    resp = httpx.get(settings.weather_api_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        logger.warning("No hourly data returned from weather API")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "timestamp"})
    logger.info(f"Fetched {len(df)} hourly weather records")
    return df


def fetch_weather_history(
    lat: float | None = None,
    lon: float | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Fetch historical hourly weather from Open-Meteo archive API."""
    settings = get_settings()
    lat = lat or settings.city_center_lat
    lon = lon or settings.city_center_lon

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }

    resp = httpx.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "timestamp"})
    logger.info(f"Fetched {len(df)} historical weather records")
    return df


def weather_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw weather to ML-ready features."""
    features = pd.DataFrame()
    features["timestamp"] = df["timestamp"]
    features["temp_c"] = df.get("temperature_2m", 15.0)
    features["humidity_pct"] = df.get("relative_humidity_2m", 50.0)
    features["precip_mm"] = df.get("precipitation", 0.0)
    features["rain_mm"] = df.get("rain", 0.0)
    features["snow_mm"] = df.get("snowfall", 0.0)
    features["wind_speed_kmh"] = df.get("wind_speed_10m", 5.0)
    features["wind_gust_kmh"] = df.get("wind_gusts_10m", 10.0)
    features["visibility_m"] = df.get("visibility", 10000.0)

    wcode = df.get("weather_code", pd.Series([0] * len(df)))
    features["is_fog"] = wcode.isin([45, 48]).astype(float)
    features["is_rain"] = wcode.isin([51, 53, 55, 61, 63, 65, 80, 81, 82]).astype(float)
    features["is_snow"] = wcode.isin([71, 73, 75, 77, 85, 86]).astype(float)
    features["is_storm"] = wcode.isin([95, 96, 99]).astype(float)

    severity = (
        features["precip_mm"].clip(0, 20) / 20 * 0.4
        + features["wind_speed_kmh"].clip(0, 80) / 80 * 0.3
        + (1 - features["visibility_m"].clip(0, 10000) / 10000) * 0.3
    )
    features["weather_severity"] = severity.clip(0, 1)
    return features
