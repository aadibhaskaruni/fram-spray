"""Location -> real hourly weather, using the free Open-Meteo APIs (no API key needed).

    from src.weather_api import load_or_fetch
    df, place_info = load_or_fetch("Pune", "2026-06-01", "2026-09-30")

The DataFrame has one row per hour with columns: time, temp_c, rain_mm, wind_kmh.
Results are cached in data/weather/ so the API is only called once per place/date range.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get_json(url: str, params: dict, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=timeout) as resp:
        return json.load(resp)


def geocode(place: str) -> dict:
    """Place name -> {'name', 'country', 'latitude', 'longitude'} (first match)."""
    data = _get_json(GEOCODE_URL, {"name": place, "count": 1})
    if not data.get("results"):
        raise ValueError(f"Place not found: {place!r}. Try a different spelling or a nearby city.")
    r = data["results"][0]
    return {"name": r["name"], "country": r.get("country", ""), "admin": r.get("admin1", ""),
            "latitude": r["latitude"], "longitude": r["longitude"]}


def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Download hourly temperature (C), precipitation (mm) and 10 m wind speed (km/h)."""
    data = _get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "kmh", "timezone": "auto"})
    h = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp_c": h["temperature_2m"],
                       "rain_mm": h["precipitation"], "wind_kmh": h["wind_speed_10m"]})
    n_missing = int(df[["temp_c", "rain_mm", "wind_kmh"]].isna().sum().sum())
    if n_missing:
        print(f"Note: {n_missing} missing values were filled by interpolation.")
        df[["temp_c", "rain_mm", "wind_kmh"]] = df[["temp_c", "rain_mm", "wind_kmh"]].interpolate().bfill().ffill()
    return df


def load_or_fetch(place: str, start: str, end: str, cache_dir: str = "data/weather"):
    """Return (DataFrame, place_info). Uses a cached CSV if this place/date range was downloaded before."""
    info = geocode(place)
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    path = Path(cache_dir) / f"{slug}_{start}_{end}.csv"
    if path.exists():
        print("Using cached file:", path)
        return pd.read_csv(path, parse_dates=["time"]), info
    df = fetch_hourly(info["latitude"], info["longitude"], start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Downloaded and saved:", path)
    return df, info