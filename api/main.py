#!/usr/bin/env python3
"""
Phase 4: Production FastAPI Backend
=====================================
Serves traffic density predictions via REST API.

Endpoints:
    POST /predict   — Predict traffic for a road at a specific date/time
    GET  /health    — Health check
    GET  /roads     — List supported roads
    GET  /docs      — Swagger UI (auto)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import json
import os
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import httpx
import holidays
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------
state = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TrafficRequest(BaseModel):
    road_name: str
    date: str  # "2024-03-15"
    time: str  # "08:00"


class GeohashPrediction(BaseModel):
    geohash: str
    latitude: float
    longitude: float
    congestion_score: float
    traffic_status: str
    road_name: str


class TrafficResponse(BaseModel):
    road_name: str
    requested_datetime: str
    predictions: list[GeohashPrediction]
    model_version: str


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts at startup."""
    logger.info("Loading model artifacts...")

    # Load regression model
    reg_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    state["reg_model"] = joblib.load(reg_path)
    logger.info("  Loaded regression model")

    # Load feature config
    with open(os.path.join(MODELS_DIR, "feature_config.json")) as f:
        state["feature_config"] = json.load(f)
    logger.info("  Loaded feature config (%d features)", state["feature_config"]["feature_count"])

    # Load road mapping
    with open(os.path.join(BASE_DIR, "road_mapping.json")) as f:
        state["road_mapping"] = json.load(f)
    logger.info("  Loaded road mapping (%d roads)", len(state["road_mapping"]["roads"]))

    # Load geohash capacity
    cap_path = os.path.join(DATA_DIR, "geohash_capacity.json")
    if os.path.exists(cap_path):
        with open(cap_path) as f:
            state["capacity"] = json.load(f)
        logger.info("  Loaded geohash capacities (%d)", len(state["capacity"]))
    else:
        state["capacity"] = {}

    # Load traffic thresholds
    thresh_path = os.path.join(DATA_DIR, "traffic_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            state["thresholds"] = json.load(f)
        logger.info("  Loaded traffic thresholds")
    else:
        state["thresholds"] = {}

    # Precompute geohash coordinates from historical data
    parquet_path = os.path.join(DATA_DIR, "filtered_traffic.parquet")
    if os.path.exists(parquet_path):
        hist_df = pd.read_parquet(parquet_path)
        # Geohash coordinate lookup
        gh_coords = hist_df.groupby("GEOHASH").agg(
            lat=("LATITUDE", "mean"),
            lon=("LONGITUDE", "mean"),
        ).to_dict("index")
        state["gh_coords"] = gh_coords

        # Historical averages for lag features (per geohash + hour)
        hist_df["hour"] = pd.to_datetime(hist_df["DATE_TIME"]).dt.hour
        hist_df["day_of_week"] = pd.to_datetime(hist_df["DATE_TIME"]).dt.dayofweek
        lag_avg = hist_df.groupby(["GEOHASH", "hour", "day_of_week"]).agg(
            avg_vehicles=("NUMBER_OF_VEHICLES", "mean"),
            avg_speed=("AVERAGE_SPEED", "mean"),
            avg_congestion=("congestion_score", "mean"),
        ).reset_index()
        state["lag_avg"] = lag_avg
        logger.info("  Loaded historical data for lag computation (%d records)", len(lag_avg))

        del hist_df
    else:
        state["gh_coords"] = {}
        state["lag_avg"] = pd.DataFrame()
        logger.warning("  No historical parquet found — predictions will have limited quality")

    # Load holidays
    state["holidays"] = holidays.Turkey(years=list(range(2023, 2028)))

    logger.info("All artifacts loaded successfully!")
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Istanbul Traffic Density Prediction API",
    description="Predict traffic congestion on Istanbul's major roads",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Feature engineering for prediction
# ---------------------------------------------------------------------------
def build_features(geohashes: list, road_name: str, dt: datetime) -> np.ndarray:
    """Build feature matrix for a list of geohashes at a given datetime."""
    feature_names = state["feature_config"]["feature_names"]
    n_features = len(feature_names)

    hour = dt.hour
    day_of_week = dt.weekday()
    day_of_month = dt.day
    month = dt.month
    week_of_year = dt.isocalendar()[1]

    # Calendar features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    is_holiday = 1 if dt.date() in state["holidays"] else 0
    is_holiday_eve = 1 if (dt.date() + timedelta(days=1)) in state["holidays"] else 0
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    is_school_period = 0 if (month in [6, 7, 8] or
                              (month == 1 and day_of_month >= 20) or
                              (month == 2 and day_of_month <= 5)) else 1

    # Build base feature dict (same for all geohashes)
    base_features = {
        "hour": hour, "day_of_week": day_of_week, "day_of_month": day_of_month,
        "month": month, "week_of_year": week_of_year,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "is_holiday": is_holiday, "is_holiday_eve": is_holiday_eve,
        "is_weekend": is_weekend, "is_rush_hour": is_rush_hour,
        "is_night": is_night, "is_school_period": is_school_period,
        # Weather defaults (will be updated if API succeeds)
        "temperature_2m": 15.0, "precipitation": 0.0,
        "snowfall": 0.0, "wind_speed_10m": 10.0,
    }

    # Per-geohash: compute lag features from historical averages
    lag_avg = state["lag_avg"]
    rows = []
    for gh in geohashes:
        feat = base_features.copy()

        # Get historical lag values for this geohash + hour + dow
        match = lag_avg[
            (lag_avg["GEOHASH"] == gh) &
            (lag_avg["hour"] == hour) &
            (lag_avg["day_of_week"] == day_of_week)
        ]
        if len(match) > 0:
            avg_v = float(match.iloc[0]["avg_vehicles"])
            avg_s = float(match.iloc[0]["avg_speed"])
            avg_c = float(match.iloc[0]["avg_congestion"])
        else:
            # Fallback: use overall average for this geohash
            gh_match = lag_avg[lag_avg["GEOHASH"] == gh]
            if len(gh_match) > 0:
                avg_v = float(gh_match["avg_vehicles"].mean())
                avg_s = float(gh_match["avg_speed"].mean())
                avg_c = float(gh_match["avg_congestion"].mean())
            else:
                avg_v = 50.0
                avg_s = 40.0
                avg_c = 0.5

        feat["lag_1h"] = avg_v
        feat["lag_2h"] = avg_v
        feat["lag_3h"] = avg_v
        feat["lag_24h"] = avg_v
        feat["lag_168h"] = avg_v
        feat["rolling_mean_3h"] = avg_v
        feat["rolling_mean_6h"] = avg_v
        feat["rolling_mean_24h"] = avg_v
        feat["speed_lag_1h"] = avg_s
        feat["speed_lag_24h"] = avg_s
        feat["cs_lag_1h"] = avg_c
        feat["cs_lag_24h"] = avg_c

        # Build feature vector in correct order
        row = [feat.get(fname, 0.0) for fname in feature_names]
        rows.append(row)

    return np.array(rows, dtype=np.float64)


async def fetch_weather(dt: datetime) -> dict:
    """Fetch weather forecast/archive from Open-Meteo."""
    try:
        date_str = dt.strftime("%Y-%m-%d")
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try forecast API first, then archive
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 41.01,
                    "longitude": 28.98,
                    "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
                    "start_date": date_str,
                    "end_date": date_str,
                    "timezone": "Europe/Istanbul",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            hourly = data["hourly"]
            hour = dt.hour
            if hour < len(hourly["temperature_2m"]):
                return {
                    "temperature_2m": hourly["temperature_2m"][hour] or 15.0,
                    "precipitation": hourly["precipitation"][hour] or 0.0,
                    "snowfall": hourly["snowfall"][hour] or 0.0,
                    "wind_speed_10m": hourly["wind_speed_10m"][hour] or 10.0,
                }
    except Exception as e:
        logger.warning("Weather API failed: %s — using defaults", e)

    return {"temperature_2m": 15.0, "precipitation": 0.0, "snowfall": 0.0, "wind_speed_10m": 10.0}


def get_traffic_status(congestion_score: float, road_name: str) -> str:
    """Determine traffic status from congestion score using thresholds."""
    thresholds = state.get("thresholds", {})

    # Determine road type
    highway_roads = ["TEM"]
    group = "highway" if road_name in highway_roads else "urban"

    if group in thresholds:
        t = thresholds[group]["thresholds"]
        # Map congestion score to approximate speed
        # congestion_score ~1.0 → high traffic → low speed → Kilit
        # congestion_score ~0.0 → low traffic → high speed → Akici
        if congestion_score >= 0.75:
            return "Kilit"
        elif congestion_score <= 0.30:
            return "Akici"
        return "Yogun"

    if congestion_score >= 0.75:
        return "Kilit"
    elif congestion_score <= 0.30:
        return "Akici"
    return "Yogun"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "reg_model" in state,
        "roads_loaded": len(state.get("road_mapping", {}).get("roads", {})),
    }


@app.get("/roads")
async def list_roads():
    roads = state.get("road_mapping", {}).get("roads", {})
    return {
        "roads": [
            {
                "name": name,
                "display_name": data["display_name"],
                "geohash_count": len(data["geohashes"]),
            }
            for name, data in roads.items()
        ]
    }


@app.post("/predict", response_model=TrafficResponse)
async def predict(request: TrafficRequest):
    """Predict traffic congestion for a road at a specific date/time."""
    roads = state["road_mapping"]["roads"]

    # Validate road name
    if request.road_name not in roads:
        raise HTTPException(
            status_code=404,
            detail=f"Road '{request.road_name}' not found. Available: {list(roads.keys())}",
        )

    # Parse datetime
    try:
        dt = datetime.strptime(f"{request.date} {request.time}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid date/time format. Use date='YYYY-MM-DD', time='HH:MM'",
        )

    # Enforce 7-day prediction horizon
    now = datetime.now()
    if dt > now + timedelta(days=7):
        raise HTTPException(
            status_code=400,
            detail="Maksimum tahmin ufku 7 gündür (168 saat). Daha kısa bir tarih seçin."
        )

    road_data = roads[request.road_name]
    geohashes = road_data["geohashes"]

    # Fetch weather
    weather = await fetch_weather(dt)

    # Build feature matrix
    X = build_features(geohashes, request.road_name, dt)

    # Update weather features
    feature_names = state["feature_config"]["feature_names"]
    for i, fname in enumerate(feature_names):
        if fname in weather:
            X[:, i] = weather[fname]

    # Predict
    model = state["reg_model"]
    congestion_scores = model.predict(X)
    congestion_scores = np.clip(congestion_scores, 0.0, 1.0)

    # Build predictions
    predictions = []
    for j, gh in enumerate(geohashes):
        coords = state.get("gh_coords", {}).get(gh, {"lat": 41.0, "lon": 29.0})
        cs = float(congestion_scores[j])
        status = get_traffic_status(cs, request.road_name)

        predictions.append(GeohashPrediction(
            geohash=gh,
            latitude=round(coords["lat"], 6),
            longitude=round(coords["lon"], 6),
            congestion_score=round(cs, 4),
            traffic_status=status,
            road_name=request.road_name,
        ))

    return TrafficResponse(
        road_name=request.road_name,
        requested_datetime=dt.strftime("%Y-%m-%d %H:%M"),
        predictions=predictions,
        model_version="1.0.0",
    )
