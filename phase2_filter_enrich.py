#!/usr/bin/env python3
"""
Phase 2: Data Filtering, Feature Engineering & Enrichment
==========================================================
Filters IBB traffic data by road geohashes, engineers features
(temporal, weather, lag), and exports enriched parquet.

Inputs:
    - data/traffic_density_*.csv
    - road_mapping.json

Outputs:
    - data/filtered_traffic.parquet
    - data/geohash_capacity.json
    - data/traffic_thresholds.json
    - data/phase2_summary.json
"""

import logging
import time
import json
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import httpx
import holidays

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
random_state = 42
START_TIME = time.time()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CSV_PATTERN = os.path.join(DATA_DIR, "traffic_density_*.csv")
ROAD_MAPPING_FILE = "road_mapping.json"

# Istanbul coordinate bounds
LAT_MIN, LAT_MAX = 40.8, 41.6
LON_MIN, LON_MAX = 27.5, 30.0

# Highway vs urban classification
HIGHWAY_ROADS = ["TEM"]
URBAN_ROADS = ["Buyukdere_Cad", "Sahil_Yolu_Avrupa"]


# ---------------------------------------------------------------------------
# Step 0: Data Quality & EDA
# ---------------------------------------------------------------------------
def step0_data_quality(con: duckdb.DuckDBPyConnection) -> dict:
    """Run data quality checks on raw CSV data."""
    logger.info("--- Step 0: Data Quality & EDA ---")
    summary = {}

    # 0a: Basic stats
    logger.info("Step 0a: Basic statistics")
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT GEOHASH) as unique_geohashes,
            MIN(DATE_TIME) as min_date,
            MAX(DATE_TIME) as max_date
        FROM read_csv_auto('{CSV_PATTERN}', header=true)
    """).fetchone()
    total_rows, unique_gh, min_date, max_date = stats
    logger.info("  Total rows: %s, Unique geohashes: %s", f"{total_rows:,}", unique_gh)
    logger.info("  Date range: %s to %s", min_date, max_date)
    summary["total_raw_rows"] = total_rows
    summary["unique_geohashes_raw"] = unique_gh
    summary["date_range"] = {"start": str(min_date), "end": str(max_date)}

    # 0b: NULL check
    logger.info("Step 0b: NULL check")
    null_counts = con.execute(f"""
        SELECT
            SUM(CASE WHEN DATE_TIME IS NULL THEN 1 ELSE 0 END) as dt_null,
            SUM(CASE WHEN GEOHASH IS NULL THEN 1 ELSE 0 END) as gh_null,
            SUM(CASE WHEN AVERAGE_SPEED IS NULL THEN 1 ELSE 0 END) as speed_null,
            SUM(CASE WHEN NUMBER_OF_VEHICLES IS NULL THEN 1 ELSE 0 END) as veh_null
        FROM read_csv_auto('{CSV_PATTERN}', header=true)
    """).fetchone()
    logger.info("  NULLs — DATE_TIME: %s, GEOHASH: %s, SPEED: %s, VEHICLES: %s",
                null_counts[0], null_counts[1], null_counts[2], null_counts[3])

    # 0c-0e: coordinate validation (sample)
    logger.info("Step 0c-0e: Coordinate validation (sample)")
    sample = con.execute(f"""
        SELECT LATITUDE, LONGITUDE
        FROM read_csv_auto('{CSV_PATTERN}', header=true)
        LIMIT 50000
    """).fetchall()
    invalid = sum(1 for lat, lon in sample if lat and lon and
                  not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX))
    invalid_ratio = invalid / len(sample) if sample else 0
    logger.info("  Coordinate validation: %d/%d invalid (%.1f%%)",
                invalid, len(sample), invalid_ratio * 100)
    if invalid_ratio > 0.30:
        logger.warning("  LAT/LON may be swapped! (%.1f%% invalid)", invalid_ratio * 100)

    return summary


# ---------------------------------------------------------------------------
# Step 1: Geohash Filtering
# ---------------------------------------------------------------------------
def step1_filter_by_geohash(con: duckdb.DuckDBPyConnection, road_mapping: dict) -> pd.DataFrame:
    """Filter CSV data to only road-relevant geohashes using DuckDB."""
    logger.info("--- Step 1: Geohash Filtering ---")

    # Build geohash -> road_name mapping
    gh_to_road = {}
    all_geohashes = set()
    for road_name, road_data in road_mapping["roads"].items():
        for gh in road_data["geohashes"]:
            all_geohashes.add(gh)
            if gh not in gh_to_road:
                gh_to_road[gh] = road_name

    logger.info("  Target geohashes: %d across %d roads",
                len(all_geohashes), len(road_mapping["roads"]))

    # Create temp table for geohashes (avoid massive IN clause)
    gh_list = [(gh,) for gh in all_geohashes]
    con.execute("CREATE TEMPORARY TABLE target_geohashes (geohash VARCHAR)")
    con.executemany("INSERT INTO target_geohashes VALUES (?)", gh_list)

    # Count before filter
    total_before = con.execute(f"""
        SELECT COUNT(*) FROM read_csv_auto('{CSV_PATTERN}', header=true)
    """).fetchone()[0]

    # Filter with SEMI JOIN
    df = con.execute(f"""
        SELECT
            CAST(t.DATE_TIME AS TIMESTAMP) AS DATE_TIME,
            CAST(t.LONGITUDE AS DOUBLE) AS LONGITUDE,
            CAST(t.LATITUDE AS DOUBLE) AS LATITUDE,
            CAST(t.GEOHASH AS VARCHAR) AS GEOHASH,
            CAST(t.MINIMUM_SPEED AS DOUBLE) AS MINIMUM_SPEED,
            CAST(t.MAXIMUM_SPEED AS DOUBLE) AS MAXIMUM_SPEED,
            CAST(t.AVERAGE_SPEED AS DOUBLE) AS AVERAGE_SPEED,
            CAST(t.NUMBER_OF_VEHICLES AS INTEGER) AS NUMBER_OF_VEHICLES
        FROM read_csv_auto('{CSV_PATTERN}', header=true) t
        WHERE t.GEOHASH IN (SELECT geohash FROM target_geohashes)
        AND t.DATE_TIME IS NOT NULL
        AND t.GEOHASH IS NOT NULL
    """).fetchdf()

    logger.info("  Before filter: %s rows", f"{total_before:,}")
    logger.info("  After filter: %s rows (%.1f%%)",
                f"{len(df):,}", len(df) / total_before * 100 if total_before else 0)

    # Add road_name column
    df["road_name"] = df["GEOHASH"].map(gh_to_road)

    # Remove outliers
    before_outlier = len(df)
    df = df[
        (df["AVERAGE_SPEED"] >= 0) & (df["AVERAGE_SPEED"] <= 200) &
        (df["NUMBER_OF_VEHICLES"] >= 0) &
        (df["MINIMUM_SPEED"] <= df["MAXIMUM_SPEED"])
    ].copy()
    logger.info("  After outlier removal: %s rows (removed %d)",
                f"{len(df):,}", before_outlier - len(df))

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["GEOHASH", "DATE_TIME"], keep="last")
    logger.info("  After dedup: %s rows (removed %d duplicates)",
                f"{len(df):,}", before_dedup - len(df))

    # Sort for lag computation
    df = df.sort_values(["GEOHASH", "DATE_TIME"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Step 2: Target Variable Engineering
# ---------------------------------------------------------------------------
def step2_target_engineering(df: pd.DataFrame, road_mapping: dict) -> tuple:
    """Create congestion_score and traffic_status targets."""
    logger.info("--- Step 2: Target Variable Engineering ---")

    # 2a: congestion_score (regression target)
    logger.info("Step 2a: Computing congestion_score")
    p99_capacity = df.groupby("GEOHASH")["NUMBER_OF_VEHICLES"].quantile(0.99)
    capacity_dict = p99_capacity.to_dict()

    # Save capacity
    with open(os.path.join(DATA_DIR, "geohash_capacity.json"), "w") as f:
        json.dump({k: round(float(v), 2) for k, v in capacity_dict.items()}, f, indent=2)
    logger.info("  Saved geohash_capacity.json (%d geohashes)", len(capacity_dict))

    df["congestion_score"] = df.apply(
        lambda row: row["NUMBER_OF_VEHICLES"] / capacity_dict.get(row["GEOHASH"], 1),
        axis=1
    ).clip(0.0, 1.0)

    # 2b: traffic_status (classification target)
    logger.info("Step 2b: Computing traffic_status with EDA-based thresholds")

    # Determine road type for each row
    highway_roads_set = set(HIGHWAY_ROADS)
    urban_roads_set = set(URBAN_ROADS)

    thresholds_data = {"methodology": "percentile_based_p25_p75"}

    for group_name, road_list in [("highway", HIGHWAY_ROADS), ("urban", URBAN_ROADS)]:
        mask = df["road_name"].isin(road_list)
        group_df = df[mask]
        if len(group_df) == 0:
            logger.warning("  No data for %s roads", group_name)
            continue

        speeds = group_df["AVERAGE_SPEED"]
        stats = {
            "mean": round(float(speeds.mean()), 2),
            "median": round(float(speeds.median()), 2),
            "std": round(float(speeds.std()), 2),
            "min": round(float(speeds.min()), 2),
            "max": round(float(speeds.max()), 2),
            "p5": round(float(speeds.quantile(0.05)), 2),
            "p10": round(float(speeds.quantile(0.10)), 2),
            "p25": round(float(speeds.quantile(0.25)), 2),
            "p33": round(float(speeds.quantile(0.33)), 2),
            "p50": round(float(speeds.quantile(0.50)), 2),
            "p66": round(float(speeds.quantile(0.66)), 2),
            "p75": round(float(speeds.quantile(0.75)), 2),
            "p90": round(float(speeds.quantile(0.90)), 2),
            "p95": round(float(speeds.quantile(0.95)), 2),
        }
        kilit_threshold = float(speeds.quantile(0.25))
        akici_threshold = float(speeds.quantile(0.75))

        logger.info("  %s — mean=%.1f, p25=%.1f (kilit<), p75=%.1f (akici>), sample=%d",
                     group_name, stats["mean"], kilit_threshold, akici_threshold, len(group_df))

        thresholds_data[group_name] = {
            "roads": road_list,
            "sample_size": int(len(group_df)),
            "speed_statistics": stats,
            "thresholds": {
                "kilit_below": round(kilit_threshold, 2),
                "akici_above": round(akici_threshold, 2),
            },
        }

    # Save thresholds
    with open(os.path.join(DATA_DIR, "traffic_thresholds.json"), "w") as f:
        json.dump(thresholds_data, f, indent=2)
    logger.info("  Saved traffic_thresholds.json")

    # Assign labels
    def classify_row(row):
        road = row["road_name"]
        speed = row["AVERAGE_SPEED"]
        group = "highway" if road in highway_roads_set else "urban"
        if group not in thresholds_data:
            return "Yogun"
        t = thresholds_data[group]["thresholds"]
        if speed < t["kilit_below"]:
            return "Kilit"
        elif speed > t["akici_above"]:
            return "Akici"
        return "Yogun"

    df["traffic_status"] = df.apply(classify_row, axis=1)

    # Log class distribution
    dist = df["traffic_status"].value_counts()
    total = len(df)
    for cls in ["Akici", "Yogun", "Kilit"]:
        count = dist.get(cls, 0)
        pct = count / total * 100
        logger.info("  %s: %d (%.1f%%)", cls, count, pct)
        if pct < 10:
            logger.warning("  Class '%s' is below 10%%!", cls)

    return df, thresholds_data


# ---------------------------------------------------------------------------
# Step 3: Holiday & Calendar Features
# ---------------------------------------------------------------------------
def step3_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday and calendar features."""
    logger.info("--- Step 3: Holiday & Calendar Features ---")

    years = sorted(df["DATE_TIME"].dt.year.unique())
    tr_holidays = holidays.Turkey(years=years)
    logger.info("  Turkish holidays loaded for years: %s (%d holidays)", years, len(tr_holidays))

    dates = df["DATE_TIME"].dt.date
    df["is_holiday"] = dates.apply(lambda d: d in tr_holidays)
    df["is_holiday_eve"] = dates.apply(lambda d: (d + timedelta(days=1)) in tr_holidays)
    df["is_weekend"] = df["DATE_TIME"].dt.dayofweek.isin([5, 6])

    holiday_count = df["is_holiday"].sum()
    logger.info("  Holiday rows: %d (%.1f%%)", holiday_count, holiday_count / len(df) * 100)

    return df


# ---------------------------------------------------------------------------
# Step 4: Weather Data
# ---------------------------------------------------------------------------
def step4_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo archive API."""
    logger.info("--- Step 4: Weather Data ---")

    min_date = df["DATE_TIME"].min().strftime("%Y-%m-%d")
    max_date = df["DATE_TIME"].max().strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 41.01,
        "longitude": 28.98,
        "start_date": min_date,
        "end_date": max_date,
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "timezone": "Europe/Istanbul",
    }

    weather_df = None
    for attempt in range(1, 4):
        try:
            logger.info("  Weather API attempt %d/3 (range: %s to %s)", attempt, min_date, max_date)
            resp = httpx.get(url, params=params, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()

            hourly = data["hourly"]
            weather_df = pd.DataFrame({
                "weather_datetime": pd.to_datetime(hourly["time"]),
                "temperature_2m": hourly["temperature_2m"],
                "precipitation": hourly["precipitation"],
                "snowfall": hourly["snowfall"],
                "wind_speed_10m": hourly["wind_speed_10m"],
            })
            logger.info("  Weather data fetched: %d hourly records", len(weather_df))
            break
        except Exception as e:
            wait = 2 ** attempt
            logger.error("  Weather API attempt %d failed: %s. Waiting %ds...", attempt, e, wait)
            time.sleep(wait)

    if weather_df is None:
        logger.warning("  All weather API attempts failed! Weather columns will be NaN.")
        df["temperature_2m"] = np.nan
        df["precipitation"] = np.nan
        df["snowfall"] = np.nan
        df["wind_speed_10m"] = np.nan
        return df

    # Round traffic datetime to nearest hour for join
    df["weather_join_key"] = df["DATE_TIME"].dt.floor("h")
    weather_df = weather_df.rename(columns={"weather_datetime": "weather_join_key"})

    before = len(df)
    df = df.merge(weather_df, on="weather_join_key", how="left")
    df = df.drop(columns=["weather_join_key"])

    # Forward-fill NaN weather values
    for col in ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]:
        null_count = df[col].isna().sum()
        df[col] = df[col].ffill()
        remaining = df[col].isna().sum()
        if null_count > 0:
            logger.info("  Weather '%s': %d NaN → ffill → %d remaining", col, null_count, remaining)

    coverage = (1 - df["temperature_2m"].isna().mean()) * 100
    logger.info("  Weather coverage: %.1f%%", coverage)

    return df


# ---------------------------------------------------------------------------
# Step 5: Temporal Feature Engineering
# ---------------------------------------------------------------------------
def step5_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from DATE_TIME."""
    logger.info("--- Step 5: Temporal Feature Engineering ---")

    dt = df["DATE_TIME"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Domain-specific
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19])
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5])
    df["is_school_period"] = ~(
        df["month"].isin([6, 7, 8]) |
        ((df["month"] == 1) & (df["day_of_month"] >= 20)) |
        ((df["month"] == 2) & (df["day_of_month"] <= 5))
    )

    logger.info("  Created %d temporal features", 17)
    return df


# ---------------------------------------------------------------------------
# Step 6: Lag Features
# ---------------------------------------------------------------------------
def step6_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features grouped by GEOHASH."""
    logger.info("--- Step 6: Lag Features (CRITICAL — grouped by GEOHASH) ---")

    # CRITICAL: Sort by (GEOHASH, DATE_TIME) before lag computation
    df = df.sort_values(["GEOHASH", "DATE_TIME"]).reset_index(drop=True)
    grouped = df.groupby("GEOHASH")

    # Vehicle count lags
    for lag_name, periods in [("lag_1h", 1), ("lag_2h", 2), ("lag_3h", 3),
                               ("lag_24h", 24), ("lag_168h", 168)]:
        df[lag_name] = grouped["NUMBER_OF_VEHICLES"].shift(periods)

    # Rolling means
    for window, name in [(3, "rolling_mean_3h"), (6, "rolling_mean_6h"), (24, "rolling_mean_24h")]:
        df[name] = grouped["NUMBER_OF_VEHICLES"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Speed lags (past speed = usable as feature, current speed = NOT)
    df["speed_lag_1h"] = grouped["AVERAGE_SPEED"].shift(1)
    df["speed_lag_24h"] = grouped["AVERAGE_SPEED"].shift(24)

    # Congestion score lags
    df["cs_lag_1h"] = grouped["congestion_score"].shift(1)
    df["cs_lag_24h"] = grouped["congestion_score"].shift(24)

    # Log NaN percentages
    lag_cols = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
                "rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h",
                "speed_lag_1h", "speed_lag_24h", "cs_lag_1h", "cs_lag_24h"]
    lag_nan_pct = {}
    for col in lag_cols:
        nan_pct = df[col].isna().mean() * 100
        lag_nan_pct[col] = round(nan_pct, 2)
        logger.info("  %s NaN: %.1f%%", col, nan_pct)

    return df, lag_nan_pct


# ---------------------------------------------------------------------------
# Step 7: Train/Test Split
# ---------------------------------------------------------------------------
def step7_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal train/validation/test split column."""
    logger.info("--- Step 7: Train/Validation/Test Split ---")

    df["split"] = "train"
    df.loc[df["DATE_TIME"] >= "2024-07-01", "split"] = "validation"
    df.loc[df["DATE_TIME"] >= "2024-10-01", "split"] = "test"

    for split_name in ["train", "validation", "test"]:
        count = (df["split"] == split_name).sum()
        logger.info("  %s: %d rows (%.1f%%)", split_name, count, count / len(df) * 100)

    return df


# ---------------------------------------------------------------------------
# Step 8: Save
# ---------------------------------------------------------------------------
def step8_save(df: pd.DataFrame, summary: dict, thresholds: dict, lag_nan_pct: dict) -> None:
    """Save parquet and summary JSON."""
    logger.info("--- Step 8: Saving outputs ---")

    # Save parquet
    parquet_path = os.path.join(DATA_DIR, "filtered_traffic.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    parquet_size = os.path.getsize(parquet_path)
    logger.info("  Saved %s (%.1f MB)", parquet_path, parquet_size / 1024 / 1024)

    # Build summary
    split_dist = df["split"].value_counts().to_dict()
    status_dist = df["traffic_status"].value_counts().to_dict()

    weather_coverage = (1 - df["temperature_2m"].isna().mean()) * 100 if "temperature_2m" in df.columns else 0

    phase2_summary = {
        "total_raw_rows": summary.get("total_raw_rows", 0),
        "after_quality_filter": len(df),
        "after_geohash_filter": len(df),
        "final_row_count": len(df),
        "unique_geohashes": int(df["GEOHASH"].nunique()),
        "date_range": summary.get("date_range", {}),
        "columns": list(df.columns),
        "split_distribution": {k: int(v) for k, v in split_dist.items()},
        "weather_coverage_percentage": round(weather_coverage, 1),
        "lag_nan_percentage": lag_nan_pct,
        "target_distribution": {
            "congestion_score": {
                "mean": round(float(df["congestion_score"].mean()), 4),
                "std": round(float(df["congestion_score"].std()), 4),
            },
            "traffic_status": {k: int(v) for k, v in status_dist.items()},
        },
        "eda_thresholds": {
            group: {
                "kilit_below": thresholds[group]["thresholds"]["kilit_below"],
                "akici_above": thresholds[group]["thresholds"]["akici_above"],
            }
            for group in ["highway", "urban"]
            if group in thresholds
        },
        "execution_time_seconds": round(time.time() - START_TIME, 1),
    }

    summary_path = os.path.join(DATA_DIR, "phase2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(phase2_summary, f, indent=2)
    logger.info("  Saved %s", summary_path)

    # Log all output files
    for fpath in [parquet_path,
                  os.path.join(DATA_DIR, "geohash_capacity.json"),
                  os.path.join(DATA_DIR, "traffic_thresholds.json"),
                  summary_path]:
        if os.path.exists(fpath):
            logger.info("  Output: %s (%.1f KB)", fpath, os.path.getsize(fpath) / 1024)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Phase 2: Data Filtering, Feature Engineering & Enrichment — START")
    logger.info("random_state = %d", random_state)
    logger.info("=" * 60)

    # Load road mapping
    with open(ROAD_MAPPING_FILE, "r") as f:
        road_mapping = json.load(f)
    logger.info("Loaded %s (%d roads)", ROAD_MAPPING_FILE, len(road_mapping["roads"]))

    # DuckDB connection
    con = duckdb.connect(database=":memory:")
    try:
        # Step 0: Data quality
        summary = step0_data_quality(con)

        # Step 1: Filter by geohash
        df = step1_filter_by_geohash(con, road_mapping)
    finally:
        con.close()

    # Step 2: Target engineering
    df, thresholds = step2_target_engineering(df, road_mapping)

    # Step 3: Holiday features
    df = step3_holiday_features(df)

    # Step 4: Weather
    df = step4_weather(df)

    # Step 5: Temporal features
    df = step5_temporal_features(df)

    # Step 6: Lag features
    df, lag_nan_pct = step6_lag_features(df)

    # Step 7: Split
    df = step7_split(df)

    # Step 8: Save
    step8_save(df, summary, thresholds, lag_nan_pct)

    elapsed = time.time() - START_TIME
    logger.info("=" * 60)
    logger.info("Phase 2 COMPLETE in %.1f seconds.", elapsed)
    logger.info("Final dataset: %s rows, %d columns", f"{len(df):,}", len(df.columns))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
