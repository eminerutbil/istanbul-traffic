#!/usr/bin/env python3
"""
Phase 3: Model Training, Comparison & Export (GPU-accelerated)
===============================================================
Trains regression and classification models on GPU (XGBoost CUDA,
LightGBM GPU) with RAM-safe sampling. Exports all artifacts.
"""

import logging
import time
import json
import os
import warnings
from itertools import product as itertools_product

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    silhouette_score,
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# ---------------------------------------------------------------------------
random_state = 42
START_TIME = time.time()
MAX_TRAIN_ROWS = 400_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)

REGRESSION_TARGET = "congestion_score"
CLASSIFICATION_TARGET = "traffic_status"

CALENDAR_FEATURES = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "is_holiday_eve", "is_weekend",
    "is_rush_hour", "is_night", "is_school_period",
]
WEATHER_FEATURES = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
LAG_FEATURES = [
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h",
    "speed_lag_1h", "speed_lag_24h", "cs_lag_1h", "cs_lag_24h",
]
ALL_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES

LEAKAGE_COLUMNS = ["AVERAGE_SPEED", "MINIMUM_SPEED", "MAXIMUM_SPEED", "NUMBER_OF_VEHICLES"]


def param_combos(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, v)) for v in itertools_product(*grid.values())]


# ---------------------------------------------------------------------------
# Step 0: Load
# ---------------------------------------------------------------------------
def load_data():
    logger.info("--- Step 0: Loading data ---")
    df = pd.read_parquet(os.path.join(DATA_DIR, "filtered_traffic.parquet"))
    logger.info("  Loaded %s rows, %d cols", f"{len(df):,}", len(df.columns))

    for col in LEAKAGE_COLUMNS:
        assert col not in ALL_FEATURES, f"LEAKAGE: {col}"

    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        logger.warning("  Missing features: %s", missing)

    for c in ["is_holiday", "is_holiday_eve", "is_weekend", "is_rush_hour", "is_night", "is_school_period"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    lag_avail = [c for c in LAG_FEATURES if c in train_df.columns]
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        b = len(sdf)
        sdf.dropna(subset=lag_avail, inplace=True)
        logger.info("  %s: %d -> %d after lag NaN drop", name, b, len(sdf))

    # Sample train if too large
    if len(train_df) > MAX_TRAIN_ROWS:
        logger.info("  Sampling train from %d to %d rows", len(train_df), MAX_TRAIN_ROWS)
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=random_state)

    logger.info("  Final — Train: %s, Val: %s, Test: %s",
                f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}")
    return train_df, val_df, test_df, available


# ---------------------------------------------------------------------------
# Step 1: Regression
# ---------------------------------------------------------------------------
def step1_regression(train_df, val_df, test_df, features):
    logger.info("--- Step 1: Regression Models ---")

    X_tr = np.nan_to_num(train_df[features].values, nan=0.0).astype(np.float32)
    y_tr = train_df[REGRESSION_TARGET].values.astype(np.float32)
    X_val = np.nan_to_num(val_df[features].values, nan=0.0).astype(np.float32)
    y_val = val_df[REGRESSION_TARGET].values.astype(np.float32)
    X_te = np.nan_to_num(test_df[features].values, nan=0.0).astype(np.float32)
    y_te = test_df[REGRESSION_TARGET].values.astype(np.float32)

    configs = {
        "RandomForest": {
            "cls": RandomForestRegressor,
            "grid": {"n_estimators": [100], "max_depth": [10],
                     "random_state": [42], "n_jobs": [-1]},
        },
        "XGBoost": {
            "cls": XGBRegressor,
            "grid": {"n_estimators": [300], "max_depth": [6, 8],
                     "learning_rate": [0.05, 0.1], "subsample": [0.8],
                     "colsample_bytree": [0.8], "device": ["cuda"],
                     "tree_method": ["hist"], "random_state": [42], "verbosity": [0]},
        },
        "LightGBM": {
            "cls": LGBMRegressor,
            "grid": {"n_estimators": [300, 500], "max_depth": [10, 15],
                     "learning_rate": [0.05, 0.1], "subsample": [0.8],
                     "colsample_bytree": [0.8], "device": ["gpu"],
                     "random_state": [42], "verbose": [-1]},
        },
    }

    best_model, best_name, best_rmse = None, None, float("inf")
    all_results = {}

    for mname, cfg in configs.items():
        logger.info("  Training %s...", mname)
        combos = param_combos(cfg["grid"])
        bm, bp, br = None, None, float("inf")

        for p in combos:
            try:
                t0 = time.time()
                m = cfg["cls"](**p)
                m.fit(X_tr, y_tr)
                vp = m.predict(X_val)
                rmse = float(np.sqrt(mean_squared_error(y_val, vp)))
                logger.info("    params=%s val_RMSE=%.4f (%.1fs)",
                            {k: v for k, v in p.items() if k not in
                             ["random_state", "n_jobs", "verbosity", "verbose",
                              "device", "tree_method"]}, rmse, time.time() - t0)
                if rmse < br:
                    br, bm, bp = rmse, m, p
            except Exception as e:
                logger.error("    FAILED: %s", e)

        if bm is None:
            continue

        tp = bm.predict(X_te)
        t_rmse = float(np.sqrt(mean_squared_error(y_te, tp)))
        t_mae = float(mean_absolute_error(y_te, tp))
        t_r2 = float(r2_score(y_te, tp))
        nz = y_te > 0.01
        t_mape = float(np.mean(np.abs(y_te[nz] - tp[nz]) / y_te[nz]) * 100) if nz.sum() > 0 else 0

        logger.info("  %s BEST — Val RMSE: %.4f | Test RMSE: %.4f, MAE: %.4f, R²: %.4f, MAPE: %.1f%%",
                     mname, br, t_rmse, t_mae, t_r2, t_mape)

        all_results[mname] = {
            "best_params": {k: str(v) for k, v in bp.items()
                           if k not in ["random_state", "n_jobs", "verbosity", "verbose"]},
            "val": {"RMSE": round(br, 4)},
            "test": {"RMSE": round(t_rmse, 4), "MAE": round(t_mae, 4),
                     "R2": round(t_r2, 4), "MAPE": round(t_mape, 2)},
        }
        if br < best_rmse:
            best_rmse, best_model, best_name = br, bm, mname

    # Feature importance
    fi = []
    if best_model and hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        for idx in np.argsort(imp)[::-1][:15]:
            fi.append({"feature": features[idx], "importance": round(float(imp[idx]), 4)})
            logger.info("    Top feature: %s = %.4f", features[idx], imp[idx])

    logger.info("  >>> Best regression: %s (RMSE=%.4f)", best_name, best_rmse)
    return best_model, best_name, all_results, fi


# ---------------------------------------------------------------------------
# Step 2: Classification
# ---------------------------------------------------------------------------
def step2_classification(train_df, val_df, test_df, features):
    logger.info("--- Step 2: Classification Models ---")

    le = LabelEncoder()
    le.fit(["Akici", "Yogun", "Kilit"])

    X_tr = np.nan_to_num(train_df[features].values, nan=0.0).astype(np.float32)
    y_tr = le.transform(train_df[CLASSIFICATION_TARGET].values)
    X_val = np.nan_to_num(val_df[features].values, nan=0.0).astype(np.float32)
    y_val = le.transform(val_df[CLASSIFICATION_TARGET].values)
    X_te = np.nan_to_num(test_df[features].values, nan=0.0).astype(np.float32)
    y_te = le.transform(test_df[CLASSIFICATION_TARGET].values)

    configs = {
        "LogisticRegression_baseline": {
            "cls": LogisticRegression,
            "grid": {"C": [1.0], "max_iter": [1000], "class_weight": ["balanced"],
                     "random_state": [42], "n_jobs": [-1]},
        },
        "RandomForest_clf": {
            "cls": RandomForestClassifier,
            "grid": {"n_estimators": [100], "max_depth": [10],
                     "class_weight": ["balanced"], "random_state": [42], "n_jobs": [-1]},
        },
        "XGBoost_clf": {
            "cls": XGBClassifier,
            "grid": {"n_estimators": [300], "max_depth": [6, 8],
                     "learning_rate": [0.05, 0.1], "subsample": [0.8],
                     "colsample_bytree": [0.8], "device": ["cuda"],
                     "tree_method": ["hist"], "random_state": [42],
                     "verbosity": [0], "eval_metric": ["mlogloss"]},
        },
        "LightGBM_clf": {
            "cls": LGBMClassifier,
            "grid": {"n_estimators": [300, 500], "max_depth": [10, 15],
                     "learning_rate": [0.05, 0.1], "subsample": [0.8],
                     "colsample_bytree": [0.8], "device": ["gpu"],
                     "random_state": [42], "verbose": [-1]},
        },
    }

    best_model, best_name, best_f1 = None, None, -1
    all_results = {}

    for mname, cfg in configs.items():
        logger.info("  Training %s...", mname)
        combos = param_combos(cfg["grid"])
        bm, bp, bf = None, None, -1

        for p in combos:
            try:
                t0 = time.time()
                m = cfg["cls"](**p)
                m.fit(X_tr, y_tr)
                vp = m.predict(X_val)
                f1w = float(f1_score(y_val, vp, average="weighted"))
                logger.info("    val_F1w=%.4f (%.1fs)", f1w, time.time() - t0)
                if f1w > bf:
                    bf, bm, bp = f1w, m, p
            except Exception as e:
                logger.error("    FAILED: %s", e)

        if bm is None:
            continue

        tp = bm.predict(X_te)
        acc = float(accuracy_score(y_te, tp))
        f1w = float(f1_score(y_te, tp, average="weighted"))
        f1m = float(f1_score(y_te, tp, average="macro"))
        f1pc = {c: round(float(v), 4) for c, v in
                zip(le.classes_, f1_score(y_te, tp, average=None))}
        cm = confusion_matrix(y_te, tp).tolist()

        logger.info("  %s — Acc: %.4f, F1w: %.4f, F1m: %.4f, per_class: %s",
                     mname, acc, f1w, f1m, f1pc)

        note = "Baseline." if "Logistic" in mname else ""
        all_results[mname] = {
            "note": note,
            "best_params": {k: str(v) for k, v in bp.items()
                           if k not in ["random_state", "n_jobs", "verbosity", "verbose"]},
            "val": {"f1_weighted": round(bf, 4)},
            "test": {"accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
                     "f1_macro": round(f1m, 4), "f1_per_class": f1pc,
                     "confusion_matrix": cm},
        }
        if bf > best_f1:
            best_f1, best_model, best_name = bf, bm, mname

    logger.info("  >>> Best classification: %s (F1w=%.4f)", best_name, best_f1)
    return best_model, best_name, all_results, le


# ---------------------------------------------------------------------------
# Step 3: Clustering
# ---------------------------------------------------------------------------
def step3_clustering(full_df):
    logger.info("--- Step 3: Clustering ---")
    profile = full_df.groupby("GEOHASH").agg(
        avg_speed=("AVERAGE_SPEED", "mean"), avg_vehicles=("NUMBER_OF_VEHICLES", "mean"),
        avg_congestion=("congestion_score", "mean"),
        lat=("LATITUDE", "mean"), lon=("LONGITUDE", "mean"),
        road_name=("road_name", "first"),
    ).reset_index()

    if len(profile) < 3:
        logger.warning("  Too few geohashes. Skipping.")
        return {"best_k": 0, "silhouette_score": 0, "cluster_profiles": []}

    X = StandardScaler().fit_transform(profile[["avg_speed", "avg_vehicles", "avg_congestion"]].values)

    best_k, best_sil = 2, -1
    for k in range(2, min(9, len(profile))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        sil = float(silhouette_score(X, labels))
        logger.info("  K=%d sil=%.4f", k, sil)
        if sil > best_sil:
            best_sil, best_k = sil, k

    profile["cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)

    profiles = []
    for c in range(best_k):
        cd = profile[profile["cluster"] == c]
        s, v = float(cd["avg_speed"].mean()), float(cd["avg_vehicles"].mean())
        label = ("Highway_High_Traffic" if s > profile["avg_speed"].median() and v > profile["avg_vehicles"].median()
                 else "Fast_Low_Traffic" if s > profile["avg_speed"].median()
                 else "Slow_High_Traffic" if v > profile["avg_vehicles"].median()
                 else "Urban_Moderate")
        profiles.append({"cluster_id": c, "label": label, "count": len(cd),
                         "avg_speed": round(s, 2), "avg_vehicles": round(v, 2)})
        logger.info("  Cluster %d (%s): %d gh", c, label, len(cd))

    profile.to_csv(os.path.join(DATA_DIR, "geohash_clusters.csv"), index=False)
    return {"best_k": best_k, "silhouette_score": round(best_sil, 4), "cluster_profiles": profiles}


# ---------------------------------------------------------------------------
# Step 4: Export
# ---------------------------------------------------------------------------
def step4_export(best_reg, best_reg_name, reg_results, fi,
                 best_clf, best_clf_name, clf_results, le,
                 clustering, features, tr, va, te):
    logger.info("--- Step 4: Export ---")

    if best_reg:
        joblib.dump(best_reg, os.path.join(MODELS_DIR, "best_regression_model.pkl"))
        logger.info("  Saved best_regression_model.pkl")
    if best_clf:
        joblib.dump(best_clf, os.path.join(MODELS_DIR, "best_classification_model.pkl"))
        logger.info("  Saved best_classification_model.pkl")

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_config.json"), "w") as f:
        json.dump({"feature_names": features, "feature_count": len(features),
                    "regression_target": REGRESSION_TARGET,
                    "classification_target": CLASSIFICATION_TARGET}, f, indent=2)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "random_state": 42,
        "data_info": {"train_rows": tr, "val_rows": va, "test_rows": te, "feature_count": len(features)},
        "regression": {"best_model": best_reg_name or "None", "all_models": reg_results,
                       "feature_importance_top15": fi},
        "classification": {"best_model": best_clf_name or "None",
                           "production_strategy": "regression_first_with_threshold_fallback",
                           "all_models": clf_results},
        "clustering": clustering,
    }
    with open(os.path.join(MODELS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  Saved evaluation_report.json")

    for fn in os.listdir(MODELS_DIR):
        fp = os.path.join(MODELS_DIR, fn)
        logger.info("  Output: %s (%.1f KB)", fp, os.path.getsize(fp) / 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Phase 3: Model Training (GPU) — START")
    logger.info("random_state=%d, max_train=%d", random_state, MAX_TRAIN_ROWS)
    logger.info("=" * 60)

    train_df, val_df, test_df, features = load_data()

    best_reg, best_reg_name, reg_results, fi = step1_regression(train_df, val_df, test_df, features)
    best_clf, best_clf_name, clf_results, le = step2_classification(train_df, val_df, test_df, features)

    full_df = pd.concat([train_df, val_df, test_df])
    clustering = step3_clustering(full_df)

    step4_export(best_reg, best_reg_name, reg_results, fi,
                 best_clf, best_clf_name, clf_results, le,
                 clustering, features, len(train_df), len(val_df), len(test_df))

    logger.info("=" * 60)
    logger.info("Phase 3 COMPLETE in %.1f seconds.", time.time() - START_TIME)
    logger.info("Best regression: %s", best_reg_name)
    logger.info("Best classification: %s", best_clf_name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
