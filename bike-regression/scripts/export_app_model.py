"""Refit the selected notebook pipeline and export a self-contained app artifact.

The model configuration and feature engineering mirror notebooks/02_modeling.ipynb,
which saves the same pipeline in step 14. Use this script to rebuild app/model.joblib
without re-running the notebook. Run from bike-regression/ with the Python 3.13
environment in bike-regression/requirements.txt.
"""

from functools import partial
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

APP_DIR = ROOT / "app"
MODEL_PATH = APP_DIR / "model.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

FEATURES = [
    "hour", "temp_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
    "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm",
    "seasons", "day_of_week", "is_weekend", "is_holiday", "month",
    "hour_sin", "hour_cos", "is_raining", "rain_prev_3h",
]
WEATHER = [
    "temp_c", "humidity_pct", "wind_speed_ms", "visibility_10m",
    "dew_point_c", "solar_radiation_mj", "rainfall_mm", "snowfall_cm",
    "rain_prev_3h",
]


def apply_scope_rules(frame):
    out = frame.loc[frame["functioning_day"].eq("Yes")].copy()
    out["humidity_pct"] = out["humidity_pct"].mask(out["humidity_pct"].eq(0))
    temperature, dew_point = out["temp_c"], out["dew_point_c"]
    rh_magnus = 100 * np.exp(
        17.625 * dew_point / (243.04 + dew_point)
        - 17.625 * temperature / (243.04 + temperature)
    )
    inconsistent = (out["humidity_pct"] - rh_magnus).abs().gt(
        config.MAGNUS_REVIEW_GAP_PP
    )
    out.loc[inconsistent & dew_point.eq(0) & temperature.ne(0), "dew_point_c"] = np.nan
    out.loc[inconsistent & temperature.eq(0) & dew_point.ne(0), "temp_c"] = np.nan
    out = out.loc[~out["date"].isin(pd.to_datetime(config.STORM_DATES))]
    return out.reset_index(drop=True)


def rain_previous_3h(frame):
    key = pd.MultiIndex.from_frame(frame[["date", "hour"]])
    if not key.is_unique:
        raise ValueError("Duplicate date/hour rows in training data")
    lookup = pd.Series(frame["rainfall_mm"].to_numpy(), index=key)
    total = np.zeros(len(frame), dtype=float)
    for lag in (1, 2, 3):
        past_key = pd.MultiIndex.from_arrays(
            [frame["date"], frame["hour"] - lag], names=["date", "hour"]
        )
        past = lookup.reindex(past_key).to_numpy(dtype=float)
        past[frame["hour"].to_numpy() < lag] = 0.0
        total += past
    return total


def add_features(frame):
    out = frame.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = (out["holiday"] == "Holiday").astype(int)
    out["month"] = out["date"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["is_raining"] = (out["rainfall_mm"] > 0).astype(int)
    out["rain_prev_3h"] = rain_previous_3h(out)
    return out


def load_split():
    data = pd.read_csv(config.DATA_CSV, encoding=config.RAW_ENCODING).rename(
        columns=config.COLUMN_MAP
    )
    data["date"] = pd.to_datetime(data["date"], format=config.DATE_FORMAT)
    train_dates = pd.to_datetime(pd.read_csv(config.TRAIN_DATES)["date"])
    test_dates = pd.to_datetime(pd.read_csv(config.TEST_DATES)["date"])
    if set(train_dates) & set(test_dates):
        raise ValueError("Train and test dates overlap")
    train = data.loc[data["date"].isin(train_dates)].sort_values(["date", "hour"])
    test = data.loc[data["date"].isin(test_dates)].sort_values(["date", "hour"])
    train = add_features(apply_scope_rules(train.reset_index(drop=True)))
    test = add_features(apply_scope_rules(test.reset_index(drop=True)))
    if len(train) != 6672 or len(test) != 1656:
        raise ValueError(f"Unexpected scoped split sizes: {len(train)}, {len(test)}")
    return train, test


def make_model():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, WEATHER + ["hour", "is_weekend", "is_holiday", "is_raining"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["seasons", "month", "day_of_week"]),
    ], verbose_feature_names_out=False)
    selector = SelectKBest(
        partial(mutual_info_regression, random_state=config.RANDOM_STATE), k=36
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=200,
        max_leaf_nodes=63,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("pre", preprocessor), ("select", selector), ("model", model)])


def main():
    train, test = load_split()
    pipeline = make_model()
    pipeline.fit(train[FEATURES], train[config.TARGET_COL])
    raw = pipeline.predict(test[FEATURES])
    predictions = np.clip(raw, 0, None)
    truth = test[config.TARGET_COL]
    metrics = {
        "model": "HistGradientBoostingRegressor",
        "test_mae": float(mean_absolute_error(truth, predictions)),
        "test_mse": float(mean_squared_error(truth, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
        "test_r2": float(r2_score(truth, predictions)),
        "test_hours": int(len(test)),
        "features": FEATURES,
    }
    expected = json.loads((ROOT / "verification" / "results.json").read_text(encoding="utf-8"))
    if FEATURES != expected["features"]:
        raise AssertionError("App feature list differs from the selected notebook model")
    for key, source_key in [("test_mae", "MAE"), ("test_mse", "MSE"),
                            ("test_rmse", "RMSE"), ("test_r2", "R2")]:
        if not np.isclose(metrics[key], expected["test_metrics"][source_key], atol=0.001):
            raise AssertionError(f"{key} differs from notebook result: {metrics[key]}")
    APP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "model_bytes": MODEL_PATH.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
