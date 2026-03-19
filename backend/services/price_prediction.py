from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PricePredictionConfig:
    dataset_dir: Path
    default_horizon: int = 7
    default_lookback: int = 365


def _load_bitcoin_data(dataset_dir: Path) -> pd.DataFrame:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    file_path = files[0]
    data = pd.read_csv(file_path)

    required = ["Timestamp", "Close", "Open", "High", "Low", "Volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset-2: {', '.join(missing)}")

    data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Timestamp", "Close"])
    data["Date"] = pd.to_datetime(data["Timestamp"], unit="s", errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")

    if data.empty:
        raise ValueError("No valid rows remain in dataset-2 after cleaning")

    return data


def _dataset_signature(dataset_dir: Path) -> tuple[str, int, int]:
    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    # Use the first CSV for module-2; include mtime+size to auto-invalidate cache on file change.
    file_path = files[0]
    stat = file_path.stat()
    return (str(file_path.resolve()), int(stat.st_mtime), int(stat.st_size))


@lru_cache(maxsize=4)
def _cached_clean_data(file_path: str, mtime: int, size: int) -> pd.DataFrame:
    _ = (mtime, size)
    data = pd.read_csv(file_path)

    required = ["Timestamp", "Close", "Open", "High", "Low", "Volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset-2: {', '.join(missing)}")

    data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Timestamp", "Close"])
    data["Date"] = pd.to_datetime(data["Timestamp"], unit="s", errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")

    if data.empty:
        raise ValueError("No valid rows remain in dataset-2 after cleaning")

    return data


@lru_cache(maxsize=8)
def _cached_resampled_data(file_path: str, mtime: int, size: int, frequency: str) -> pd.DataFrame:
    base = _cached_clean_data(file_path, mtime, size)
    if frequency not in {"D", "W"}:
        raise ValueError("frequency must be 'D' (daily) or 'W' (weekly)")

    rule = "D" if frequency == "D" else "W"
    series = (
        base.set_index("Date")
        .resample(rule)
        .agg({"Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
        .reset_index()
    )

    if len(series) < 120:
        raise ValueError("Not enough data points after resampling; need at least 120 points")

    return series


def _resample_close(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency not in {"D", "W"}:
        raise ValueError("frequency must be 'D' (daily) or 'W' (weekly)")

    rule = "D" if frequency == "D" else "W"
    series = (
        df.set_index("Date")
        .resample(rule)
        .agg({"Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
        .reset_index()
    )

    if len(series) < 120:
        raise ValueError("Not enough data points after resampling; need at least 120 points")

    return series


def _split_train_test(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_size = max(horizon * 3, 30)
    if len(series) <= test_size + 60:
        test_size = max(20, min(horizon * 2, len(series) // 4))

    train = series.iloc[:-test_size].copy()
    test = series.iloc[-test_size:].copy()

    if len(train) < 60 or len(test) < 10:
        raise ValueError("Insufficient data after train/test split")

    return train, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _linear_regression_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_train = np.arange(len(train), dtype=float)
    y_train = train["Close"].to_numpy(dtype=float)

    coef = np.polyfit(x_train, y_train, 1)

    x_test = np.arange(len(train), len(train) + len(test), dtype=float)
    test_pred = coef[0] * x_test + coef[1]

    x_future = np.arange(len(train) + len(test), len(train) + len(test) + horizon, dtype=float)
    future_pred = coef[0] * x_future + coef[1]

    metrics = _metrics(test["Close"].to_numpy(dtype=float), test_pred)
    return test_pred, future_pred, metrics


def _autoregressive_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    lag: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = train["Close"].to_numpy(dtype=float)
    if len(values) <= lag + 10:
        raise ValueError("Not enough training data for autoregressive model")

    x_rows = []
    y_rows = []
    for i in range(lag, len(values)):
        x_rows.append(values[i - lag : i])
        y_rows.append(values[i])

    x_train = np.array(x_rows, dtype=float)
    y_train = np.array(y_rows, dtype=float)

    x_design = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    beta = np.linalg.lstsq(x_design, y_train, rcond=None)[0]

    def predict_one(window: np.ndarray) -> float:
        feat = np.concatenate([[1.0], window])
        return float(np.dot(feat, beta))

    test_preds: list[float] = []
    history = values.tolist()
    for _ in range(len(test)):
        window = np.array(history[-lag:], dtype=float)
        pred = predict_one(window)
        test_preds.append(pred)
        # Walk-forward evaluation uses true test value for stable scoring.
        history.append(float(test.iloc[len(test_preds) - 1]["Close"]))

    future_preds: list[float] = []
    future_history = np.concatenate([values, test["Close"].to_numpy(dtype=float)]).tolist()
    for _ in range(horizon):
        window = np.array(future_history[-lag:], dtype=float)
        pred = predict_one(window)
        future_preds.append(pred)
        future_history.append(pred)

    test_pred_arr = np.array(test_preds, dtype=float)
    future_pred_arr = np.array(future_preds, dtype=float)
    metrics = _metrics(test["Close"].to_numpy(dtype=float), test_pred_arr)
    return test_pred_arr, future_pred_arr, metrics


def _holt_linear_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    alpha: float = 0.35,
    beta: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y = train["Close"].to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("Not enough observations for Holt linear forecast")

    level = y[0]
    trend = y[1] - y[0]

    for i in range(1, len(y)):
        prev_level = level
        level = alpha * y[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    test_pred = np.array([level + (i + 1) * trend for i in range(len(test))], dtype=float)
    future_pred = np.array([level + (i + 1) * trend for i in range(horizon)], dtype=float)

    metrics = _metrics(test["Close"].to_numpy(dtype=float), test_pred)
    return test_pred, future_pred, metrics


def _future_dates(last_date: pd.Timestamp, horizon: int, frequency: str) -> list[str]:
    if frequency == "D":
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    else:
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    return [d.strftime("%Y-%m-%d") for d in dates]


def build_price_prediction_payload(
    config: PricePredictionConfig,
    horizon: int | None = None,
    lookback: int | None = None,
    frequency: str = "D",
) -> dict[str, Any]:
    horizon = config.default_horizon if horizon is None else int(max(3, min(90, horizon)))
    lookback = config.default_lookback if lookback is None else int(max(120, min(4000, lookback)))

    sig_path, sig_mtime, sig_size = _dataset_signature(config.dataset_dir)
    series = _cached_resampled_data(sig_path, sig_mtime, sig_size, frequency).copy()

    if len(series) > lookback:
        series = series.iloc[-lookback:].copy().reset_index(drop=True)

    train, test = _split_train_test(series, horizon=horizon)

    lr_test, lr_future, lr_metrics = _linear_regression_forecast(train, test, horizon)
    ar_test, ar_future, ar_metrics = _autoregressive_forecast(train, test, horizon)
    holt_test, holt_future, holt_metrics = _holt_linear_forecast(train, test, horizon)

    model_metrics = [
        {"model": "Linear Regression", **lr_metrics},
        {"model": "Autoregressive (AR-7)", **ar_metrics},
        {"model": "Holt Linear Trend", **holt_metrics},
    ]

    best = min(model_metrics, key=lambda m: m["rmse"])
    best_map = {
        "Linear Regression": lr_future,
        "Autoregressive (AR-7)": ar_future,
        "Holt Linear Trend": holt_future,
    }
    best_test_map = {
        "Linear Regression": lr_test,
        "Autoregressive (AR-7)": ar_test,
        "Holt Linear Trend": holt_test,
    }

    future_dates = _future_dates(series["Date"].iloc[-1], horizon, frequency)

    chart_history = series[["Date", "Close"]].copy()
    chart_history["Date"] = chart_history["Date"].dt.strftime("%Y-%m-%d")

    test_dates = test["Date"].dt.strftime("%Y-%m-%d").tolist()
    test_actual = test["Close"].round(6).tolist()

    test_predictions = []
    for i, date in enumerate(test_dates):
        test_predictions.append(
            {
                "Date": date,
                "Actual": float(test_actual[i]),
                "Linear Regression": float(lr_test[i]),
                "Autoregressive (AR-7)": float(ar_test[i]),
                "Holt Linear Trend": float(holt_test[i]),
            }
        )

    best_future = best_map[best["model"]]
    residual_std = float(np.std(test["Close"].to_numpy(dtype=float) - best_test_map[best["model"]]))

    future_predictions = []
    for i, date in enumerate(future_dates):
        point = float(best_future[i])
        future_predictions.append(
            {
                "Date": date,
                "PredictedClose": point,
                "LowerBand": max(0.0, point - 1.96 * residual_std),
                "UpperBand": point + 1.96 * residual_std,
            }
        )

    return {
        "overview": {
            "points": int(len(series)),
            "frequency": "Daily" if frequency == "D" else "Weekly",
            "lookback": int(lookback),
            "horizon": int(horizon),
            "trainPoints": int(len(train)),
            "testPoints": int(len(test)),
            "dateStart": series["Date"].min().strftime("%Y-%m-%d"),
            "dateEnd": series["Date"].max().strftime("%Y-%m-%d"),
            "lastClose": float(series["Close"].iloc[-1]),
        },
        "bestModel": best["model"],
        "modelMetrics": model_metrics,
        "nextForecast": float(future_predictions[0]["PredictedClose"]) if future_predictions else None,
        "charts": {
            "history": chart_history.to_dict(orient="records"),
            "testComparison": test_predictions,
            "futureForecast": future_predictions,
        },
    }
