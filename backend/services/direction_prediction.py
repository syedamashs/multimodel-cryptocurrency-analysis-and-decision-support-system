from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DirectionPredictionConfig:
    dataset_dir: Path
    default_lookback: int = 2000
    default_horizon: int = 7


REQUIRED_BASE_COLUMNS = ["Open", "High", "Low", "Close"]


def _resolve_dataset_file(dataset_dir: Path) -> Path:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Create dataset-3 and place a Bitcoin direction dataset CSV there."
        )

    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {dataset_dir}. "
            "Place dataset-3 CSV with columns Date/Timestamp, Open, High, Low, Close (Volume optional)."
        )

    return files[0]


def _load_dataset(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    # Normalize common variations.
    normalized = data.rename(
        columns={
            "timestamp": "Timestamp",
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).copy()

    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for direction prediction: {', '.join(missing)}")

    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    elif "Timestamp" in normalized.columns:
        normalized["Timestamp"] = pd.to_numeric(normalized["Timestamp"], errors="coerce")
        normalized["Date"] = pd.to_datetime(normalized["Timestamp"], unit="s", errors="coerce")
    else:
        raise ValueError("Dataset needs either 'Date' or 'Timestamp' column")

    for col in ["Open", "High", "Low", "Close"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "Volume" in normalized.columns:
        normalized["Volume"] = pd.to_numeric(normalized["Volume"], errors="coerce").fillna(0.0)
    else:
        normalized["Volume"] = 0.0

    normalized = normalized.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    normalized = normalized.sort_values("Date").reset_index(drop=True)

    if len(normalized) < 300:
        raise ValueError("Direction prediction needs at least 300 valid rows")

    return normalized


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["return_1"] = features["Close"].pct_change()
    features["range_pct"] = (features["High"] - features["Low"]) / features["Close"].replace(0, np.nan)
    features["body_pct"] = (features["Close"] - features["Open"]) / features["Open"].replace(0, np.nan)
    features["volume_chg"] = features["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    features["close_ma_5"] = features["Close"].rolling(5).mean()
    features["close_ma_10"] = features["Close"].rolling(10).mean()
    features["close_ma_20"] = features["Close"].rolling(20).mean()
    features["volatility_10"] = features["return_1"].rolling(10).std()

    features["target"] = (features["Close"].shift(-1) > features["Close"]).astype(int)

    features = features.dropna().reset_index(drop=True)
    if len(features) < 200:
        raise ValueError("Not enough rows after feature engineering; provide a longer dataset")

    return features


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 900,
    lr: float = 0.03,
    l2: float = 0.001,
) -> tuple[np.ndarray, float]:
    n_samples, n_features = x_train.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(epochs):
        logits = x_train @ w + b
        probs = _sigmoid(logits)
        err = probs - y_train

        dw = (x_train.T @ err) / n_samples + l2 * w
        db = float(np.mean(err))

        w -= lr * dw
        b -= lr * db

    return w, b


def _predict_probs(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(x @ w + b)


def _knn_predict_prob(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 11,
) -> np.ndarray:
    k = min(k, len(x_train))
    probs: list[float] = []

    for row in x_test:
        dists = np.sqrt(np.sum((x_train - row) ** 2, axis=1))
        idx = np.argsort(dists)[:k]
        probs.append(float(np.mean(y_train[idx])))

    return np.array(probs, dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float((tp + tn) / max(len(y_true), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2 * precision * recall) / max(precision + recall, 1e-12))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    train_scaled = (x_train - mean) / std
    test_scaled = (x_test - mean) / std
    return train_scaled, test_scaled, mean, std


def build_direction_prediction_payload(
    config: DirectionPredictionConfig,
    lookback: int | None = None,
    threshold: float = 0.5,
    horizon: int | None = None,
) -> dict[str, Any]:
    lookback = config.default_lookback if lookback is None else int(max(300, min(10000, lookback)))
    horizon = config.default_horizon if horizon is None else int(max(3, min(30, horizon)))
    threshold = float(max(0.35, min(0.65, threshold)))

    file_path = _resolve_dataset_file(config.dataset_dir)
    raw = _load_dataset(file_path)

    if len(raw) > lookback:
        raw = raw.iloc[-lookback:].copy().reset_index(drop=True)

    feat_df = _build_features(raw)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "return_1",
        "range_pct",
        "body_pct",
        "volume_chg",
        "close_ma_5",
        "close_ma_10",
        "close_ma_20",
        "volatility_10",
    ]

    x = feat_df[feature_cols].to_numpy(dtype=float)
    y = feat_df["target"].to_numpy(dtype=float)

    split_idx = int(len(feat_df) * 0.8)
    split_idx = min(max(split_idx, 120), len(feat_df) - 30)

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    x_train_scaled, x_test_scaled, mean, std = _standardize(x_train, x_test)

    # Model 1: Logistic regression from scratch.
    w, b = _train_logistic_regression(x_train_scaled, y_train)
    lr_test_prob = _predict_probs(x_test_scaled, w, b)
    lr_test_pred = (lr_test_prob >= threshold).astype(int)
    lr_metrics = _metrics(y_test.astype(int), lr_test_pred.astype(int))

    # Model 2: KNN baseline for comparison.
    knn_test_prob = _knn_predict_prob(x_train_scaled, y_train, x_test_scaled, k=11)
    knn_test_pred = (knn_test_prob >= threshold).astype(int)
    knn_metrics = _metrics(y_test.astype(int), knn_test_pred.astype(int))

    models = [
        {"model": "Logistic Regression", **lr_metrics},
        {"model": "KNN (k=11)", **knn_metrics},
    ]

    best_model_name = max(models, key=lambda m: m["f1"])["model"]
    best_probs = lr_test_prob if best_model_name == "Logistic Regression" else knn_test_prob
    best_preds = lr_test_pred if best_model_name == "Logistic Regression" else knn_test_pred

    x_latest = x[-1]
    x_latest_scaled = (x_latest - mean) / std
    latest_lr_prob = float(_predict_probs(np.array([x_latest_scaled]), w, b)[0])
    latest_knn_prob = float(_knn_predict_prob(x_train_scaled, y_train, np.array([x_latest_scaled]), k=11)[0])

    latest_prob = latest_lr_prob if best_model_name == "Logistic Regression" else latest_knn_prob
    latest_signal = "UP" if latest_prob >= threshold else "DOWN"

    # Multi-day signal projection uses the latest probability as a confidence anchor.
    future_signals = []
    start_date = feat_df["Date"].iloc[-1]
    for i in range(horizon):
        day = start_date + pd.Timedelta(days=i + 1)
        confidence = float(max(0.5, min(0.99, latest_prob - 0.01 * i if latest_prob >= 0.5 else (1 - latest_prob) - 0.01 * i)))
        future_signals.append(
            {
                "Date": day.strftime("%Y-%m-%d"),
                "Direction": latest_signal,
                "Confidence": confidence,
            }
        )

    test_dates = feat_df["Date"].iloc[split_idx:].dt.strftime("%Y-%m-%d").tolist()
    probability_curve = [
        {
            "Date": test_dates[i],
            "ProbabilityUp": float(best_probs[i]),
            "Predicted": "UP" if best_preds[i] == 1 else "DOWN",
            "Actual": "UP" if int(y_test[i]) == 1 else "DOWN",
        }
        for i in range(len(test_dates))
    ]

    importance = []
    lr_weight_magnitude = np.abs(w)
    if np.sum(lr_weight_magnitude) <= 0:
        lr_weight_magnitude = np.ones_like(lr_weight_magnitude)

    norm = lr_weight_magnitude / np.sum(lr_weight_magnitude)
    for i, name in enumerate(feature_cols):
        importance.append({"feature": name, "importance": float(norm[i])})

    class_dist = {
        "up": int(np.sum(y == 1)),
        "down": int(np.sum(y == 0)),
    }

    return {
        "overview": {
            "rows": int(len(feat_df)),
            "trainRows": int(len(x_train)),
            "testRows": int(len(x_test)),
            "lookback": int(lookback),
            "threshold": threshold,
            "datasetFile": file_path.name,
            "dateStart": feat_df["Date"].min().strftime("%Y-%m-%d"),
            "dateEnd": feat_df["Date"].max().strftime("%Y-%m-%d"),
        },
        "bestModel": best_model_name,
        "latestSignal": {
            "direction": latest_signal,
            "probabilityUp": float(latest_prob),
            "threshold": threshold,
        },
        "modelMetrics": models,
        "charts": {
            "classDistribution": [
                {"label": "UP", "count": class_dist["up"]},
                {"label": "DOWN", "count": class_dist["down"]},
            ],
            "confusionMatrix": [
                {"label": "TP", "value": float(max(models, key=lambda m: m["f1"])["tp"] )},
                {"label": "TN", "value": float(max(models, key=lambda m: m["f1"])["tn"] )},
                {"label": "FP", "value": float(max(models, key=lambda m: m["f1"])["fp"] )},
                {"label": "FN", "value": float(max(models, key=lambda m: m["f1"])["fn"] )},
            ],
            "probabilityCurve": probability_curve,
            "featureImportance": sorted(importance, key=lambda x: x["importance"], reverse=True),
            "futureSignals": future_signals,
        },
    }
