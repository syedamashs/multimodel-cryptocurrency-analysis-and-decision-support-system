from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class AnalysisConfig:
    dataset_dir: Path
    default_days: int = 365


def _read_coin_files(dataset_dir: Path) -> pd.DataFrame:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in csv_files:
        frame = pd.read_csv(file_path)
        frame["__source_file"] = file_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "Name": "Coin",
            "name": "Coin",
            "symbol": "Symbol",
            "marketcap": "Marketcap",
        }
    ).copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")

    if "Coin" not in renamed.columns:
        raise ValueError("Dataset requires either 'Coin' or 'Name' column")

    renamed["Date"] = pd.to_datetime(renamed["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    renamed = renamed.dropna(subset=["Date", "Coin", "Open", "High", "Low", "Close", "Volume"])

    renamed = renamed.sort_values(["Coin", "Date"]).reset_index(drop=True)
    if renamed.empty:
        raise ValueError("No valid rows remain after cleaning.")

    return renamed


def _coin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics: list[dict[str, Any]] = []

    for coin, coin_df in df.groupby("Coin", sort=True):
        coin_df = coin_df.copy().sort_values("Date")
        coin_df["Return"] = coin_df["Close"].pct_change()

        x = np.arange(len(coin_df))
        slope = float(np.polyfit(x, coin_df["Close"], 1)[0]) if len(coin_df) > 1 else 0.0

        corr = coin_df[["Close", "Volume"]].corr().iloc[0, 1]
        corr_value = pd.to_numeric(corr, errors="coerce")
        if pd.isna(corr_value):
            corr_value = 0.0

        metrics.append(
            {
                "coin": str(coin),
                "records": int(len(coin_df)),
                "meanClose": float(coin_df["Close"].mean()),
                "varianceClose": float(coin_df["Close"].var(ddof=0)),
                "annualizedVolatility": float(coin_df["Return"].std(ddof=0) * np.sqrt(365)),
                "trendSlope": slope,
                "priceVolumeCorrelation": float(cast(float, corr_value)),
            }
        )

    result = pd.DataFrame(metrics)
    return result.sort_values("annualizedVolatility", ascending=False).reset_index(drop=True)


def _volatility_label(avg_volatility: float) -> str:
    if avg_volatility >= 0.8:
        return "Highly Volatile"
    if avg_volatility >= 0.4:
        return "Moderately Volatile"
    return "Relatively Stable"


def _trend_label(avg_slope: float) -> str:
    if avg_slope > 0:
        return "Upward"
    if avg_slope < 0:
        return "Downward"
    return "Sideways"


def _date_filtered(df: pd.DataFrame, days: int) -> pd.DataFrame:
    max_date = df["Date"].max()
    min_date = max_date - pd.Timedelta(days=days)
    filtered = df[df["Date"] >= min_date].copy()
    return filtered if not filtered.empty else df.copy()


def build_market_analysis_payload(config: AnalysisConfig, days: int | None = None) -> dict[str, Any]:
    data = _read_coin_files(config.dataset_dir)
    data = _normalize_schema(data)
    days = config.default_days if days is None else max(30, days)
    data = _date_filtered(data, days)

    metrics = _coin_metrics(data)

    avg_volatility = float(metrics["annualizedVolatility"].mean())
    avg_slope = float(metrics["trendSlope"].mean())

    corr_series = pd.to_numeric(metrics["priceVolumeCorrelation"], errors="coerce").fillna(0.0)
    strongest_corr_pos = int(corr_series.abs().to_numpy().argmax())
    strongest_corr_row = metrics.iloc[strongest_corr_pos]

    insight = {
        "marketVolatility": _volatility_label(avg_volatility),
        "marketTrend": _trend_label(avg_slope),
        "strongestPriceVolumeCoin": strongest_corr_row["coin"],
        "strongestPriceVolumeCorrelation": float(strongest_corr_row["priceVolumeCorrelation"]),
    }

    trend_points = data[["Date", "Coin", "Close"]].copy()
    trend_points["Date"] = trend_points["Date"].dt.strftime("%Y-%m-%d")

    return {
        "overview": {
            "coins": int(data["Coin"].nunique()),
            "rows": int(len(data)),
            "dateStart": data["Date"].min().strftime("%Y-%m-%d"),
            "dateEnd": data["Date"].max().strftime("%Y-%m-%d"),
            "windowDays": int(days),
        },
        "insight": insight,
        "coinMetrics": metrics.round(6).to_dict(orient="records"),
        "charts": {
            "trend": trend_points.to_dict(orient="records"),
            "volatility": metrics[["coin", "annualizedVolatility"]].to_dict(orient="records"),
            "correlation": metrics[["coin", "priceVolumeCorrelation"]].to_dict(orient="records"),
        },
    }
