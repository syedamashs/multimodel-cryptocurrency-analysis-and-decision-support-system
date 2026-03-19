from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass
class RiskClusteringConfig:
    dataset_dir: Path
    default_days: int = 365
    cluster_count: int = 3


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _read_multi_coin_dataset(dataset_dir: Path) -> pd.DataFrame:
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
    data = df.rename(
        columns={
            "Name": "Coin",
            "name": "Coin",
            "symbol": "Symbol",
            "marketcap": "Marketcap",
        }
    ).copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if "Coin" not in data.columns:
        raise ValueError("Dataset requires either 'Coin' or 'Name' column")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Marketcap" in data.columns:
        data["Marketcap"] = pd.to_numeric(data["Marketcap"], errors="coerce")
    else:
        data["Marketcap"] = np.nan

    data = data.dropna(subset=["Date", "Coin", "Open", "High", "Low", "Close", "Volume"])
    data = data.sort_values(["Coin", "Date"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid rows remain after cleaning")

    return data


def _filter_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    max_date = df["Date"].max()
    min_date = max_date - pd.Timedelta(days=days)
    filtered = df[df["Date"] >= min_date].copy()
    return filtered if not filtered.empty else df.copy()


def _max_drawdown(close_series: pd.Series) -> float:
    if close_series.empty:
        return 0.0
    cum_max = close_series.cummax()
    drawdown = (close_series - cum_max) / cum_max.replace(0, np.nan)
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(abs(drawdown.min()))


def _coin_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for coin, coin_df in df.groupby("Coin", sort=True):
        coin_df = coin_df.sort_values("Date").copy()
        coin_df["return_1"] = coin_df["Close"].pct_change()

        valid_returns = coin_df["return_1"].dropna()
        negative_returns = valid_returns[valid_returns < 0]

        avg_return = float(valid_returns.mean()) if not valid_returns.empty else 0.0
        volatility = float(valid_returns.std(ddof=0) * np.sqrt(365)) if len(valid_returns) > 1 else 0.0
        downside_vol = (
            float(negative_returns.std(ddof=0) * np.sqrt(365)) if len(negative_returns) > 1 else 0.0
        )

        max_dd = _max_drawdown(coin_df["Close"])
        volume_mean = float(coin_df["Volume"].mean())
        volume_std = float(coin_df["Volume"].std(ddof=0))
        volume_cv = float(volume_std / (volume_mean + 1e-9))

        price_range = float(((coin_df["High"] - coin_df["Low"]) / coin_df["Close"].replace(0, np.nan)).mean())
        trend_slope = (
            float(np.polyfit(np.arange(len(coin_df)), coin_df["Close"].to_numpy(dtype=float), 1)[0])
            if len(coin_df) > 1
            else 0.0
        )

        marketcap_mean = float(pd.to_numeric(coin_df["Marketcap"], errors="coerce").dropna().mean())

        rows.append(
            {
                "coin": str(coin),
                "records": int(len(coin_df)),
                "avgReturn": avg_return,
                "volatility": volatility,
                "downsideVolatility": downside_vol,
                "maxDrawdown": max_dd,
                "volumeCV": volume_cv,
                "priceRange": float(0.0 if pd.isna(price_range) else price_range),
                "trendSlope": trend_slope,
                "meanVolume": volume_mean,
                "meanMarketcap": float(0.0 if pd.isna(marketcap_mean) else marketcap_mean),
            }
        )

    feat = pd.DataFrame(rows)
    if len(feat) < 3:
        raise ValueError("Need at least 3 coins for risk clustering")
    return feat


def _minmax(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if abs(high - low) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - low) / (high - low)


def _kmeans(x: np.ndarray, k: int, seed: int = 42, max_iter: int = 120) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]

    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dists = np.sqrt(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        new_labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for i in range(k):
            members = x[new_labels == i]
            if len(members) == 0:
                new_centroids[i] = x[rng.integers(0, n)]
            else:
                new_centroids[i] = members.mean(axis=0)

        if np.array_equal(new_labels, labels):
            centroids = new_centroids
            labels = new_labels
            break

        labels = new_labels
        centroids = new_centroids

    inertia = float(np.sum((x - centroids[labels]) ** 2))
    return labels, centroids, inertia


def build_risk_clustering_payload(
    config: RiskClusteringConfig,
    days: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    days = config.default_days if days is None else int(max(90, min(5000, days)))
    cluster_count = int(config.cluster_count)

    data = _read_multi_coin_dataset(config.dataset_dir)
    data = _normalize_schema(data)
    data = _filter_days(data, days)

    feat = _coin_features(data)

    risk_features = ["volatility", "downsideVolatility", "maxDrawdown", "volumeCV", "priceRange"]
    x = feat[risk_features].to_numpy(dtype=float)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std

    labels, centroids, inertia = _kmeans(x_scaled, k=cluster_count, seed=seed)
    feat["clusterId"] = labels

    risk_index = (
        0.35 * _minmax(feat["volatility"])
        + 0.25 * _minmax(feat["downsideVolatility"])
        + 0.20 * _minmax(feat["maxDrawdown"])
        + 0.10 * _minmax(feat["volumeCV"])
        + 0.10 * _minmax(feat["priceRange"])
    )
    feat["riskIndex"] = risk_index

    cluster_risk = feat.groupby("clusterId")["riskIndex"].mean().sort_values()
    rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_risk.index.tolist())}
    label_map = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk",
    }

    feat["riskLevel"] = feat["clusterId"].map(lambda cid: label_map.get(rank_map.get(cid, 1), "Medium Risk"))

    safest_row = feat.sort_values("riskIndex", ascending=True).iloc[0]
    riskiest_row = feat.sort_values("riskIndex", ascending=False).iloc[0]

    summary_rows = []
    for cluster_id, group in feat.groupby("clusterId"):
        cluster_id_int = int(cast(int, pd.to_numeric(cluster_id, errors="coerce")))
        sorted_rank = rank_map.get(cluster_id, 1)
        summary_rows.append(
            {
                "clusterId": cluster_id_int,
                "riskLevel": label_map.get(sorted_rank, "Medium Risk"),
                "coins": int(len(group)),
                "avgRiskIndex": float(group["riskIndex"].mean()),
                "avgVolatility": float(group["volatility"].mean()),
                "avgDrawdown": float(group["maxDrawdown"].mean()),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda x: x["avgRiskIndex"])

    level_counts = feat["riskLevel"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)

    return {
        "overview": {
            "coins": int(feat["coin"].nunique()),
            "rows": int(len(data)),
            "windowDays": int(days),
            "clusters": int(cluster_count),
            "dateStart": data["Date"].min().strftime("%Y-%m-%d"),
            "dateEnd": data["Date"].max().strftime("%Y-%m-%d"),
            "inertia": inertia,
        },
        "insight": {
            "safestCoin": str(safest_row["coin"]),
            "safestCoinRisk": float(safest_row["riskIndex"]),
            "riskiestCoin": str(riskiest_row["coin"]),
            "riskiestCoinRisk": float(riskiest_row["riskIndex"]),
        },
        "clusterSummary": summary_rows,
        "coinAssignments": feat.round(6).to_dict(orient="records"),
        "charts": {
            "scatter": feat[["coin", "volatility", "maxDrawdown", "riskLevel", "meanVolume"]]
            .round(6)
            .to_dict(orient="records"),
            "riskLevelCounts": [
                {"riskLevel": "Low Risk", "count": int(level_counts["Low Risk"])},
                {"riskLevel": "Medium Risk", "count": int(level_counts["Medium Risk"])},
                {"riskLevel": "High Risk", "count": int(level_counts["High Risk"])},
            ],
            "clusterRiskBars": summary_rows,
            "centroids": [
                {
                    "clusterId": int(i),
                    "volatility": float(centroids[i][0]),
                    "downsideVolatility": float(centroids[i][1]),
                    "maxDrawdown": float(centroids[i][2]),
                    "volumeCV": float(centroids[i][3]),
                    "priceRange": float(centroids[i][4]),
                }
                for i in range(len(centroids))
            ],
        },
    }
