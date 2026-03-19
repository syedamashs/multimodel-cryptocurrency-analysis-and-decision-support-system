from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request
from flask_cors import CORS

from services.chatbot import ChatbotConfig, ask_ollama
from services.direction_prediction import DirectionPredictionConfig, build_direction_prediction_payload
from services.image_analysis import ImageAnalysisConfig, analyze_image_news
from services.market_analysis import AnalysisConfig, build_market_analysis_payload
from services.price_prediction import PricePredictionConfig, build_price_prediction_payload
from services.risk_clustering import RiskClusteringConfig, build_risk_clustering_payload
from services.sentiment_analysis import SentimentConfig, analyze_sentiment


def _resolve_data_root(root_dir: Path) -> Path:
    datasets_dir = root_dir / "datasets"
    return datasets_dir if datasets_dir.exists() and datasets_dir.is_dir() else root_dir


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    root_dir = Path(__file__).resolve().parent.parent
    data_root = _resolve_data_root(root_dir)

    dataset_dir = Path(os.getenv("DATASET_DIR", data_root / "dataset-1"))
    dataset2_dir = Path(os.getenv("DATASET2_DIR", data_root / "dataset-2"))
    dataset3_dir = Path(os.getenv("DATASET3_DIR", data_root / "dataset-2"))
    sentiment_dataset_dir = Path(os.getenv("SENTIMENT_DATASET_DIR", data_root / "dataset-sentiment"))
    config = AnalysisConfig(dataset_dir=dataset_dir, default_days=365)
    price_config = PricePredictionConfig(dataset_dir=dataset2_dir, default_horizon=7, default_lookback=365)
    direction_config = DirectionPredictionConfig(dataset_dir=dataset3_dir, default_lookback=2000, default_horizon=7)
    risk_config = RiskClusteringConfig(dataset_dir=dataset_dir, default_days=365, cluster_count=3)
    sentiment_config = SentimentConfig(confidence_floor=0.45, dataset_dir=sentiment_dataset_dir, max_vocab=3000)
    image_config = ImageAnalysisConfig(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        vision_model=os.getenv("OLLAMA_VISION_MODEL", "llava"),
        text_model=os.getenv("OLLAMA_MODEL", "llama3"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90")),
    )
    chatbot_config = ChatbotConfig(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90")),
    )

    @app.get("/api/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.get("/api/market-analysis")
    def market_analysis() -> tuple[dict, int]:
        days_param = request.args.get("days")
        days = int(days_param) if days_param and days_param.isdigit() else None

        try:
            payload = build_market_analysis_payload(config, days=days)
            return payload, 200
        except (FileNotFoundError, ValueError) as err:
            return {"error": str(err)}, 400
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.get("/api/modules")
    def modules() -> tuple[dict, int]:
        return (
            {
                "modules": [
                    {"id": "market-analysis", "title": "Market Analysis", "status": "active"},
                    {"id": "price-prediction", "title": "Price Prediction", "status": "active"},
                    {"id": "direction-prediction", "title": "Direction Prediction", "status": "active"},
                    {"id": "sentiment-analysis", "title": "Sentiment Analysis", "status": "active"},
                    {"id": "risk-clustering", "title": "Risk Clustering", "status": "active"},
                    {"id": "image-analysis", "title": "Image Intelligence", "status": "active"},
                ]
            },
            200,
        )

    @app.get("/api/price-prediction")
    def price_prediction() -> tuple[dict, int]:
        horizon_param = request.args.get("horizon")
        lookback_param = request.args.get("lookback")
        frequency = (request.args.get("frequency") or "D").upper()

        horizon = int(horizon_param) if horizon_param and horizon_param.isdigit() else None
        lookback = int(lookback_param) if lookback_param and lookback_param.isdigit() else None

        try:
            payload = build_price_prediction_payload(
                price_config,
                horizon=horizon,
                lookback=lookback,
                frequency=frequency,
            )
            return payload, 200
        except (FileNotFoundError, ValueError) as err:
            return {"error": str(err)}, 400
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.get("/api/direction-prediction")
    def direction_prediction() -> tuple[dict, int]:
        lookback_param = request.args.get("lookback")
        threshold_param = request.args.get("threshold")
        horizon_param = request.args.get("horizon")

        lookback = int(lookback_param) if lookback_param and lookback_param.isdigit() else None
        horizon = int(horizon_param) if horizon_param and horizon_param.isdigit() else None

        try:
            threshold = float(threshold_param) if threshold_param is not None else 0.5
        except ValueError:
            threshold = 0.5

        try:
            payload = build_direction_prediction_payload(
                direction_config,
                lookback=lookback,
                threshold=threshold,
                horizon=horizon,
            )
            return payload, 200
        except (FileNotFoundError, ValueError) as err:
            return {"error": str(err)}, 400
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.post("/api/sentiment-analysis")
    def sentiment_analysis() -> tuple[dict, int]:
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", ""))

        try:
            result = analyze_sentiment(text, config=sentiment_config)
            return result, 200
        except ValueError as err:
            return {"error": str(err)}, 400
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.get("/api/risk-clustering")
    def risk_clustering() -> tuple[dict, int]:
        days_param = request.args.get("days")
        seed_param = request.args.get("seed")

        days = int(days_param) if days_param and days_param.isdigit() else None
        seed = int(seed_param) if seed_param and seed_param.lstrip("-").isdigit() else 42

        try:
            payload = build_risk_clustering_payload(risk_config, days=days, seed=seed)
            return payload, 200
        except (FileNotFoundError, ValueError) as err:
            return {"error": str(err)}, 400
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.post("/api/chatbot")
    def chatbot() -> tuple[dict, int]:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", ""))
        history = payload.get("history")
        safe_history = history if isinstance(history, list) else None

        try:
            result = ask_ollama(message, config=chatbot_config, history=safe_history)
            return result, 200
        except ValueError as err:
            return {"error": str(err)}, 400
        except RuntimeError as err:
            return {"error": str(err)}, 502
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    @app.post("/api/image-analysis")
    def image_analysis() -> tuple[dict, int]:
        uploaded = request.files.get("image")
        question = request.form.get("question", "")

        if uploaded is None:
            return {"error": "Missing image file. Send multipart form-data with field 'image'."}, 400

        try:
            image_bytes = uploaded.read()
            payload = analyze_image_news(
                image_bytes=image_bytes,
                filename=uploaded.filename or "uploaded-image",
                config=image_config,
                question=question,
            )
            return payload, 200
        except ValueError as err:
            return {"error": str(err)}, 400
        except RuntimeError as err:
            return {"error": str(err)}, 502
        except Exception as err:
            return {"error": f"Unexpected server error: {err}"}, 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
