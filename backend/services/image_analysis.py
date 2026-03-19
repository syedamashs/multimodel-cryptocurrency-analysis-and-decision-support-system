from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from services.sentiment_analysis import SentimentConfig, analyze_sentiment


@dataclass
class ImageAnalysisConfig:
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model: str = "llava"
    text_model: str = "llama3"
    timeout_seconds: int = 120


def _normalize_model_name(model_name: str) -> str:
    return model_name.split(":", 1)[0].strip().lower()


def _list_ollama_models(host: str, timeout_seconds: int) -> list[str]:
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/tags",
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
    except Exception:
        return []

    models = parsed.get("models", [])
    names: list[str] = []
    for item in models:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _is_vision_model_name(name: str) -> bool:
    lower = _normalize_model_name(name)
    return bool(re.search(r"vision|llava|moondream|bakllava|minicpm", lower))


def _select_vision_model(config: ImageAnalysisConfig) -> str:
    installed = _list_ollama_models(config.ollama_host, timeout_seconds=max(8, config.timeout_seconds // 3))
    if not installed:
        # Fallback to configured model if tags endpoint is unavailable.
        return config.vision_model

    normalized_map = {_normalize_model_name(name): name for name in installed}

    preferred = _normalize_model_name(config.vision_model)
    if preferred in normalized_map:
        return normalized_map[preferred]

    for candidate in ["llava", "llama3.2-vision", "moondream", "bakllava"]:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for name in installed:
        if _is_vision_model_name(name):
            return name

    installed_list = ", ".join(installed)
    raise RuntimeError(
        "No vision model found in Ollama. Image analysis needs a vision-capable model. "
        "Run one of: `ollama pull llava` or `ollama pull llama3.2-vision`. "
        f"Installed models: [{installed_list}]"
    )


def _ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
    images: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if images:
        payload["images"] = images

    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {err.code}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. Ensure Ollama is running and models are pulled."
        ) from err
    except TimeoutError as err:
        raise RuntimeError("Ollama request timed out") from err
    except json.JSONDecodeError as err:
        raise RuntimeError("Invalid JSON response from Ollama") from err

    answer = str(parsed.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned empty response")
    return answer


def _extract_text_from_image(image_bytes: bytes, config: ImageAnalysisConfig) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    vision_model = _select_vision_model(config)

    prompt = (
        "Extract all visible text from this image. "
        "Return plain readable text only. "
        "Do not add explanations."
    )

    extracted = _ollama_generate(
        host=config.ollama_host,
        model=vision_model,
        prompt=prompt,
        timeout_seconds=config.timeout_seconds,
        images=[encoded],
    )

    return extracted.strip()


def _human_friendly_summary(extracted_text: str, sentiment_payload: dict[str, Any], config: ImageAnalysisConfig) -> str:
    result = sentiment_payload.get("result", {})
    label = str(result.get("label", "Neutral"))
    score = float(result.get("score", 0.0))
    confidence = float(result.get("confidence", 0.5))

    prompt = (
        "You are a helpful crypto analyst assistant. "
        "Given OCR-extracted crypto news text and sentiment metrics, provide a human-friendly explanation in 5-7 lines. "
        "Use plain language, no markdown tables, no hype, no guarantee wording.\n\n"
        f"Extracted Text:\n{extracted_text}\n\n"
        f"Sentiment Label: {label}\n"
        f"Sentiment Score: {score:.4f}\n"
        f"Confidence: {confidence:.2%}\n\n"
        "Output format:\n"
        "1) Short summary\n"
        "2) Market mood interpretation\n"
        "3) Practical caution / next checks"
    )

    return _ollama_generate(
        host=config.ollama_host,
        model=config.text_model,
        prompt=prompt,
        timeout_seconds=config.timeout_seconds,
    )


def analyze_image_news(
    image_bytes: bytes,
    filename: str,
    config: ImageAnalysisConfig,
    question: str | None = None,
) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("Image file is empty")

    if len(image_bytes) > 12 * 1024 * 1024:
        raise ValueError("Image too large. Keep image under 12 MB")

    extracted_text = _extract_text_from_image(image_bytes, config=config)
    if not extracted_text:
        raise RuntimeError("No text could be extracted from image")

    sentiment_payload = analyze_sentiment(extracted_text, config=SentimentConfig(confidence_floor=0.45))

    summary = _human_friendly_summary(extracted_text, sentiment_payload, config=config)

    qa_answer = ""
    question_clean = (question or "").strip()
    if question_clean:
        qa_prompt = (
            "Answer user question using only this extracted crypto text. "
            "If answer is uncertain, say uncertain. Keep it concise and readable.\n\n"
            f"Extracted Text:\n{extracted_text}\n\n"
            f"User Question: {question_clean}"
        )
        qa_answer = _ollama_generate(
            host=config.ollama_host,
            model=config.text_model,
            prompt=qa_prompt,
            timeout_seconds=config.timeout_seconds,
        )

    return {
        "file": {
            "name": filename,
            "sizeBytes": len(image_bytes),
        },
        "extractedText": extracted_text,
        "sentiment": sentiment_payload,
        "ollamaSummary": summary,
        "qa": {
            "question": question_clean,
            "answer": qa_answer,
        },
    }
