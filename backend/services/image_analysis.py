from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import easyocr
import numpy as np
from PIL import Image
from io import BytesIO

from services.sentiment_analysis import SentimentConfig, analyze_sentiment


# Initialize OCR reader (lazy loaded on first use)
_ocr_reader = None


def _get_ocr_reader():
    """Lazy load OCR reader to avoid initialization overhead."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


@dataclass
class ImageAnalysisConfig:
    ollama_host: str = "http://127.0.0.1:11434"
    text_model: str = "llama3"
    timeout_seconds: int = 120


def _ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

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
    """Extract text from image using EasyOCR (pure Python, no external binaries needed)."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
        
        # Convert PIL Image to numpy array for EasyOCR
        image_array = np.array(image)
        
        reader = _get_ocr_reader()
        results = reader.readtext(image_array)
        
        # Combine all detected text
        extracted = "\n".join([text for (_, text, _) in results])
        return extracted.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from image using OCR: {str(e)}") from e



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
    sentiment_config: SentimentConfig | None = None,
) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("Image file is empty")

    if len(image_bytes) > 12 * 1024 * 1024:
        raise ValueError("Image too large. Keep image under 12 MB")

    extracted_text = _extract_text_from_image(image_bytes, config=config)
    if not extracted_text:
        raise RuntimeError("No text could be extracted from image")

    if sentiment_config is None:
        sentiment_config = SentimentConfig(confidence_floor=0.45)
    sentiment_payload = analyze_sentiment(extracted_text, config=sentiment_config)

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
