from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatbotConfig:
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = "llama3"
    timeout_seconds: int = 90


SYSTEM_PROMPT = (
    "You are CryptoPilot, an assistant inside a crypto decision support dashboard. "
    "Keep responses clear, practical, and concise. "
    "Use plain language, avoid financial guarantees, and mention uncertainty when needed."
)

CRYPTO_KEYWORDS = {
    "crypto",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "blockchain",
    "coin",
    "token",
    "altcoin",
    "defi",
    "nft",
    "marketcap",
    "wallet",
    "bullish",
    "bearish",
    "volatility",
    "price",
    "trading",
    "buy",
    "sell",
    "ohlc",
    "sentiment",
    "cluster",
    "risk",
}

DASHBOARD_CONTEXT_KEYWORDS = {
    "dashboard",
    "insight",
    "insights",
    "module",
    "modules",
    "prediction",
    "forecast",
    "direction",
    "sentiment",
    "clustering",
    "risk",
    "analysis",
    "buy",
    "sell",
    "hold",
    "recommendation",
    "summarize",
    "summary",
}


def _is_crypto_relevant(text: str) -> bool:
    lowered = text.lower()
    has_crypto_term = any(keyword in lowered for keyword in CRYPTO_KEYWORDS)
    has_dashboard_term = any(keyword in lowered for keyword in DASHBOARD_CONTEXT_KEYWORDS)
    return has_crypto_term or has_dashboard_term


def _build_prompt(user_message: str, history: list[dict[str, str]] | None = None) -> str:
    sections: list[str] = [f"System: {SYSTEM_PROMPT}"]

    if history:
        trimmed = history[-6:]
        for item in trimmed:
            role = item.get("role", "user").strip().lower()
            content = item.get("content", "").strip()
            if not content:
                continue
            normalized_role = "Assistant" if role == "assistant" else "User"
            sections.append(f"{normalized_role}: {content}")

    sections.append(f"User: {user_message.strip()}")
    sections.append("Assistant:")
    return "\n".join(sections)


def ask_ollama(
    user_message: str,
    config: ChatbotConfig,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    text = user_message.strip()
    if not text:
        raise ValueError("Message is empty")

    if not _is_crypto_relevant(text):
        return {
            "answer": "Pls ask only relevent to Crypto.",
            "model": config.model,
            "done": True,
        }

    payload = {
        "model": config.model,
        "prompt": _build_prompt(text, history=history),
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    url = f"{config.ollama_host.rstrip('/')}/api/generate"

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {err.code}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(
            "Could not connect to Ollama. Ensure Ollama is running at "
            f"{config.ollama_host} and model '{config.model}' is available."
        ) from err
    except TimeoutError as err:
        raise RuntimeError("Ollama request timed out. Try a shorter question.") from err
    except json.JSONDecodeError as err:
        raise RuntimeError("Invalid response from Ollama") from err

    answer = str(parsed.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response")

    return {
        "answer": answer,
        "model": str(parsed.get("model", config.model)),
        "done": bool(parsed.get("done", True)),
    }
