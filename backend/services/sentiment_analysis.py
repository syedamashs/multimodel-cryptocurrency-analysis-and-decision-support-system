from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SentimentConfig:
    confidence_floor: float = 0.45
    dataset_dir: Path | None = None
    max_vocab: int = 3000


POSITIVE_LEXICON: dict[str, float] = {
    "bullish": 2.4,
    "rally": 2.0,
    "surge": 2.0,
    "adoption": 1.8,
    "growth": 1.5,
    "strong": 1.2,
    "breakout": 2.2,
    "recovery": 1.5,
    "optimistic": 1.7,
    "gain": 1.4,
    "institutional": 1.5,
    "approved": 1.8,
    "upgrade": 1.3,
    "secure": 1.1,
    "partnership": 1.4,
    "momentum": 1.3,
    "support": 1.0,
    "buy": 1.6,
    "green": 1.0,
    "profit": 1.4,
    "boom": 1.8,
    "soar": 2.0,
    "bull": 2.2,
    "upside": 1.6,
    "win": 1.5,
    "excellent": 1.7,
    "amazing": 1.6,
    "success": 1.7,
    "opportunity": 1.5,
    "bullrun": 2.3,
    "confidence": 1.4,
    "breakthrough": 1.8,
    "record": 1.5,
    "pump": 1.4,
}

NEGATIVE_LEXICON: dict[str, float] = {
    "bearish": -2.4,
    "crash": -2.5,
    "dump": -2.1,
    "selloff": -2.2,
    "fear": -1.7,
    "hack": -2.6,
    "ban": -2.2,
    "lawsuit": -1.9,
    "liquidation": -2.1,
    "weak": -1.2,
    "drop": -1.6,
    "decline": -1.5,
    "uncertain": -1.4,
    "volatility": -0.8,
    "resistance": -1.0,
    "sell": -1.6,
    "red": -1.0,
    "loss": -1.4,
    "fraud": -2.5,
    "recession": -1.8,
    "scam": -3.0,
    "poor": -1.8,
    "bad": -1.6,
    "terrible": -2.0,
    "awful": -2.2,
    "horrible": -2.3,
    "doomed": -2.5,
    "disaster": -2.4,
    "collapse": -2.6,
    "fail": -2.1,
    "failure": -2.2,
    "risk": -1.3,
    "risky": -1.5,
    "danger": -1.7,
    "dangerous": -1.9,
    "prison": -2.4,
    "jail": -2.3,
    "arrest": -2.2,
    "indictment": -2.3,
    "investigation": -1.5,
    "scandal": -2.2,
    "ponzi": -3.0,
    "scheme": -2.4,
    "suspicious": -1.6,
    "doubt": -1.4,
    "concerned": -1.3,
    "worried": -1.5,
    "panic": -2.0,
    "bear": -1.8,
    "downside": -1.5,
    "correction": -1.2,
    "dump": -2.2,
    "plunge": -2.3,
    "slump": -1.8,
    "loss": -1.8,
}

NEGATIONS = {"not", "never", "no", "none", "hardly", "barely"}
INTENSIFIERS = {"very": 1.35, "extremely": 1.55, "highly": 1.45, "strongly": 1.25, "slightly": 0.75}

ASPECT_KEYWORDS = {
    "Adoption": {"adoption", "users", "institutional", "etf", "mainstream", "merchant", "acceptance"},
    "Regulation": {"regulation", "ban", "legal", "sec", "compliance", "policy", "approved"},
    "Technology": {"upgrade", "network", "scaling", "protocol", "security", "layer", "performance"},
    "Market Mood": {"bullish", "bearish", "rally", "crash", "momentum", "volatility", "trend"},
    "Risk": {"hack", "fraud", "liquidation", "uncertain", "lawsuit", "recession", "risk"},
}

TOKEN_REGEX = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
LABEL_TO_INT = {"Negative": 0, "Neutral": 1, "Positive": 2}
INT_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_REGEX.findall(text)]


def _sentence_split(text: str) -> list[str]:
    raw = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return raw if raw else [text.strip()]


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset-sentiment"


def _dataset_signature(dataset_dir: Path) -> tuple[str, int, int]:
    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No sentiment CSV found in {dataset_dir}. Place a file like crypto_sentiment_dataset.csv"
        )
    path = files[0]
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime), int(stat.st_size))


def _load_dataset_rows(path: str) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("Sentiment dataset must have 'text' and 'label' columns")

        for row in reader:
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip().title()
            if not text or label not in LABEL_TO_INT:
                continue
            texts.append(text)
            labels.append(LABEL_TO_INT[label])

    if len(texts) < 300:
        raise ValueError("Sentiment dataset too small. Need at least 300 labeled rows")

    return texts, np.array(labels, dtype=int)


def _build_vocab(texts: list[str], max_vocab: int) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(_tokenize(text))

    tokens = [token for token, cnt in counts.most_common(max_vocab) if cnt >= 2]
    return {token: idx for idx, token in enumerate(tokens)}


def _vectorize_counts(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, text in enumerate(texts):
        for token in _tokenize(text):
            idx = vocab.get(token)
            if idx is not None:
                x[i, idx] += 1.0
    return x


def _train_test_split(texts: list[str], y: np.ndarray, seed: int = 42) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)

    split = int(0.8 * len(texts))
    tr_idx = idx[:split]
    te_idx = idx[split:]

    x_train = [texts[i] for i in tr_idx]
    x_test = [texts[i] for i in te_idx]
    y_train = y[tr_idx]
    y_test = y[te_idx]
    return x_train, x_test, y_train, y_test


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s: list[float] = []
    for cls in range(n_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _tfidf_transform(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = np.sum(x_train > 0, axis=0)
    idf = np.log((x_train.shape[0] + 1.0) / (df + 1.0)) + 1.0

    x_train_tfidf = x_train * idf
    x_test_tfidf = x_test * idf

    train_norm = np.linalg.norm(x_train_tfidf, axis=1, keepdims=True) + 1e-9
    test_norm = np.linalg.norm(x_test_tfidf, axis=1, keepdims=True) + 1e-9
    return x_train_tfidf / train_norm, x_test_tfidf / test_norm, idf


def _train_multinomial_nb(x_train: np.ndarray, y_train: np.ndarray, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0
    class_log_prior = np.zeros(n_classes, dtype=float)
    feature_log_prob = np.zeros((n_classes, x_train.shape[1]), dtype=float)

    for cls in range(n_classes):
        x_cls = x_train[y_train == cls]
        class_log_prior[cls] = np.log(max(len(x_cls), 1) / len(x_train))
        token_sum = np.sum(x_cls, axis=0) + alpha
        token_total = np.sum(token_sum)
        feature_log_prob[cls] = np.log(token_sum / max(token_total, 1e-9))

    return class_log_prior, feature_log_prob


def _predict_nb_proba(x: np.ndarray, class_log_prior: np.ndarray, feature_log_prob: np.ndarray) -> np.ndarray:
    logits = class_log_prior[None, :] + x @ feature_log_prob.T
    return _softmax(logits)


def _train_centroid_classifier(x_train_tfidf: np.ndarray, y_train: np.ndarray, n_classes: int = 3) -> np.ndarray:
    centroids = np.zeros((n_classes, x_train_tfidf.shape[1]), dtype=float)
    for cls in range(n_classes):
        rows = x_train_tfidf[y_train == cls]
        if len(rows) > 0:
            centroid = np.mean(rows, axis=0)
            centroids[cls] = centroid / (np.linalg.norm(centroid) + 1e-9)
    return centroids


def _predict_centroid_proba(x_tfidf: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    sims = x_tfidf @ centroids.T
    return _softmax(sims)


def _token_score(tokens: list[str]) -> tuple[float, list[dict[str, Any]]]:
    score = 0.0
    impacts: list[dict[str, Any]] = []

    for i, token in enumerate(tokens):
        base = 0.0
        if token in POSITIVE_LEXICON:
            base = POSITIVE_LEXICON[token]
        elif token in NEGATIVE_LEXICON:
            base = NEGATIVE_LEXICON[token]

        if base == 0.0:
            continue

        modifier = 1.0
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            modifier *= INTENSIFIERS[tokens[i - 1]]
        if i > 0 and tokens[i - 1] in NEGATIONS:
            modifier *= -1.0
        if i > 1 and tokens[i - 2] in NEGATIONS:
            modifier *= -1.0

        impact = base * modifier
        score += impact
        impacts.append(
            {
                "token": token,
                "impact": float(impact),
                "polarity": "positive" if impact > 0 else "negative",
            }
        )

    return score, impacts


def _label_from_score(score: float, pos_thr: float = 0.10, neg_thr: float = -0.10) -> str:
    if score >= pos_thr:
        return "Positive"
    if score <= neg_thr:
        return "Negative"
    return "Neutral"


def _lexicon_predict_proba(texts: list[str]) -> np.ndarray:
    probs = np.zeros((len(texts), 3), dtype=float)
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        raw_score, _ = _token_score(tokens)
        score = float(np.tanh(raw_score / max(len(tokens), 1)))

        # More aggressive thresholds for detecting strong signals
        if score >= 0.20:
            p = np.array([0.05, 0.15, 0.80], dtype=float)
        elif score >= 0.10:
            p = np.array([0.10, 0.25, 0.65], dtype=float)
        elif score <= -0.25:
            p = np.array([0.80, 0.15, 0.05], dtype=float)
        elif score <= -0.10:
            p = np.array([0.65, 0.25, 0.10], dtype=float)
        else:
            p = np.array([0.22, 0.56, 0.22], dtype=float)

        # Strengthen prediction based on magnitude
        magnitude = min(abs(score), 0.95)
        if score > 0:
            p[2] += 0.22 * magnitude
            p[0] -= 0.11 * magnitude
        elif score < 0:
            p[0] += 0.22 * magnitude
            p[2] -= 0.11 * magnitude

        p = np.clip(p, 1e-4, None)
        p = p / np.sum(p)
        probs[i] = p

    return probs


def _aspect_sentiment(sentences: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        aspect_scores: list[float] = []
        mentions = 0

        for sentence in sentences:
            sent_tokens = _tokenize(sentence)
            if not sent_tokens:
                continue

            if any(token in keywords for token in sent_tokens):
                mentions += 1
                sentence_score, _ = _token_score(sent_tokens)
                normalized = float(np.tanh(sentence_score / max(len(sent_tokens), 1)))
                aspect_scores.append(normalized)

        score = float(np.mean(aspect_scores)) if aspect_scores else 0.0
        output.append(
            {
                "aspect": aspect,
                "score": score,
                "label": _label_from_score(score),
                "mentions": mentions,
            }
        )

    return output


@lru_cache(maxsize=4)
def _trained_bundle(path: str, mtime: int, size: int, max_vocab: int) -> dict[str, Any]:
    _ = (mtime, size)
    texts, y = _load_dataset_rows(path)
    x_train_texts, x_test_texts, y_train, y_test = _train_test_split(texts, y, seed=42)

    vocab = _build_vocab(x_train_texts, max_vocab=max_vocab)
    x_train_counts = _vectorize_counts(x_train_texts, vocab)
    x_test_counts = _vectorize_counts(x_test_texts, vocab)

    class_log_prior, feature_log_prob = _train_multinomial_nb(x_train_counts, y_train)
    nb_proba_test = _predict_nb_proba(x_test_counts, class_log_prior, feature_log_prob)
    nb_pred = np.argmax(nb_proba_test, axis=1)

    x_train_tfidf, x_test_tfidf, idf = _tfidf_transform(x_train_counts, x_test_counts)
    centroids = _train_centroid_classifier(x_train_tfidf, y_train)
    centroid_proba_test = _predict_centroid_proba(x_test_tfidf, centroids)
    centroid_pred = np.argmax(centroid_proba_test, axis=1)

    lexicon_proba_test = _lexicon_predict_proba(x_test_texts)
    lexicon_pred = np.argmax(lexicon_proba_test, axis=1)

    metrics = {
        "Multinomial Naive Bayes": {
            "accuracy": float(np.mean(nb_pred == y_test)),
            "macroF1": _macro_f1(y_test, nb_pred),
        },
        "TF-IDF Centroid": {
            "accuracy": float(np.mean(centroid_pred == y_test)),
            "macroF1": _macro_f1(y_test, centroid_pred),
        },
        "Lexicon Weighted": {
            "accuracy": float(np.mean(lexicon_pred == y_test)),
            "macroF1": _macro_f1(y_test, lexicon_pred),
        },
    }

    return {
        "vocab": vocab,
        "idf": idf,
        "class_log_prior": class_log_prior,
        "feature_log_prob": feature_log_prob,
        "centroids": centroids,
        "metrics": metrics,
        "dataset_rows": len(texts),
    }


def _single_text_model_outputs(text: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    vocab = bundle["vocab"]
    x_count = _vectorize_counts([text], vocab)

    nb_proba = _predict_nb_proba(x_count, bundle["class_log_prior"], bundle["feature_log_prob"])[0]
    nb_label = INT_TO_LABEL[int(np.argmax(nb_proba))]
    nb_score = float(nb_proba[2] - nb_proba[0])

    idf = bundle["idf"]
    x_tfidf = x_count * idf
    x_tfidf = x_tfidf / (np.linalg.norm(x_tfidf, axis=1, keepdims=True) + 1e-9)
    centroid_proba = _predict_centroid_proba(x_tfidf, bundle["centroids"])[0]
    centroid_label = INT_TO_LABEL[int(np.argmax(centroid_proba))]
    centroid_score = float(centroid_proba[2] - centroid_proba[0])

    lexicon_proba = _lexicon_predict_proba([text])[0]
    lexicon_label = INT_TO_LABEL[int(np.argmax(lexicon_proba))]
    lexicon_score = float(lexicon_proba[2] - lexicon_proba[0])

    metrics = bundle["metrics"]
    return [
        {
            "model": "Multinomial Naive Bayes",
            "label": nb_label,
            "score": nb_score,
            "confidence": float(np.max(nb_proba)),
            "validationAccuracy": float(metrics["Multinomial Naive Bayes"]["accuracy"]),
        },
        {
            "model": "TF-IDF Centroid",
            "label": centroid_label,
            "score": centroid_score,
            "confidence": float(np.max(centroid_proba)),
            "validationAccuracy": float(metrics["TF-IDF Centroid"]["accuracy"]),
        },
        {
            "model": "Lexicon Weighted",
            "label": lexicon_label,
            "score": lexicon_score,
            "confidence": float(np.max(lexicon_proba)),
            "validationAccuracy": float(metrics["Lexicon Weighted"]["accuracy"]),
        },
    ]


def analyze_sentiment(text: str, config: SentimentConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = SentimentConfig()

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty. Please provide market text, tweet, or news paragraph.")

    dataset_dir = config.dataset_dir or _default_dataset_dir()
    path, mtime, size = _dataset_signature(dataset_dir)
    bundle = _trained_bundle(path, mtime, size, config.max_vocab)

    model_outputs = _single_text_model_outputs(cleaned, bundle)
    best_model = max(model_outputs, key=lambda x: x["validationAccuracy"])

    # Weight all models equally (not by accuracy) for better diversity
    # Lexicon is good at catching explicit signals, NB at learned patterns, Centroid at semantic similarity
    weighted_score = sum(m["score"] for m in model_outputs) / len(model_outputs)

    label = _label_from_score(weighted_score)
    # Calculate confidence based on agreement and magnitude
    score_magnitudes = [abs(m["score"]) for m in model_outputs]
    agreement_strength = float(np.mean(score_magnitudes))
    confidence = float(max(config.confidence_floor, min(0.99, agreement_strength + 0.35)))

    sentences = _sentence_split(cleaned)
    sentence_rows: list[dict[str, Any]] = []
    all_impacts: list[dict[str, Any]] = []

    for sentence in sentences:
        tokens = _tokenize(sentence)
        raw_score, impacts = _token_score(tokens)
        norm_score = float(np.tanh(raw_score / max(len(tokens), 1)))
        sent_label = _label_from_score(norm_score)
        sentence_rows.append({"sentence": sentence, "score": norm_score, "label": sent_label})
        all_impacts.extend(impacts)

    pos_count = int(sum(1 for s in sentence_rows if s["label"] == "Positive"))
    neg_count = int(sum(1 for s in sentence_rows if s["label"] == "Negative"))
    neu_count = int(sum(1 for s in sentence_rows if s["label"] == "Neutral"))

    token_map: dict[str, float] = {}
    for item in all_impacts:
        token_map[item["token"]] = token_map.get(item["token"], 0.0) + float(item["impact"])

    token_impacts = [
        {
            "token": token,
            "impact": impact,
            "polarity": "positive" if impact > 0 else "negative",
        }
        for token, impact in token_map.items()
    ]
    token_impacts = sorted(token_impacts, key=lambda x: abs(x["impact"]), reverse=True)[:12]

    aspects = _aspect_sentiment(sentences)

    if label == "Positive":
        recommendation = "Sentiment supports bullish bias. Combine with modules 2 and 3 before BUY decision."
    elif label == "Negative":
        recommendation = "Sentiment signals caution. Combine with risk and direction modules before entry."
    else:
        recommendation = "Mixed sentiment. Wait for confirmation from trend and direction signals."

    return {
        "overview": {
            "chars": len(cleaned),
            "words": len(_tokenize(cleaned)),
            "sentences": len(sentences),
            "trainingRows": int(bundle["dataset_rows"]),
        },
        "result": {
            "label": label,
            "score": float(weighted_score),
            "confidence": confidence,
            "recommendation": recommendation,
        },
        "bestModel": str(best_model["model"]),
        "models": model_outputs,
        "charts": {
            "sentenceDistribution": [
                {"label": "Positive", "count": pos_count},
                {"label": "Neutral", "count": neu_count},
                {"label": "Negative", "count": neg_count},
            ],
            "modelComparison": [
                {
                    "model": m["model"],
                    "score": float(m["score"]),
                    "confidence": float(m["confidence"]),
                }
                for m in model_outputs
            ],
            "aspectSentiment": aspects,
            "tokenImpact": token_impacts,
            "sentenceTimeline": [
                {
                    "index": idx + 1,
                    "score": row["score"],
                    "label": row["label"],
                }
                for idx, row in enumerate(sentence_rows)
            ],
        },
        "sentenceAnalysis": sentence_rows,
    }
