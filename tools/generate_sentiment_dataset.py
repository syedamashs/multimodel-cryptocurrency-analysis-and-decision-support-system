from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "dataset-sentiment" / "crypto_sentiment_dataset.csv"


SUBJECTS = [
    "Bitcoin",
    "Ethereum",
    "Solana",
    "altcoins",
    "the crypto market",
    "institutional investors",
    "retail traders",
    "ETF flows",
    "on-chain activity",
    "market sentiment",
]

POS_EVENTS = [
    "shows strong adoption",
    "breaks a key resistance level",
    "records bullish momentum",
    "attracts institutional demand",
    "posts healthy on-chain growth",
    "signals a possible rally",
    "prints a higher high pattern",
    "receives positive regulatory clarity",
    "sees increasing wallet activity",
    "gains strong market confidence",
]

NEG_EVENTS = [
    "faces bearish pressure",
    "breaks below major support",
    "shows liquidation stress",
    "triggers market fear",
    "records weak momentum",
    "signals downside risk",
    "sees regulatory uncertainty",
    "suffers from security concerns",
    "shows declining on-chain activity",
    "experiences heavy selloff",
]

NEU_EVENTS = [
    "moves in a narrow range",
    "shows mixed market signals",
    "remains unchanged in trend",
    "trades near recent averages",
    "shows balanced buy and sell pressure",
    "stays in consolidation phase",
    "has no clear breakout signal",
    "reflects neutral market mood",
    "shows stable participation",
    "prints sideways action",
]

POS_CONTEXT = [
    "while volume supports the move.",
    "and traders expect continuation.",
    "with improving macro sentiment.",
    "as confidence returns to the market.",
    "and risk appetite is rising.",
]

NEG_CONTEXT = [
    "while volatility remains elevated.",
    "and traders cut risk exposure.",
    "as macro uncertainty persists.",
    "with weak buyer interest.",
    "and downside momentum builds.",
]

NEU_CONTEXT = [
    "while traders wait for confirmation.",
    "and volume stays average.",
    "as the market searches for direction.",
    "with no strong catalyst in sight.",
    "and short-term conviction stays low.",
]


def build_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    for subject in SUBJECTS:
        for event in POS_EVENTS:
            for context in POS_CONTEXT:
                rows.append((f"{subject} {event} {context}", "Positive"))

    for subject in SUBJECTS:
        for event in NEG_EVENTS:
            for context in NEG_CONTEXT:
                rows.append((f"{subject} {event} {context}", "Negative"))

    for subject in SUBJECTS:
        for event in NEU_EVENTS:
            for context in NEU_CONTEXT:
                rows.append((f"{subject} {event} {context}", "Neutral"))

    # Add social-media style short texts.
    rows.extend(
        [
            ("BTC breakout looks strong, bulls in control, likely upside continuation.", "Positive"),
            ("Major dump in crypto, fear increasing and support broken.", "Negative"),
            ("Market is flat today, no strong directional edge.", "Neutral"),
            ("ETH adoption rising and buyers stepping in aggressively.", "Positive"),
            ("Heavy liquidations and panic selling across altcoins.", "Negative"),
            ("Traders are waiting for Fed update before taking positions.", "Neutral"),
        ]
        * 120
    )

    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
