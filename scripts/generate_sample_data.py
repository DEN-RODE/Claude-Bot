"""Generate the synthetic OHLCV series used by the examples and the tests.

Synthetic rather than recorded, for two reasons. Exchange data carries licensing
terms that vary by venue, and a committed snapshot of a real market quietly
becomes the thing every strategy is tuned against. A generated series with a
fixed seed is reproducible, carries no terms, and makes it obvious that the
example output is a demonstration of the harness and not a claim about returns.

The generator produces a regime-switching random walk: trending, ranging and
high-volatility stretches follow one another, so the regime classifier has
something to classify and per-regime statistics are not all zero.

Usage::

    python scripts/generate_sample_data.py --out data/sample_BTCUSDT_1h.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

REGIME_PLAN = [
    ("trend_up", 180, 0.0012, 0.004),
    ("range", 150, 0.0000, 0.003),
    ("high_vol", 90, -0.0005, 0.011),
    ("trend_down", 170, -0.0010, 0.005),
    ("range", 160, 0.0001, 0.0025),
    ("trend_up", 200, 0.0009, 0.0045),
    ("high_vol", 80, 0.0002, 0.010),
    ("range", 170, -0.0001, 0.0028),
]

BAR_MS = 3_600_000


def generate(seed: int = 20260101, start_price: float = 42_000.0) -> list[dict]:
    """Return a list of OHLCV bars as dicts."""
    rng = random.Random(seed)
    rows: list[dict] = []
    price = start_price
    timestamp = 1_735_689_600_000  # 2025-01-01T00:00:00Z, arbitrary and fixed

    for _label, length, drift, volatility in REGIME_PLAN:
        for _ in range(length):
            shock = rng.gauss(drift, volatility)
            open_price = price
            close_price = max(0.01, open_price * (1 + shock))

            # Wicks scale with the bar's own body and the regime volatility, so
            # high-volatility stretches get genuinely wider ranges rather than
            # the same range with a different close.
            span = abs(close_price - open_price) + open_price * volatility * rng.uniform(
                0.3, 1.2
            )
            high = max(open_price, close_price) + span * rng.uniform(0.0, 0.6)
            low = min(open_price, close_price) - span * rng.uniform(0.0, 0.6)
            low = max(0.01, low)

            base_volume = 800 + 4000 * abs(shock) / max(volatility, 1e-9)
            volume = round(base_volume * rng.uniform(0.6, 1.6), 2)

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                }
            )
            price = close_price
            timestamp += BAR_MS

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> dict[str, list]:
    """Read a generated CSV back into named series."""
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "timestamps": [int(r["timestamp"]) for r in rows],
        "opens": [float(r["open"]) for r in rows],
        "highs": [float(r["high"]) for r in rows],
        "lows": [float(r["low"]) for r in rows],
        "closes": [float(r["close"]) for r in rows],
        "volumes": [float(r["volume"]) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sample_BTCUSDT_1h.csv"),
        help="output CSV path",
    )
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()

    rows = generate(seed=args.seed)
    write_csv(rows, args.out)
    closes = [r["close"] for r in rows]
    print(f"wrote {len(rows)} bars to {args.out}")
    print(
        f"price range {min(closes):,.0f} to {max(closes):,.0f}, "
        f"final {closes[-1]:,.0f}"
    )


if __name__ == "__main__":
    main()
