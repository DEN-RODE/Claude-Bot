"""Market regime classification.

A strategy's headline win rate is an average over conditions that are not
comparable. Breakout strategies earn their keep in trends and bleed in ranges;
mean-reversion does the reverse. Reporting one number for both hides the only
thing worth knowing, so every bar is labelled and results are attributed.

Six regimes, decided in this order:

``HIGH_VOL``
    ATR above its 80th percentile over the trailing window. Volatility
    dominates whatever the trend is doing, so it is checked first.
``DORMANT``
    ATR below its 30th percentile with RSI in the 40 to 60 band. Nothing is
    happening and most signals here are noise.
``BULL_TREND`` / ``BEAR_TREND``
    Reference asset trending in that direction with ADX above 25 and ATR above
    its median.
``BULL_RANGE`` / ``BEAR_RANGE``
    Same directional bias, but without the strength to qualify as a trend.

Percentiles are computed over a trailing window rather than the whole series on
purpose: using full-series percentiles would let a bar's classification depend
on volatility that had not happened yet, which is look-ahead bias smuggled in
through the labels.
"""

from __future__ import annotations

from typing import Sequence

REGIMES = (
    "BULL_TREND",
    "BEAR_TREND",
    "BULL_RANGE",
    "BEAR_RANGE",
    "DORMANT",
    "HIGH_VOL",
)

LOOKBACK = 50
MIN_SAMPLES = 10
ADX_TREND_THRESHOLD = 25.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty series")
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def classify_regime(
    i: int,
    closes: Sequence[float],
    reference_closes: Sequence[float],
    indicators: dict,
    lookback: int = LOOKBACK,
) -> str | None:
    """Return the regime for bar ``i``, or ``None`` during warm-up.

    ``reference_closes`` supplies the directional bias. In crypto this is
    usually BTC, since altcoins mostly follow it; pass ``closes`` itself to
    classify an asset on its own terms.
    """
    if i < lookback:
        return None

    atr_series = indicators.get("atr14") or []
    if i >= len(atr_series) or atr_series[i] is None:
        return None

    window = [
        atr_series[j]
        for j in range(max(0, i - lookback), i + 1)
        if j < len(atr_series) and atr_series[j] is not None
    ]
    if len(window) < MIN_SAMPLES:
        return None

    ordered = sorted(window)
    p30 = _percentile(ordered, 0.30)
    p50 = _percentile(ordered, 0.50)
    p80 = _percentile(ordered, 0.80)
    current_atr = atr_series[i]

    if current_atr > p80:
        return "HIGH_VOL"

    rsi_value = indicators.get("rsi14", [None] * (i + 1))[i]
    if current_atr < p30 and rsi_value is not None and 40 <= rsi_value <= 60:
        return "DORMANT"

    # Directional bias from a 9 versus 21 bar average of the reference asset.
    if i < 21 or i >= len(reference_closes):
        return "DORMANT"
    fast = sum(reference_closes[i - 8 : i + 1]) / 9
    slow = sum(reference_closes[i - 20 : i + 1]) / 21

    adx_series = indicators.get("adx", [])
    adx_value = adx_series[i] if i < len(adx_series) else 0.0
    is_ranging = current_atr < p50
    is_trending = adx_value > ADX_TREND_THRESHOLD and not is_ranging

    if fast > slow:
        return "BULL_TREND" if is_trending else "BULL_RANGE"
    if fast < slow:
        return "BEAR_TREND" if is_trending else "BEAR_RANGE"
    return "DORMANT"
