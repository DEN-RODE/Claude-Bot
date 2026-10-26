"""The strategy engine shared by backtest, paper trading and live execution.

Two rules live here, and both exist because of bugs that were expensive to find.

**Closed bars only.** :func:`evaluate_entry` drops the last element of every
series before computing anything, because in live trading the last bar is still
forming. An indicator computed mid-bar is not the indicator the backtest saw:
Supertrend and Donchian in particular flip several times inside an hourly bar
and then settle, so a live bot polling every thirty seconds fires a burst of
entries and exits on a signal that never existed once the bar closed.

**One registry.** Live and paper execution resolve strategies through the same
:mod:`claudebot.strategies` registry the backtester uses. A name that backtests
runs the same code in production, or it does not run at all.

The precompute cache is a small optimisation with a real effect: evaluating
twenty strategies on one symbol in one tick would otherwise recompute forty
indicator series twenty times.
"""

from __future__ import annotations

import time
from typing import Sequence

from .indicators import precompute
from .strategies import REGISTRY, Strategy, is_known, list_names

#: Bars required before any strategy is evaluated. Long EMAs need far more than
#: this to be meaningful; it is a floor, not a recommendation.
MIN_HISTORY = 60

_CACHE_TTL_SECONDS = 5.0
_CACHE_MAX_ENTRIES = 50
_cache: dict[tuple, tuple[float, dict]] = {}


def _cache_key(closes: Sequence[float], highs: Sequence[float]) -> tuple:
    return (
        len(closes),
        closes[0] if closes else 0.0,
        closes[-1] if closes else 0.0,
        highs[-1] if highs else 0.0,
    )


def get_indicators(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> dict:
    """Return precomputed indicators, reusing a recent result when possible."""
    key = _cache_key(closes, highs)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    computed = precompute(closes, highs, lows, volumes)
    _cache[key] = (now, computed)

    if len(_cache) > _CACHE_MAX_ENTRIES:
        for stale_key, _ in sorted(_cache.items(), key=lambda kv: kv[1][0])[
            : _CACHE_MAX_ENTRIES // 2
        ]:
            _cache.pop(stale_key, None)

    return computed


def clear_cache() -> None:
    """Drop every cached indicator set. Used by tests and on reconnect."""
    _cache.clear()


def _closed_bars(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> tuple[list, list, list, list]:
    return list(closes[:-1]), list(highs[:-1]), list(lows[:-1]), list(volumes[:-1])


def evaluate_entry(
    strategy_name: str,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> str | None:
    """Return ``"BUY"``, ``"SELL"`` or ``None`` for the last *closed* bar.

    The series passed in are expected to include the bar currently forming; it
    is dropped here. Raises ``KeyError`` for an unknown strategy rather than
    returning ``None``, so a typo in a config file fails loudly instead of
    quietly never trading.
    """
    strategy = REGISTRY[strategy_name]
    if len(closes) < MIN_HISTORY + 1:
        return None

    c, h, l, v = _closed_bars(closes, highs, lows, volumes)
    indicators = get_indicators(c, h, l, v)
    side, _exit_fn = strategy.signal_fn(len(c) - 1, c, h, l, v, indicators)
    return side if side in ("BUY", "SELL") else None


def evaluate_exit(
    strategy_name: str,
    position_side: str,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> bool:
    """Return whether the strategy wants to close the position.

    Stops and targets are handled by the caller; this covers only the
    discretionary exit a strategy defines for itself.
    """
    strategy = REGISTRY[strategy_name]
    if len(closes) < MIN_HISTORY + 1:
        return False

    c, h, l, v = _closed_bars(closes, highs, lows, volumes)
    indicators = get_indicators(c, h, l, v)
    i = len(c) - 1
    _side, exit_fn = strategy.signal_fn(i, c, h, l, v, indicators)
    if not callable(exit_fn):
        return False
    return bool(exit_fn(i, position_side, c, h, l, indicators))


def strategy_meta(strategy_name: str) -> Strategy:
    """Return the full strategy record, including leverage and ATR filter."""
    return REGISTRY[strategy_name]


def status() -> dict:
    """Small self-check used by the CLI and by smoke tests."""
    return {
        "strategies": len(REGISTRY),
        "names": list_names(),
        "min_history": MIN_HISTORY,
        "cache_entries": len(_cache),
    }


__all__ = [
    "MIN_HISTORY",
    "clear_cache",
    "evaluate_entry",
    "evaluate_exit",
    "get_indicators",
    "is_known",
    "status",
    "strategy_meta",
]
