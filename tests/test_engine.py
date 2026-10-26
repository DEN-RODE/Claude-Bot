"""Tests for the shared strategy engine.

The closed-bar test is the important one: it is the guarantee that live
execution evaluates the same data the backtest did.
"""

from __future__ import annotations

import math

import pytest

from claudebot import engine, strategies


@pytest.fixture(autouse=True)
def _clear_cache():
    engine.clear_cache()
    yield
    engine.clear_cache()


def wave(n: int = 300) -> tuple[list, list, list, list]:
    closes = [100 + 15 * math.sin(i / 9) for i in range(n)]
    highs = [c + 0.8 for c in closes]
    lows = [c - 0.8 for c in closes]
    return closes, highs, lows, [1000.0] * n


def test_registry_is_not_empty():
    assert len(strategies.list_names()) >= 20


def test_every_registered_strategy_evaluates_without_raising():
    closes, highs, lows, volumes = wave()
    for name in strategies.list_names():
        side = engine.evaluate_entry(name, closes, highs, lows, volumes)
        assert side in (None, "BUY", "SELL"), f"{name} returned {side!r}"


def test_unknown_strategy_raises_rather_than_returning_none():
    """A typo in a config file must fail loudly, not silently never trade."""
    closes, highs, lows, volumes = wave()
    with pytest.raises(KeyError):
        engine.evaluate_entry("does_not_exist", closes, highs, lows, volumes)


def test_short_history_returns_none():
    closes = [100.0] * 10
    assert (
        engine.evaluate_entry("macd_flip", closes, closes, closes, [1.0] * 10) is None
    )


def test_the_forming_bar_is_excluded():
    """Mutating the last bar must not change the decision.

    This is the whole closed-bar rule: the final element of each series is the
    bar currently forming, and a signal that moves when it moves is a signal
    that fires and unfires several times an hour in production.
    """
    closes, highs, lows, volumes = wave()

    baseline = {
        name: engine.evaluate_entry(name, closes, highs, lows, volumes)
        for name in strategies.list_names()
    }

    # A violent move on the still-forming bar.
    spiked_closes = list(closes)
    spiked_highs = list(highs)
    spiked_lows = list(lows)
    spiked_closes[-1] *= 1.5
    spiked_highs[-1] *= 1.6
    spiked_lows[-1] *= 0.6

    engine.clear_cache()
    after = {
        name: engine.evaluate_entry(name, spiked_closes, spiked_highs, spiked_lows, volumes)
        for name in strategies.list_names()
    }

    assert after == baseline


def test_evaluate_exit_returns_a_boolean():
    closes, highs, lows, volumes = wave()
    for name in strategies.list_names():
        result = engine.evaluate_exit(name, "BUY", closes, highs, lows, volumes)
        assert isinstance(result, bool)


def test_indicator_cache_returns_the_same_object():
    closes, highs, lows, volumes = wave()
    first = engine.get_indicators(closes, highs, lows, volumes)
    second = engine.get_indicators(closes, highs, lows, volumes)
    assert first is second


def test_clear_cache_forces_recomputation():
    closes, highs, lows, volumes = wave()
    first = engine.get_indicators(closes, highs, lows, volumes)
    engine.clear_cache()
    second = engine.get_indicators(closes, highs, lows, volumes)
    assert first is not second
    assert first["rsi14"] == second["rsi14"]


def test_strategy_metadata_is_sane():
    for name in strategies.list_names():
        meta = engine.strategy_meta(name)
        assert meta.name == name
        assert meta.leverage >= 1
        assert 0 < meta.min_atr_pct < 1
        assert meta.description


def test_is_known_agrees_with_the_registry():
    assert engine.is_known("macd_flip")
    assert not engine.is_known("macd_flop")


def test_status_reports_the_registry():
    status = engine.status()
    assert status["strategies"] == len(strategies.list_names())
    assert "macd_flip" in status["names"]


def test_a_known_strategy_fires_on_a_constructed_setup():
    """RSI reversion must produce BUY on a series driven into oversold."""
    n = 200
    closes = [100.0] * (n - 40) + [100 - i * 1.2 for i in range(1, 41)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    side = engine.evaluate_entry(
        "rsi_reversion", closes, highs, lows, [1000.0] * n
    )
    assert side == "BUY"
