"""Tests for the backtest harness.

The first three tests here are the ones that matter. They assert the
methodological guarantees the module docstring claims, so that a future
refactor that quietly reintroduces look-ahead bias fails the suite instead of
producing better-looking results.
"""

from __future__ import annotations

import pytest

from claudebot import backtest
from claudebot.indicators import precompute

WARMUP = backtest.WARMUP_BARS


def flat_series(n: int = 200, price: float = 100.0):
    closes = [price] * n
    return closes, list(closes), list(closes), [1000.0] * n


def rising_series(n: int = 200, start: float = 100.0, step: float = 0.5):
    closes = [start + i * step for i in range(n)]
    highs = [c + 0.2 for c in closes]
    lows = [c - 0.2 for c in closes]
    return closes, highs, lows, [1000.0] * n


def signal_once_at(index: int, side: str = "BUY"):
    """A signal function that fires exactly once, at ``index``."""

    def fn(i, closes, highs, lows, volumes, ind):
        return (side if i == index else None), None

    return fn


def always(side: str = "BUY"):
    def fn(i, closes, highs, lows, volumes, ind):
        return side, None

    return fn


# -- methodology -----------------------------------------------------------


def test_entry_fills_on_the_next_bar_not_the_signal_bar():
    """The core anti-look-ahead guarantee."""
    closes, highs, lows, volumes = rising_series()
    signal_index = WARMUP + 10

    result = backtest.run(
        signal_once_at(signal_index),
        closes,
        highs,
        lows,
        volumes,
        min_atr_pct=0.0,
    )

    assert result.count == 1
    trade = result.trades[0]
    assert trade.entry_index == signal_index + 1
    assert trade.entry_price == pytest.approx(closes[signal_index + 1])
    assert trade.entry_price != pytest.approx(closes[signal_index])


def test_round_trip_fees_are_charged_on_a_flat_market():
    """A position opened and closed at the same price must lose exactly the fee."""
    closes, highs, lows, volumes = flat_series()
    margin, leverage, fee = 10.0, 3, 0.0005

    result = backtest.run(
        signal_once_at(WARMUP + 5),
        closes,
        highs,
        lows,
        volumes,
        leverage=leverage,
        margin=margin,
        fee_pct=fee,
        min_atr_pct=0.0,
    )

    assert result.count == 1
    expected = -(margin * leverage * fee * 2)
    assert result.trades[0].pnl == pytest.approx(expected, abs=1e-6)


def test_stop_wins_when_a_bar_covers_both_stop_and_target():
    """Intrabar ambiguity must resolve conservatively, in favour of the stop."""
    n = 120
    entry = 100.0
    closes = [entry] * n
    highs = [entry] * n
    lows = [entry] * n

    signal_index = WARMUP + 1
    # The bar after entry spans far enough to touch both levels.
    outlier = signal_index + 2
    highs[outlier] = entry * 1.20  # well past a 6% target
    lows[outlier] = entry * 0.90  # well past a 2.4% stop

    result = backtest.run(
        signal_once_at(signal_index),
        closes,
        highs,
        lows,
        [1000.0] * n,
        min_atr_pct=0.0,
        stop_pct=0.024,
        target_pct=0.060,
    )

    assert result.count == 1
    assert result.trades[0].reason == "stop"
    assert result.trades[0].pnl < 0


def test_no_entry_is_taken_on_the_final_bar():
    """There is no next bar to fill on, so the signal must be ignored."""
    closes, highs, lows, volumes = rising_series(n=120)
    result = backtest.run(
        signal_once_at(len(closes) - 1),
        closes,
        highs,
        lows,
        volumes,
        min_atr_pct=0.0,
    )
    assert result.count == 0


# -- mechanics -------------------------------------------------------------


def test_no_signal_produces_no_trades():
    closes, highs, lows, volumes = rising_series()
    result = backtest.run(
        lambda *_a: (None, None), closes, highs, lows, volumes, min_atr_pct=0.0
    )
    assert result.count == 0
    assert result.win_rate == 0.0
    assert result.net_pnl == 0.0


def test_mismatched_series_lengths_are_rejected():
    with pytest.raises(ValueError):
        backtest.run(always(), [1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])


def test_open_position_at_the_end_is_recorded():
    closes, highs, lows, volumes = rising_series()
    result = backtest.run(
        signal_once_at(WARMUP + 2),
        closes,
        highs,
        lows,
        volumes,
        min_atr_pct=0.0,
        stop_pct=0.99,
        target_pct=0.99,
    )
    assert result.count == 1
    assert result.trades[0].reason == "open_at_end"


def test_atr_filter_blocks_entries_in_dead_markets():
    closes, highs, lows, volumes = flat_series()
    result = backtest.run(
        always(), closes, highs, lows, volumes, min_atr_pct=0.01
    )
    assert result.count == 0


def test_only_one_position_is_open_at_a_time():
    closes, highs, lows, volumes = rising_series(n=400)
    result = backtest.run(
        always(), closes, highs, lows, volumes, min_atr_pct=0.0
    )
    # Entries and exits must alternate: no trade may start before the previous
    # one has closed.
    ordered = sorted(result.trades, key=lambda t: t.entry_index)
    for earlier, later in zip(ordered, ordered[1:]):
        assert later.entry_index > earlier.exit_index


def test_a_rising_market_pays_a_long_its_target():
    closes, highs, lows, volumes = rising_series(n=300, step=1.0)
    result = backtest.run(
        signal_once_at(WARMUP + 1),
        closes,
        highs,
        lows,
        volumes,
        min_atr_pct=0.0,
    )
    assert result.count == 1
    assert result.trades[0].reason == "target"
    assert result.trades[0].pnl > 0


# -- statistics ------------------------------------------------------------


def test_win_rate_and_profit_factor():
    result = backtest.BacktestResult()
    result.trades = [
        backtest.Trade(0, 1, "BUY", 100, 110, 10.0, "target"),
        backtest.Trade(2, 3, "BUY", 100, 110, 10.0, "target"),
        backtest.Trade(4, 5, "BUY", 100, 95, -5.0, "stop"),
    ]
    assert result.count == 3
    assert result.wins == 2
    assert result.win_rate == pytest.approx(66.7, abs=0.1)
    assert result.net_pnl == pytest.approx(15.0)
    assert result.profit_factor == pytest.approx(4.0)


def test_profit_factor_is_none_without_losses():
    result = backtest.BacktestResult()
    result.trades = [backtest.Trade(0, 1, "BUY", 100, 110, 10.0, "target")]
    assert result.profit_factor is None


def test_max_drawdown_measures_peak_to_trough():
    result = backtest.BacktestResult()
    result.trades = [
        backtest.Trade(0, 1, "BUY", 100, 110, 10.0, "target"),
        backtest.Trade(2, 3, "BUY", 100, 95, -4.0, "stop"),
        backtest.Trade(4, 5, "BUY", 100, 95, -3.0, "stop"),
        backtest.Trade(6, 7, "BUY", 100, 110, 20.0, "target"),
    ]
    # Equity runs 10, 6, 3, 23. The worst drop from the peak of 10 is -7.
    assert result.max_drawdown == pytest.approx(-7.0)


def test_summary_is_serialisable():
    import json

    closes, highs, lows, volumes = rising_series()
    result = backtest.run(
        signal_once_at(WARMUP + 1), closes, highs, lows, volumes, min_atr_pct=0.0
    )
    json.dumps(result.summary())


def test_regime_labels_align_with_the_series():
    closes, highs, lows, volumes = rising_series(n=300)
    indicators = precompute(closes, highs, lows, volumes)
    labels = backtest.label_regimes(closes, highs, lows, indicators)
    assert len(labels) == len(closes)
    assert all(label is None for label in labels[:50])
