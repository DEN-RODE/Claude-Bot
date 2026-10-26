"""Indicator correctness, against hand-computable cases."""

from __future__ import annotations

import math

import pytest

from claudebot import indicators as ind


@pytest.fixture
def ramp() -> list[float]:
    """A strictly increasing series: every momentum indicator should be pinned high."""
    return [float(i) for i in range(1, 101)]


def test_sma_matches_manual_average():
    data = [1, 2, 3, 4, 5, 6]
    result = ind.sma(data, 3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[-1] == pytest.approx(5.0)


def test_ema_seeds_with_simple_average():
    data = [10.0] * 20
    result = ind.ema(data, 5)
    # A flat series must stay flat: the seed and every subsequent value equal 10.
    assert result[4] == pytest.approx(10.0)
    assert result[-1] == pytest.approx(10.0)


def test_ema_returns_all_none_when_series_too_short():
    assert ind.ema([1.0, 2.0], 10) == [None, None]


def test_series_length_is_preserved(ramp):
    """Every indicator must return a series aligned with the input."""
    highs = [v + 1 for v in ramp]
    lows = [v - 1 for v in ramp]
    volumes = [100.0] * len(ramp)
    computed = ind.precompute(ramp, highs, lows, volumes)
    for name, series in computed.items():
        assert len(series) == len(ramp), f"{name} has the wrong length"


def test_rsi_saturates_on_a_pure_uptrend(ramp):
    result = ind.rsi(ramp, 14)
    assert result[-1] == pytest.approx(100.0)


def test_rsi_bottoms_on_a_pure_downtrend(ramp):
    result = ind.rsi(list(reversed(ramp)), 14)
    assert result[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_stays_within_bounds():
    data = [50 + 10 * math.sin(i / 3) for i in range(200)]
    for value in ind.rsi(data, 14):
        if value is not None:
            assert 0.0 <= value <= 100.0


def test_atr_is_positive_and_warms_up():
    highs = [10 + i * 0.5 for i in range(50)]
    lows = [9 + i * 0.5 for i in range(50)]
    closes = [9.5 + i * 0.5 for i in range(50)]
    result = ind.atr(highs, lows, closes, 14)
    assert result[13] is None
    assert result[20] is not None and result[20] > 0


def test_bollinger_bands_are_ordered():
    data = [50 + 5 * math.sin(i / 4) for i in range(100)]
    upper, mid, lower = ind.bollinger(data, 20, 2.0)
    for i in range(20, len(data)):
        assert lower[i] < mid[i] < upper[i]


def test_bollinger_collapses_on_a_flat_series():
    data = [42.0] * 50
    upper, mid, lower = ind.bollinger(data, 20, 2.0)
    assert upper[-1] == pytest.approx(42.0)
    assert lower[-1] == pytest.approx(42.0)
    assert mid[-1] == pytest.approx(42.0)


def test_macd_histogram_is_zero_on_a_flat_series():
    data = [100.0] * 120
    _line, _signal, hist = ind.macd(data)
    assert hist[-1] == pytest.approx(0.0, abs=1e-9)


def test_macd_histogram_is_not_identically_zero_on_real_movement():
    """Guards a regression where the signal EMA was computed over a 1-element
    list, which made the histogram exactly zero at every bar."""
    data = [100 + 20 * math.sin(i / 7) for i in range(300)]
    _line, _signal, hist = ind.macd(data)
    non_zero = [h for h in hist if h is not None and abs(h) > 1e-6]
    assert len(non_zero) > 50


def test_supertrend_reports_only_valid_directions():
    highs = [10 + math.sin(i / 5) for i in range(200)]
    lows = [9 + math.sin(i / 5) for i in range(200)]
    closes = [9.5 + math.sin(i / 5) for i in range(200)]
    trend, _u, _l = ind.supertrend(highs, lows, closes)
    for value in trend:
        assert value in (None, 1, -1)


def test_williams_r_is_bounded():
    highs = [10 + math.sin(i / 3) for i in range(100)]
    lows = [9 + math.sin(i / 3) for i in range(100)]
    closes = [9.5 + math.sin(i / 3) for i in range(100)]
    for value in ind.williams_r(highs, lows, closes, 14):
        if value is not None:
            assert -100.0 <= value <= 0.0


def test_donchian_channel_contains_price():
    highs = [10 + math.sin(i / 4) for i in range(100)]
    lows = [9 + math.sin(i / 4) for i in range(100)]
    upper, lower = ind.donchian(highs, lows, 20)
    for i in range(20, 100):
        assert lower[i] <= lows[i]
        assert upper[i] >= highs[i]


def test_vwap_equals_price_on_constant_series():
    closes = [100.0] * 50
    result = ind.vwap(closes, closes, closes, [10.0] * 50)
    assert result[-1] == pytest.approx(100.0)


def test_adx_is_non_negative():
    highs = [10 + i * 0.1 for i in range(100)]
    lows = [9 + i * 0.1 for i in range(100)]
    closes = [9.5 + i * 0.1 for i in range(100)]
    adx_vals, di_pos, di_neg = ind.adx(highs, lows, closes, 14)
    assert all(v >= 0 for v in adx_vals)
    assert all(v >= 0 for v in di_pos)
    assert all(v >= 0 for v in di_neg)


def test_precompute_exposes_every_key_strategies_use():
    from claudebot import strategies  # local import keeps the dependency visible

    data = [100 + 10 * math.sin(i / 6) for i in range(300)]
    highs = [v + 1 for v in data]
    lows = [v - 1 for v in data]
    volumes = [1000.0] * len(data)
    computed = ind.precompute(data, highs, lows, volumes)

    # Every strategy must be able to run against a precompute dict without
    # raising KeyError, which is the failure mode when a key is renamed.
    for name in strategies.list_names():
        strategy = strategies.get(name)
        strategy.signal_fn(len(data) - 1, data, highs, lows, volumes, computed)
