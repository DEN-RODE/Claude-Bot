"""Technical indicators implemented from scratch on plain Python lists.

No numpy, no pandas, no TA-Lib. Every function takes and returns lists of
``float | None``, where ``None`` marks a bar that does not yet have enough
history for the indicator to be defined. Keeping the ``None`` prefix instead of
silently zero-filling matters: a zero RSI is a valid value and a missing RSI is
not, and conflating them is how backtests end up trading on warm-up noise.

All series returned have the same length as the input, so an indicator value at
index ``i`` always describes the bar at index ``i``.
"""

from __future__ import annotations

import math
from typing import Sequence

Series = list[float | None]


def ema(data: Sequence[float], period: int) -> Series:
    """Exponential moving average, seeded with a simple average of the first
    ``period`` values."""
    out: Series = [None] * len(data)
    if len(data) < period or period < 1:
        return out
    out[period - 1] = sum(data[:period]) / period
    k = 2 / (period + 1)
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def sma(data: Sequence[float], period: int) -> Series:
    """Simple moving average."""
    out: Series = [None] * len(data)
    for i in range(period - 1, len(data)):
        out[i] = sum(data[i - period + 1 : i + 1]) / period
    return out


def macd(
    closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Series, Series, Series]:
    """Return ``(macd_line, signal_line, histogram)``."""
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    line: Series = [
        a - b if a is not None and b is not None else None
        for a, b in zip(fast_ema, slow_ema)
    ]
    # The signal EMA needs a dense series; zero-fill only for that computation,
    # then mask the histogram back to None wherever the MACD line is undefined.
    dense = [x if x is not None else 0.0 for x in line]
    signal_line = ema(dense, signal)
    hist: Series = [
        line[i] - signal_line[i]
        if line[i] is not None and signal_line[i] is not None
        else None
        for i in range(len(closes))
    ]
    return line, signal_line, hist


def rsi(closes: Sequence[float], period: int = 14) -> Series:
    """Wilder's relative strength index."""
    out: Series = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        out[i + 1] = 100 - 100 / (1 + rs)
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Series:
    """Average true range."""
    out: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        trs = [
            max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j - 1]),
                abs(lows[j] - closes[j - 1]),
            )
            for j in range(i - period + 1, i + 1)
        ]
        out[i] = sum(trs) / period
    return out


def bollinger(
    closes: Sequence[float], period: int = 20, k: float = 2.0
) -> tuple[Series, Series, Series]:
    """Return ``(upper, middle, lower)`` Bollinger bands."""
    mid = sma(closes, period)
    upper: Series = [None] * len(closes)
    lower: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        mean = mid[i]
        variance = sum((closes[i - period + 1 + j] - mean) ** 2 for j in range(period))
        std = math.sqrt(variance / period)
        upper[i] = mean + k * std
        lower[i] = mean - k * std
    return upper, mid, lower


def bb_width(closes: Sequence[float], period: int = 20, k: float = 2.0) -> Series:
    """Bollinger band width, ``(upper - lower) / middle``."""
    up, mid, lo = bollinger(closes, period, k)
    out: Series = [None] * len(closes)
    for i in range(len(closes)):
        if up[i] is not None and mid[i] not in (None, 0):
            out[i] = (up[i] - lo[i]) / mid[i]
    return out


def bb_pct_b(closes: Sequence[float], period: int = 20, k: float = 2.0) -> Series:
    """Bollinger %B, the position of price within the bands."""
    up, _mid, lo = bollinger(closes, period, k)
    out: Series = [None] * len(closes)
    for i in range(len(closes)):
        if up[i] is not None and lo[i] is not None and up[i] != lo[i]:
            out[i] = (closes[i] - lo[i]) / (up[i] - lo[i])
    return out


def stoch_rsi(
    closes: Sequence[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[Series, Series]:
    """Return the smoothed ``(%K, %D)`` of the stochastic RSI."""
    rsi_vals = rsi(closes, rsi_period)
    raw_k: Series = [None] * len(closes)
    for i in range(rsi_period + stoch_period, len(closes)):
        window = [
            rsi_vals[j]
            for j in range(i - stoch_period + 1, i + 1)
            if rsi_vals[j] is not None
        ]
        if len(window) < stoch_period:
            continue
        lo, hi = min(window), max(window)
        raw_k[i] = (rsi_vals[i] - lo) / (hi - lo) * 100 if hi != lo else 50.0

    def _smooth(series: Series, n: int) -> Series:
        out: Series = [None] * len(series)
        for i in range(n - 1, len(series)):
            vals = [series[j] for j in range(i - n + 1, i + 1) if series[j] is not None]
            if len(vals) == n:
                out[i] = sum(vals) / n
        return out

    k = _smooth(raw_k, k_smooth)
    d = _smooth(k, d_smooth)
    return k, d


def cci(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 20,
) -> Series:
    """Commodity channel index."""
    out: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        typical = [
            (highs[j] + lows[j] + closes[j]) / 3 for j in range(i - period + 1, i + 1)
        ]
        mean = sum(typical) / period
        mean_dev = sum(abs(v - mean) for v in typical) / period
        out[i] = (typical[-1] - mean) / (0.015 * mean_dev) if mean_dev > 0 else 0.0
    return out


def williams_r(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Series:
    """Williams %R, bounded in ``[-100, 0]``."""
    out: Series = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        hh = max(highs[i - period + 1 : i + 1])
        ll = min(lows[i - period + 1 : i + 1])
        out[i] = (hh - closes[i]) / (hh - ll) * -100 if hh != ll else -50.0
    return out


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> Series:
    """Cumulative volume weighted average price from the start of the series."""
    out: Series = [None] * len(closes)
    cum_pv = cum_vol = 0.0
    for i in range(len(closes)):
        typical = (highs[i] + lows[i] + closes[i]) / 3
        cum_pv += typical * volumes[i]
        cum_vol += volumes[i]
        out[i] = cum_pv / cum_vol if cum_vol > 0 else closes[i]
    return out


def supertrend(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[Series, Series, Series]:
    """Return ``(trend, upper_band, lower_band)``.

    ``trend`` is ``1`` for an uptrend and ``-1`` for a downtrend.
    """
    atr_vals = atr(highs, lows, closes, period)
    upper: Series = [None] * len(closes)
    lower: Series = [None] * len(closes)
    trend: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        if atr_vals[i] is None:
            continue
        mid = (highs[i] + lows[i]) / 2
        basic_upper = mid + multiplier * atr_vals[i]
        basic_lower = mid - multiplier * atr_vals[i]
        if i == period or upper[i - 1] is None:
            upper[i], lower[i], trend[i] = basic_upper, basic_lower, 1
            continue
        upper[i] = (
            basic_upper
            if basic_upper < upper[i - 1] or closes[i - 1] > upper[i - 1]
            else upper[i - 1]
        )
        lower[i] = (
            basic_lower
            if basic_lower > lower[i - 1] or closes[i - 1] < lower[i - 1]
            else lower[i - 1]
        )
        if trend[i - 1] == -1 and closes[i] > upper[i]:
            trend[i] = 1
        elif trend[i - 1] == 1 and closes[i] < lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return trend, upper, lower


def donchian(
    highs: Sequence[float], lows: Sequence[float], period: int = 20
) -> tuple[Series, Series]:
    """Return ``(upper_channel, lower_channel)``."""
    up: Series = [None] * len(highs)
    lo: Series = [None] * len(lows)
    for i in range(period - 1, len(highs)):
        up[i] = max(highs[i - period + 1 : i + 1])
        lo[i] = min(lows[i - period + 1 : i + 1])
    return up, lo


def keltner(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[Series, Series, Series]:
    """Return ``(upper, middle, lower)`` Keltner channels."""
    mid = ema(closes, ema_period)
    atr_vals = atr(highs, lows, closes, atr_period)
    up: Series = [
        m + multiplier * a if m is not None and a is not None else None
        for m, a in zip(mid, atr_vals)
    ]
    lo: Series = [
        m - multiplier * a if m is not None and a is not None else None
        for m, a in zip(mid, atr_vals)
    ]
    return up, mid, lo


def hma(closes: Sequence[float], period: int = 20) -> Series:
    """Hull-style smoothed average built from EMAs rather than weighted MAs."""
    half = max(1, period // 2)
    fast = ema(closes, half)
    slow = ema(closes, period)
    diff = [
        2 * a - b if a is not None and b is not None else None
        for a, b in zip(fast, slow)
    ]
    dense = [x if x is not None else 0.0 for x in diff]
    return ema(dense, max(1, int(math.sqrt(period))))


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(adx, +DI, -DI)`` as dense lists (zero before warm-up)."""
    n = len(closes)
    dm_pos = [0.0] * n
    dm_neg = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        dm_pos[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        dm_neg[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    def _wilder(arr: list[float]) -> list[float]:
        out = [0.0] * len(arr)
        if period < len(arr):
            out[period] = sum(arr[1 : period + 1])
        for i in range(period + 1, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / period + arr[i]
        return out

    str_tr = _wilder(tr)
    str_pos = _wilder(dm_pos)
    str_neg = _wilder(dm_neg)
    di_pos = [100 * str_pos[i] / str_tr[i] if str_tr[i] > 0 else 0.0 for i in range(n)]
    di_neg = [100 * str_neg[i] / str_tr[i] if str_tr[i] > 0 else 0.0 for i in range(n)]
    dx = [
        100 * abs(di_pos[i] - di_neg[i]) / (di_pos[i] + di_neg[i])
        if (di_pos[i] + di_neg[i]) > 0
        else 0.0
        for i in range(n)
    ]
    out = [0.0] * n
    if period * 2 < n:
        out[period * 2] = sum(dx[period : period * 2]) / period
    for i in range(period * 2 + 1, n):
        out[i] = (out[i - 1] * (period - 1) + dx[i]) / period
    return out, di_pos, di_neg


def roc(closes: Sequence[float], period: int = 14) -> Series:
    """Rate of change, in percent."""
    out: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            out[i] = (closes[i] - closes[i - period]) / closes[i - period] * 100
    return out


def momentum(closes: Sequence[float], period: int = 14) -> Series:
    """Momentum oscillator, ``price / price[-period] * 100``."""
    out: Series = [None] * len(closes)
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            out[i] = closes[i] / closes[i - period] * 100
    return out


def precompute(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> dict[str, Series]:
    """Compute every indicator once and return them keyed by short name.

    Strategies receive this dict rather than recomputing indicators themselves.
    With ~40 series and several strategies sharing most of them, computing once
    per bar window is the difference between a backtest that runs in seconds and
    one that runs in minutes.
    """
    macd_line, macd_signal, macd_hist = macd(closes, 12, 26, 9)
    _, _, macd_hist_fast = macd(closes, 8, 21, 5)
    _, _, macd_hist_slow = macd(closes, 19, 39, 9)
    _, _, macd_hist_scalp = macd(closes, 5, 13, 4)
    bb_up, bb_mid, bb_lo = bollinger(closes, 20, 2.0)
    bb_up_tight, _, bb_lo_tight = bollinger(closes, 20, 1.5)
    bb_up_wide, _, bb_lo_wide = bollinger(closes, 20, 2.5)
    st_trend, _, _ = supertrend(highs, lows, closes, 10, 3.0)
    st_trend_fast, _, _ = supertrend(highs, lows, closes, 7, 2.0)
    don_h20, don_l20 = donchian(highs, lows, 20)
    don_h55, don_l55 = donchian(highs, lows, 55)
    kel_up, _, kel_lo = keltner(highs, lows, closes, 20, 10, 2.0)
    adx_vals, di_pos, di_neg = adx(highs, lows, closes, 14)
    stoch_k, stoch_d = stoch_rsi(closes)

    return {
        "rsi14": rsi(closes, 14),
        "rsi7": rsi(closes, 7),
        "rsi21": rsi(closes, 21),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_h": macd_hist,
        "macd_h_fast": macd_hist_fast,
        "macd_h_slow": macd_hist_slow,
        "macd_h_scalp": macd_hist_scalp,
        "ema5": ema(closes, 5),
        "ema9": ema(closes, 9),
        "ema13": ema(closes, 13),
        "ema20": ema(closes, 20),
        "ema21": ema(closes, 21),
        "ema50": ema(closes, 50),
        "ema200": ema(closes, 200),
        "sma20": sma(closes, 20),
        "sma50": sma(closes, 50),
        "bb_up": bb_up,
        "bb_mid": bb_mid,
        "bb_lo": bb_lo,
        "bb_up_tight": bb_up_tight,
        "bb_lo_tight": bb_lo_tight,
        "bb_up_wide": bb_up_wide,
        "bb_lo_wide": bb_lo_wide,
        "bb_width": bb_width(closes, 20, 2.0),
        "bb_pctb": bb_pct_b(closes, 20, 2.0),
        "st": st_trend,
        "st_fast": st_trend_fast,
        "don_h20": don_h20,
        "don_l20": don_l20,
        "don_h55": don_h55,
        "don_l55": don_l55,
        "kel_up": kel_up,
        "kel_lo": kel_lo,
        "cci20": cci(highs, lows, closes, 20),
        "wr14": williams_r(highs, lows, closes, 14),
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "adx": adx_vals,
        "di_pos": di_pos,
        "di_neg": di_neg,
        "hma20": hma(closes, 20),
        "hma9": hma(closes, 9),
        "vwap": vwap(highs, lows, closes, volumes),
        "atr14": atr(highs, lows, closes, 14),
        "roc14": roc(closes, 14),
        "mom14": momentum(closes, 14),
        "volume": list(volumes),
    }
