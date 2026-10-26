"""Strategy definitions and the registry that backtest, paper and live share.

A strategy is a factory that returns a *signal function*:

    signal_fn(i, closes, highs, lows, volumes, indicators)
        -> (side, exit_fn)

``side`` is ``"BUY"``, ``"SELL"`` or ``None``. ``exit_fn`` is a callable that
decides whether an open position should be closed on discretionary grounds;
stops and targets are handled by the caller, not here.

The single-registry design is deliberate. In an earlier version of this project
the backtester had its own strategy implementations and the live trader had a
separate, smaller set, with a lookup table mapping one onto the other. The
result was that a strategy validated under one name ran entirely different logic
in production. Everything now resolves through :data:`REGISTRY`, so a name that
backtests is the same code that trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Indicators = dict[str, list]
ExitFn = Callable[[int, str, Sequence[float], Sequence[float], Sequence[float], Indicators], bool]
SignalFn = Callable[
    [int, Sequence[float], Sequence[float], Sequence[float], Sequence[float], Indicators],
    tuple[str | None, ExitFn | None],
]


@dataclass(frozen=True)
class Strategy:
    """A named signal function plus the context it was validated in."""

    name: str
    signal_fn: SignalFn
    leverage: int = 3
    min_atr_pct: float = 0.003
    description: str = ""


def _no_exit(*_args) -> bool:
    return False


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------


def make_macd_flip(hist_key: str = "macd_h", min_hist_pct: float = 0.00005) -> SignalFn:
    """Enter on a MACD histogram sign change, exit on the opposite flip.

    ``min_hist_pct`` filters flips whose histogram is negligible relative to
    price, which otherwise fire constantly in flat markets.
    """

    def fn(i, closes, highs, lows, volumes, ind):
        hist = ind[hist_key]
        if i < 1 or hist[i] is None or hist[i - 1] is None:
            return None, _no_exit
        if closes[i] <= 0 or abs(hist[i]) / closes[i] < min_hist_pct:
            return None, _no_exit
        side = None
        if hist[i - 1] <= 0 < hist[i]:
            side = "BUY"
        elif hist[i - 1] >= 0 > hist[i]:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            h2 = ind2[hist_key]
            if j < 1 or h2[j] is None or h2[j - 1] is None:
                return False
            return (position_side == "BUY" and h2[j - 1] >= 0 > h2[j]) or (
                position_side == "SELL" and h2[j - 1] <= 0 < h2[j]
            )

        return side, exit_fn

    return fn


def make_rsi_reversion(
    overbought: float = 70,
    oversold: float = 30,
    exit_overbought: float = 55,
    exit_oversold: float = 45,
    rsi_key: str = "rsi14",
) -> SignalFn:
    """Fade RSI extremes and exit once it returns toward the middle."""

    def fn(i, closes, highs, lows, volumes, ind):
        value = ind[rsi_key][i]
        if value is None:
            return None, _no_exit
        side = "BUY" if value < oversold else ("SELL" if value > overbought else None)

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            v2 = ind2[rsi_key][j]
            if v2 is None:
                return False
            return (position_side == "BUY" and v2 > exit_oversold) or (
                position_side == "SELL" and v2 < exit_overbought
            )

        return side, exit_fn

    return fn


def make_rsi_volume_surge(
    overbought: float = 70, oversold: float = 30, volume_multiple: float = 1.5
) -> SignalFn:
    """RSI reversion that only fires when volume confirms the move."""

    def fn(i, closes, highs, lows, volumes, ind):
        value = ind["rsi14"][i]
        if value is None or i < 20:
            return None, _no_exit
        avg_volume = sum(ind["volume"][i - 20 : i]) / 20
        if avg_volume <= 0 or ind["volume"][i] < avg_volume * volume_multiple:
            return None, _no_exit
        side = "BUY" if value < oversold else ("SELL" if value > overbought else None)

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            v2 = ind2["rsi14"][j]
            if v2 is None:
                return False
            return (position_side == "BUY" and v2 > 55) or (
                position_side == "SELL" and v2 < 45
            )

        return side, exit_fn

    return fn


def make_ema_cross(fast_key: str = "ema9", slow_key: str = "ema21") -> SignalFn:
    """Classic fast/slow EMA crossover, exit on the reverse cross."""

    def fn(i, closes, highs, lows, volumes, ind):
        fast, slow = ind[fast_key], ind[slow_key]
        if i < 1 or any(
            x is None for x in (fast[i], fast[i - 1], slow[i], slow[i - 1])
        ):
            return None, _no_exit
        side = None
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            side = "BUY"
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            f2, s2 = ind2[fast_key], ind2[slow_key]
            if j < 1 or any(
                x is None for x in (f2[j], f2[j - 1], s2[j], s2[j - 1])
            ):
                return False
            return (
                position_side == "BUY" and f2[j - 1] >= s2[j - 1] and f2[j] < s2[j]
            ) or (
                position_side == "SELL" and f2[j - 1] <= s2[j - 1] and f2[j] > s2[j]
            )

        return side, exit_fn

    return fn


def make_triple_ema() -> SignalFn:
    """Require 9/21/50 EMAs stacked in the same direction before entering."""

    def fn(i, closes, highs, lows, volumes, ind):
        e9, e21, e50 = ind["ema9"][i], ind["ema21"][i], ind["ema50"][i]
        if None in (e9, e21, e50):
            return None, _no_exit
        side = None
        if e9 > e21 > e50:
            side = "BUY"
        elif e9 < e21 < e50:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            a, b = ind2["ema9"][j], ind2["ema21"][j]
            if None in (a, b):
                return False
            return (position_side == "BUY" and a < b) or (
                position_side == "SELL" and a > b
            )

        return side, exit_fn

    return fn


def make_bb_bounce() -> SignalFn:
    """Buy the lower Bollinger band, sell the upper, exit at the middle."""

    def fn(i, closes, highs, lows, volumes, ind):
        up, mid, lo = ind["bb_up"][i], ind["bb_mid"][i], ind["bb_lo"][i]
        if None in (up, mid, lo):
            return None, _no_exit
        side = None
        if closes[i] <= lo:
            side = "BUY"
        elif closes[i] >= up:
            side = "SELL"

        def exit_fn(j, position_side, c2, _h, _l, ind2):
            m2 = ind2["bb_mid"][j]
            if m2 is None:
                return False
            return (position_side == "BUY" and c2[j] >= m2) or (
                position_side == "SELL" and c2[j] <= m2
            )

        return side, exit_fn

    return fn


def make_bb_squeeze_breakout(squeeze_threshold: float = 0.02) -> SignalFn:
    """Trade the breakout that follows a volatility contraction."""

    def fn(i, closes, highs, lows, volumes, ind):
        width = ind["bb_width"][i]
        prev_width = ind["bb_width"][i - 1] if i >= 1 else None
        up, lo = ind["bb_up"][i], ind["bb_lo"][i]
        if None in (width, prev_width, up, lo):
            return None, _no_exit
        if prev_width > squeeze_threshold:
            return None, _no_exit
        side = None
        if closes[i] > up:
            side = "BUY"
        elif closes[i] < lo:
            side = "SELL"
        return side, _no_exit

    return fn


def make_supertrend(trend_key: str = "st") -> SignalFn:
    """Follow Supertrend direction changes."""

    def fn(i, closes, highs, lows, volumes, ind):
        trend = ind[trend_key]
        if i < 1 or trend[i] is None or trend[i - 1] is None:
            return None, _no_exit
        side = None
        if trend[i - 1] == -1 and trend[i] == 1:
            side = "BUY"
        elif trend[i - 1] == 1 and trend[i] == -1:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            t2 = ind2[trend_key][j]
            if t2 is None:
                return False
            return (position_side == "BUY" and t2 == -1) or (
                position_side == "SELL" and t2 == 1
            )

        return side, exit_fn

    return fn


def make_cci_reversion(overbought: float = 100, oversold: float = -100) -> SignalFn:
    """Fade CCI extremes, exit when it crosses back through zero."""

    def fn(i, closes, highs, lows, volumes, ind):
        value = ind["cci20"][i]
        if value is None:
            return None, _no_exit
        side = "BUY" if value < oversold else ("SELL" if value > overbought else None)

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            v2 = ind2["cci20"][j]
            if v2 is None:
                return False
            return (position_side == "BUY" and v2 > 0) or (
                position_side == "SELL" and v2 < 0
            )

        return side, exit_fn

    return fn


def make_williams_r(overbought: float = -20, oversold: float = -80) -> SignalFn:
    """Fade Williams %R extremes."""

    def fn(i, closes, highs, lows, volumes, ind):
        value = ind["wr14"][i]
        if value is None:
            return None, _no_exit
        side = "BUY" if value < oversold else ("SELL" if value > overbought else None)

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            v2 = ind2["wr14"][j]
            if v2 is None:
                return False
            return (position_side == "BUY" and v2 > -50) or (
                position_side == "SELL" and v2 < -50
            )

        return side, exit_fn

    return fn


def make_stoch_rsi_cross(overbought: float = 80, oversold: float = 20) -> SignalFn:
    """Enter on a %K/%D cross taking place in an extreme zone."""

    def fn(i, closes, highs, lows, volumes, ind):
        k, d = ind["stoch_k"], ind["stoch_d"]
        if i < 1 or any(x is None for x in (k[i], k[i - 1], d[i], d[i - 1])):
            return None, _no_exit
        side = None
        if k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i] < oversold + 20:
            side = "BUY"
        elif k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i] > overbought - 20:
            side = "SELL"
        return side, _no_exit

    return fn


def make_donchian_breakout(
    upper_key: str = "don_h20", lower_key: str = "don_l20"
) -> SignalFn:
    """Breakout of the N-bar high or low."""

    def fn(i, closes, highs, lows, volumes, ind):
        up, lo = ind[upper_key][i - 1] if i >= 1 else None, (
            ind[lower_key][i - 1] if i >= 1 else None
        )
        if None in (up, lo):
            return None, _no_exit
        side = None
        if closes[i] > up:
            side = "BUY"
        elif closes[i] < lo:
            side = "SELL"
        return side, _no_exit

    return fn


def make_keltner_breakout() -> SignalFn:
    """Breakout of the Keltner channel."""

    def fn(i, closes, highs, lows, volumes, ind):
        up, lo = ind["kel_up"][i], ind["kel_lo"][i]
        if None in (up, lo):
            return None, _no_exit
        side = None
        if closes[i] > up:
            side = "BUY"
        elif closes[i] < lo:
            side = "SELL"
        return side, _no_exit

    return fn


def make_adx_directional(adx_threshold: float = 25) -> SignalFn:
    """Trade the dominant directional index once ADX confirms a trend."""

    def fn(i, closes, highs, lows, volumes, ind):
        adx_v, pos, neg = ind["adx"][i], ind["di_pos"][i], ind["di_neg"][i]
        if adx_v is None or adx_v < adx_threshold:
            return None, _no_exit
        side = "BUY" if pos > neg else "SELL"

        def exit_fn(j, _side, _c, _h, _l, ind2):
            return ind2["adx"][j] < adx_threshold

        return side, exit_fn

    return fn


def make_hma_cross() -> SignalFn:
    """Fast/slow Hull-style moving average cross."""

    def fn(i, closes, highs, lows, volumes, ind):
        fast, slow = ind["hma9"], ind["hma20"]
        if i < 1 or any(
            x is None for x in (fast[i], fast[i - 1], slow[i], slow[i - 1])
        ):
            return None, _no_exit
        side = None
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            side = "BUY"
        elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            side = "SELL"
        return side, _no_exit

    return fn


def make_roc_momentum(threshold: float = 1.0) -> SignalFn:
    """Enter in the direction of a strong rate of change."""

    def fn(i, closes, highs, lows, volumes, ind):
        value = ind["roc14"][i]
        if value is None:
            return None, _no_exit
        side = None
        if value > threshold:
            side = "BUY"
        elif value < -threshold:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            v2 = ind2["roc14"][j]
            if v2 is None:
                return False
            return (position_side == "BUY" and v2 < 0) or (
                position_side == "SELL" and v2 > 0
            )

        return side, exit_fn

    return fn


def make_vwap_reversion() -> SignalFn:
    """Fade price stretched away from VWAP, exit on the return to it."""

    def fn(i, closes, highs, lows, volumes, ind):
        anchor = ind["vwap"][i]
        atr_v = ind["atr14"][i]
        if anchor is None or atr_v is None or atr_v <= 0:
            return None, _no_exit
        distance = (closes[i] - anchor) / atr_v
        side = None
        if distance < -1.5:
            side = "BUY"
        elif distance > 1.5:
            side = "SELL"

        def exit_fn(j, position_side, c2, _h, _l, ind2):
            a2 = ind2["vwap"][j]
            if a2 is None:
                return False
            return (position_side == "BUY" and c2[j] >= a2) or (
                position_side == "SELL" and c2[j] <= a2
            )

        return side, exit_fn

    return fn


def make_macd_rsi_combo(rsi_bull: float = 60, rsi_bear: float = 40) -> SignalFn:
    """Require MACD and RSI to agree before entering."""

    def fn(i, closes, highs, lows, volumes, ind):
        hist, rsi_v = ind["macd_h"][i], ind["rsi14"][i]
        if hist is None or rsi_v is None:
            return None, _no_exit
        side = None
        if hist > 0 and rsi_v > rsi_bull:
            side = "BUY"
        elif hist < 0 and rsi_v < rsi_bear:
            side = "SELL"

        def exit_fn(j, position_side, _c, _h, _l, ind2):
            h2 = ind2["macd_h"][j]
            if h2 is None:
                return False
            return (position_side == "BUY" and h2 < 0) or (
                position_side == "SELL" and h2 > 0
            )

        return side, exit_fn

    return fn


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_ALL: list[Strategy] = [
    Strategy("macd_flip", make_macd_flip(), 3, 0.003, "MACD histogram sign change"),
    Strategy(
        "macd_flip_fast",
        make_macd_flip("macd_h_fast"),
        3,
        0.004,
        "MACD 8/21/5 sign change",
    ),
    Strategy("rsi_reversion", make_rsi_reversion(), 3, 0.003, "Fade RSI 30/70"),
    Strategy(
        "rsi_reversion_wide",
        make_rsi_reversion(75, 25),
        3,
        0.004,
        "Fade RSI 25/75",
    ),
    Strategy(
        "rsi_volume_surge",
        make_rsi_volume_surge(),
        3,
        0.003,
        "RSI reversion confirmed by volume",
    ),
    Strategy("ema_cross_9_21", make_ema_cross(), 3, 0.003, "EMA 9/21 cross"),
    Strategy(
        "ema_cross_20_50",
        make_ema_cross("ema20", "ema50"),
        3,
        0.003,
        "EMA 20/50 cross",
    ),
    Strategy("triple_ema", make_triple_ema(), 3, 0.003, "EMA 9/21/50 stacked"),
    Strategy("bb_bounce", make_bb_bounce(), 3, 0.003, "Bollinger band reversion"),
    Strategy(
        "bb_squeeze_breakout",
        make_bb_squeeze_breakout(),
        3,
        0.002,
        "Breakout after volatility contraction",
    ),
    Strategy("supertrend", make_supertrend(), 3, 0.004, "Supertrend 10/3 flips"),
    Strategy(
        "supertrend_fast",
        make_supertrend("st_fast"),
        3,
        0.004,
        "Supertrend 7/2 flips",
    ),
    Strategy("cci_reversion", make_cci_reversion(), 3, 0.003, "Fade CCI +/-100"),
    Strategy("williams_r", make_williams_r(), 3, 0.003, "Fade Williams %R extremes"),
    Strategy(
        "stoch_rsi_cross",
        make_stoch_rsi_cross(),
        3,
        0.003,
        "Stochastic RSI cross in extreme zone",
    ),
    Strategy(
        "donchian_20",
        make_donchian_breakout(),
        3,
        0.003,
        "20-bar channel breakout",
    ),
    Strategy(
        "donchian_55",
        make_donchian_breakout("don_h55", "don_l55"),
        3,
        0.003,
        "55-bar channel breakout",
    ),
    Strategy("keltner_breakout", make_keltner_breakout(), 3, 0.003, "Keltner breakout"),
    Strategy("adx_directional", make_adx_directional(), 3, 0.004, "ADX-confirmed trend"),
    Strategy("hma_cross", make_hma_cross(), 3, 0.003, "Hull average cross"),
    Strategy("roc_momentum", make_roc_momentum(), 3, 0.004, "Rate-of-change momentum"),
    Strategy("vwap_reversion", make_vwap_reversion(), 3, 0.003, "Fade distance to VWAP"),
    Strategy(
        "macd_rsi_combo",
        make_macd_rsi_combo(),
        3,
        0.003,
        "MACD and RSI must agree",
    ),
]

REGISTRY: dict[str, Strategy] = {s.name: s for s in _ALL}


def get(name: str) -> Strategy:
    """Return a strategy by name, raising ``KeyError`` if it does not exist."""
    return REGISTRY[name]


def list_names() -> list[str]:
    """Return every registered strategy name, sorted."""
    return sorted(REGISTRY)


def is_known(name: str) -> bool:
    """Return whether ``name`` resolves to an executable strategy."""
    return name in REGISTRY
