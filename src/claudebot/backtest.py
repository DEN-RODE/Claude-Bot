"""Backtest harness.

The whole value of this module is in four decisions, each of which exists
because the naive version of it produced results that were too good and turned
out to be wrong.

**Entry on the next bar.** A signal computed from bar ``i`` is filled at the
open of bar ``i + 1``, never at ``closes[i]``. Filling at the close of the bar
that produced the signal means buying at a price that was only knowable once the
bar had finished, which is look-ahead bias and is the single most common way a
backtest invents profit that does not exist.

**Round-trip fees are charged on every trade.** Notional times fee rate times
two. At 3x leverage and a 0.05% taker fee this is roughly 0.3% of margin per
trade, which is the difference between a profitable-looking high-frequency
strategy and a real one.

**Stops are checked before targets.** When a bar's range covers both the stop
and the target, intrabar order is unknowable from OHLC data alone. Resolving the
ambiguity in favour of the stop is the conservative choice. Resolving it in
favour of the target is how a backtest reports a win rate it can never reproduce.

**Trades are attributed to a market regime.** A strategy with a 60% win rate
overall may be 75% in trending markets and 40% in ranges. The aggregate hides
the thing worth knowing, so per-regime statistics are reported alongside it.

None of this makes a strategy profitable. It makes the measurement honest, which
is a prerequisite rather than an achievement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .indicators import atr, precompute
from .regime import REGIMES, classify_regime
from .strategies import SignalFn

DEFAULT_MARGIN = 10.0
DEFAULT_FEE_PCT = 0.0005
DEFAULT_STOP_PCT = 0.024
DEFAULT_TARGET_PCT = 0.060
WARMUP_BARS = 40


@dataclass
class Trade:
    """A single closed position."""

    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    regime: str | None = None

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Aggregate statistics for one strategy over one series."""

    trades: list[Trade] = field(default_factory=list)
    stop_pct: float = DEFAULT_STOP_PCT
    target_pct: float = DEFAULT_TARGET_PCT

    @property
    def count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def win_rate(self) -> float:
        """Win rate in percent, or 0.0 when there are no trades."""
        return round(self.wins / self.count * 100, 1) if self.count else 0.0

    @property
    def net_pnl(self) -> float:
        return round(sum(t.pnl for t in self.trades), 2)

    @property
    def avg_pnl(self) -> float:
        return round(self.net_pnl / self.count, 3) if self.count else 0.0

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough drop of the cumulative PnL curve."""
        peak = 0.0
        equity = 0.0
        worst = 0.0
        for trade in self.trades:
            equity += trade.pnl
            peak = max(peak, equity)
            worst = min(worst, equity - peak)
        return round(worst, 2)

    @property
    def profit_factor(self) -> float | None:
        """Gross profit divided by gross loss, or ``None`` with no losses."""
        gains = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = -sum(t.pnl for t in self.trades if t.pnl < 0)
        return round(gains / losses, 2) if losses > 0 else None

    def exit_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in self.trades:
            counts[trade.reason] = counts.get(trade.reason, 0) + 1
        return counts

    def by_regime(self) -> dict[str, dict[str, float]]:
        """Per-regime count, win rate and net PnL."""
        out: dict[str, dict[str, float]] = {}
        for regime in REGIMES:
            subset = [t for t in self.trades if t.regime == regime]
            if subset:
                wins = sum(1 for t in subset if t.is_win)
                out[regime] = {
                    "n": len(subset),
                    "win_rate": round(wins / len(subset) * 100, 1),
                    "net_pnl": round(sum(t.pnl for t in subset), 2),
                }
            else:
                out[regime] = {"n": 0, "win_rate": 0.0, "net_pnl": 0.0}
        return out

    def summary(self) -> dict:
        return {
            "trades": self.count,
            "win_rate": self.win_rate,
            "net_pnl": self.net_pnl,
            "avg_pnl": self.avg_pnl,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "exit_reasons": self.exit_reasons(),
            "by_regime": self.by_regime(),
        }


def run(
    signal_fn: SignalFn,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
    *,
    leverage: int = 3,
    margin: float = DEFAULT_MARGIN,
    fee_pct: float = DEFAULT_FEE_PCT,
    stop_pct: float = DEFAULT_STOP_PCT,
    target_pct: float = DEFAULT_TARGET_PCT,
    min_atr_pct: float = 0.003,
    indicators: dict | None = None,
    regime_labels: Sequence[str | None] | None = None,
) -> BacktestResult:
    """Run ``signal_fn`` over the series and return the resulting statistics.

    ``min_atr_pct`` skips bars whose ATR is too small relative to price for the
    stop and target to be reachable within a sensible horizon. Without it,
    mean-reversion strategies accumulate trades in dead markets that neither win
    nor lose and that dilute every statistic.
    """
    if not (len(closes) == len(highs) == len(lows) == len(volumes)):
        raise ValueError("closes, highs, lows and volumes must be the same length")

    if indicators is None:
        indicators = precompute(closes, highs, lows, volumes)
    atr_series = indicators.get("atr14") or atr(highs, lows, closes, 14)

    notional = margin * leverage
    round_trip_fee = notional * fee_pct * 2

    result = BacktestResult(stop_pct=stop_pct, target_pct=target_pct)
    position: dict | None = None

    # Stop at len-1: an entry signal on the last bar has no next bar to fill on.
    for i in range(WARMUP_BARS, len(closes) - 1):
        if atr_series[i] is None or closes[i] <= 0:
            continue
        if atr_series[i] / closes[i] < min_atr_pct:
            continue

        entry_side, exit_fn = signal_fn(i, closes, highs, lows, volumes, indicators)

        if position is not None:
            entry_price = position["entry_price"]
            side = position["side"]
            hit_stop = hit_target = False

            if side == "BUY":
                if lows[i] <= entry_price * (1 - stop_pct):
                    hit_stop = True
                elif highs[i] >= entry_price * (1 + target_pct):
                    hit_target = True
            else:
                if highs[i] >= entry_price * (1 + stop_pct):
                    hit_stop = True
                elif lows[i] <= entry_price * (1 - target_pct):
                    hit_target = True

            discretionary_exit = (
                bool(exit_fn(i, side, closes, highs, lows, indicators))
                if callable(exit_fn)
                else False
            )

            if hit_stop or hit_target or discretionary_exit:
                if hit_stop:
                    exit_price = entry_price * (
                        (1 - stop_pct) if side == "BUY" else (1 + stop_pct)
                    )
                    reason = "stop"
                elif hit_target:
                    exit_price = entry_price * (
                        (1 + target_pct) if side == "BUY" else (1 - target_pct)
                    )
                    reason = "target"
                else:
                    exit_price = closes[i + 1]
                    reason = "signal"

                move = (
                    (exit_price - entry_price) / entry_price
                    if side == "BUY"
                    else (entry_price - exit_price) / entry_price
                )
                result.trades.append(
                    Trade(
                        entry_index=position["entry_index"],
                        exit_index=i,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=round(notional * move - round_trip_fee, 4),
                        reason=reason,
                        regime=regime_labels[i] if regime_labels else None,
                    )
                )
                position = None

        if position is None and entry_side in ("BUY", "SELL"):
            # Fill on the NEXT bar. See the module docstring.
            position = {
                "side": entry_side,
                "entry_price": closes[i + 1],
                "entry_index": i + 1,
            }

    # A position still open at the end is marked to the last close rather than
    # silently dropped, so the statistics account for every entry taken.
    if position is not None:
        entry_price = position["entry_price"]
        side = position["side"]
        exit_price = closes[-1]
        move = (
            (exit_price - entry_price) / entry_price
            if side == "BUY"
            else (entry_price - exit_price) / entry_price
        )
        result.trades.append(
            Trade(
                entry_index=position["entry_index"],
                exit_index=len(closes) - 1,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=round(notional * move - round_trip_fee, 4),
                reason="open_at_end",
                regime=None,
            )
        )

    return result


def label_regimes(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    indicators: dict,
    reference_closes: Sequence[float] | None = None,
) -> list[str | None]:
    """Classify every bar into a market regime, for per-regime attribution."""
    reference = reference_closes if reference_closes is not None else closes
    return [
        classify_regime(i, closes, reference, indicators) for i in range(len(closes))
    ]
