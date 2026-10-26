"""Paper trading: run strategies on live prices without sending orders.

The point of this agent is to close the gap between a backtest and production.
A backtest sees a clean series; live trading sees gaps, reconnects and bars that
have not finished forming. Running the same strategy code against live data with
no money at risk is what tells you whether a backtested edge survives contact
with the feed, before it costs anything to find out.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..core.event_bus import Event, EventBus, EventType
from ..engine import evaluate_entry, evaluate_exit, strategy_meta
from .base import Agent

log = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    strategy: str
    opened_at: float = field(default_factory=time.time)


@dataclass
class PaperResult:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    strategy: str
    reason: str
    opened_at: float
    closed_at: float = field(default_factory=time.time)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


class PaperTrader(Agent):
    """Simulates fills at the current close, charging the same fees as live."""

    interval = 60.0

    def __init__(
        self,
        bus: EventBus,
        analyzer,
        state,
        fee_pct: float = 0.0005,
    ) -> None:
        super().__init__("paper_trader", bus)
        self.analyzer = analyzer
        self.state = state
        self.fee_pct = fee_pct
        self.positions: dict[str, PaperPosition] = {}
        self.results: list[PaperResult] = []

    # -- statistics ------------------------------------------------------

    def stats(self) -> dict:
        if not self.results:
            return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0}
        wins = sum(1 for r in self.results if r.is_win)
        return {
            "trades": len(self.results),
            "win_rate": round(wins / len(self.results) * 100, 1),
            "net_pnl": round(sum(r.pnl for r in self.results), 2),
        }

    def stats_by_strategy(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name in {r.strategy for r in self.results}:
            subset = [r for r in self.results if r.strategy == name]
            wins = sum(1 for r in subset if r.is_win)
            out[name] = {
                "trades": len(subset),
                "win_rate": round(wins / len(subset) * 100, 1),
                "net_pnl": round(sum(r.pnl for r in subset), 2),
            }
        return out

    # -- trading ---------------------------------------------------------

    def _close(self, symbol: str, price: float, reason: str) -> None:
        position = self.positions.pop(symbol)
        move = (
            (price - position.entry_price) / position.entry_price
            if position.side == "BUY"
            else (position.entry_price - price) / position.entry_price
        )
        notional = position.quantity * position.entry_price
        pnl = notional * move - notional * self.fee_pct * 2
        self.results.append(
            PaperResult(
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=round(pnl, 4),
                strategy=position.strategy,
                reason=reason,
                opened_at=position.opened_at,
            )
        )
        log.info(
            "paper close %s %s at %.4f (%s), pnl %.4f",
            symbol,
            position.side,
            price,
            reason,
            pnl,
        )

    async def tick(self) -> None:
        for symbol in self.state.symbols:
            snapshot = self.analyzer.latest(symbol)
            if snapshot is None:
                continue

            series = snapshot["series"]
            closes = series["closes"]
            highs = series["highs"]
            lows = series["lows"]
            volumes = series["volumes"]
            price = closes[-1]
            strategy_name = self.state.strategy_for(symbol)

            position = self.positions.get(symbol)
            if position is not None:
                stop_hit = (
                    price <= position.entry_price * (1 - self.state.stop_pct)
                    if position.side == "BUY"
                    else price >= position.entry_price * (1 + self.state.stop_pct)
                )
                target_hit = (
                    price >= position.entry_price * (1 + self.state.target_pct)
                    if position.side == "BUY"
                    else price <= position.entry_price * (1 - self.state.target_pct)
                )
                if stop_hit:
                    self._close(symbol, price, "stop")
                elif target_hit:
                    self._close(symbol, price, "target")
                elif evaluate_exit(
                    strategy_name, position.side, closes, highs, lows, volumes
                ):
                    self._close(symbol, price, "signal")
                continue

            if len(self.positions) >= self.state.max_open_positions:
                continue

            side = evaluate_entry(strategy_name, closes, highs, lows, volumes)
            if side is None:
                continue

            meta = strategy_meta(strategy_name)
            atr_value = snapshot["indicators"]["atr14"][-1]
            if atr_value is None or price <= 0:
                continue
            if atr_value / price < meta.min_atr_pct:
                continue

            notional = self.state.margin_per_trade * self.state.leverage
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=notional / price,
                strategy=strategy_name,
            )
            log.info("paper open %s %s at %.4f (%s)", symbol, side, price, strategy_name)
            await self.bus.publish(
                Event(
                    type=EventType.TRADE_SIGNAL,
                    source=self.name,
                    payload={
                        "symbol": symbol,
                        "side": side,
                        "price": price,
                        "strategy": strategy_name,
                        "paper": True,
                    },
                )
            )
