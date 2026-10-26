"""Fetches candles, computes indicators and publishes them."""

from __future__ import annotations

import asyncio
import logging

from ..core.event_bus import Event, EventBus, EventType
from ..engine import get_indicators
from ..regime import classify_regime
from .base import Agent

log = logging.getLogger(__name__)


class MarketAnalyzer(Agent):
    """Single source of market data for every other agent.

    Agents that each fetched their own candles would multiply the rate-limit
    cost by the number of agents and, worse, disagree with each other about the
    current price. One fetcher, one snapshot, everyone reads the same numbers.
    """

    interval = 300.0

    def __init__(
        self,
        bus: EventBus,
        client,
        symbols: list[str],
        interval_label: str = "1h",
        candle_limit: int = 300,
        reference_symbol: str = "BTCUSDT",
    ) -> None:
        super().__init__("market_analyzer", bus)
        self.client = client
        self.symbols = symbols
        self.interval_label = interval_label
        self.candle_limit = candle_limit
        self.reference_symbol = reference_symbol
        self._snapshots: dict[str, dict] = {}

    def latest(self, symbol: str) -> dict | None:
        """Most recent snapshot for ``symbol``, or ``None`` if not fetched yet."""
        return self._snapshots.get(symbol)

    async def tick(self) -> None:
        reference_closes: list[float] | None = None

        for symbol in self.symbols:
            try:
                series = await asyncio.to_thread(
                    self.client.ohlcv, symbol, self.interval_label, self.candle_limit
                )
            except Exception:
                log.exception("failed to fetch candles for %s", symbol)
                continue

            indicators = get_indicators(
                series["closes"], series["highs"], series["lows"], series["volumes"]
            )
            if symbol == self.reference_symbol:
                reference_closes = series["closes"]

            regime = classify_regime(
                len(series["closes"]) - 1,
                series["closes"],
                reference_closes or series["closes"],
                indicators,
            )

            self._snapshots[symbol] = {
                "series": series,
                "indicators": indicators,
                "regime": regime,
                "price": series["closes"][-1],
            }

            await self.bus.publish(
                Event(
                    type=EventType.INDICATOR_UPDATE,
                    source=self.name,
                    payload={
                        "symbol": symbol,
                        "price": series["closes"][-1],
                        "regime": regime,
                    },
                )
            )
