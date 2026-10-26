"""Tests for the agents, with a fake client and a fake analyzer."""

from __future__ import annotations

import math

import pytest

from claudebot.agents import MarketAnalyzer, PaperTrader, RiskLimits, RiskManager
from claudebot.core.event_bus import EventBus, EventType
from claudebot.core.state import StrategyState
from claudebot.engine import get_indicators


def wave(n: int = 300, amplitude: float = 15.0) -> dict[str, list]:
    closes = [100 + amplitude * math.sin(i / 9) for i in range(n)]
    return {
        "timestamps": list(range(n)),
        "opens": closes,
        "highs": [c + 1.0 for c in closes],
        "lows": [c - 1.0 for c in closes],
        "closes": closes,
        "volumes": [1000.0] * n,
    }


class FakeClient:
    def __init__(self, series: dict[str, list]) -> None:
        self.series = series
        self.calls = 0

    def ohlcv(self, symbol, interval="1h", limit=300):
        self.calls += 1
        return self.series


class FakeAnalyzer:
    def __init__(self, series: dict[str, list]) -> None:
        indicators = get_indicators(
            series["closes"], series["highs"], series["lows"], series["volumes"]
        )
        self._snapshot = {
            "series": series,
            "indicators": indicators,
            "regime": "BULL_RANGE",
            "price": series["closes"][-1],
        }

    def latest(self, _symbol):
        return self._snapshot

    def set_price(self, price: float) -> None:
        self._snapshot = dict(self._snapshot)
        series = dict(self._snapshot["series"])
        closes = list(series["closes"])
        closes[-1] = price
        series["closes"] = closes
        self._snapshot["series"] = series
        self._snapshot["price"] = price


# -- market analyzer -------------------------------------------------------


@pytest.mark.asyncio
async def test_analyzer_publishes_and_caches_a_snapshot():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    await bus.subscribe(EventType.INDICATOR_UPDATE, handler)

    analyzer = MarketAnalyzer(bus, FakeClient(wave()), ["BTCUSDT"])
    await analyzer.tick()

    assert len(received) == 1
    assert received[0].payload["symbol"] == "BTCUSDT"
    snapshot = analyzer.latest("BTCUSDT")
    assert snapshot is not None
    assert "rsi14" in snapshot["indicators"]


@pytest.mark.asyncio
async def test_analyzer_survives_a_failing_client():
    class Broken:
        def ohlcv(self, *_a, **_kw):
            raise ConnectionError("down")

    analyzer = MarketAnalyzer(EventBus(), Broken(), ["BTCUSDT"])
    await analyzer.tick()  # must not raise
    assert analyzer.latest("BTCUSDT") is None


# -- risk manager ----------------------------------------------------------


def test_loss_streak_triggers_a_cooldown():
    risk = RiskManager(EventBus(), RiskLimits(max_consecutive_losses=3))
    for _ in range(3):
        risk.record_trade_result(-1.0)
    assert risk.is_paused
    allowed, reason = risk.can_open(open_positions=0, leverage=3)
    assert not allowed and "cooldown" in reason


def test_a_win_resets_the_streak():
    risk = RiskManager(EventBus(), RiskLimits(max_consecutive_losses=3))
    risk.record_trade_result(-1.0)
    risk.record_trade_result(-1.0)
    risk.record_trade_result(5.0)
    assert risk.consecutive_losses == 0
    assert not risk.is_paused


def test_position_limit_is_enforced():
    risk = RiskManager(EventBus(), RiskLimits(max_open_positions=2))
    assert risk.can_open(open_positions=1, leverage=3)[0] is True
    assert risk.can_open(open_positions=2, leverage=3)[0] is False


def test_leverage_ceiling_is_enforced():
    risk = RiskManager(EventBus(), RiskLimits(max_leverage=5))
    allowed, reason = risk.can_open(open_positions=0, leverage=10)
    assert not allowed and "leverage" in reason


def test_drawdown_blocks_new_positions():
    risk = RiskManager(EventBus(), RiskLimits(max_drawdown_pct=0.20))
    risk.record_equity(1000.0)
    risk.record_equity(750.0)
    assert risk.drawdown_pct() == pytest.approx(0.25)
    assert risk.can_open(open_positions=0, leverage=3)[0] is False


def test_equity_above_the_start_is_not_a_drawdown():
    risk = RiskManager(EventBus(), RiskLimits())
    risk.record_equity(1000.0)
    risk.record_equity(1200.0)
    assert risk.drawdown_pct() == 0.0


@pytest.mark.asyncio
async def test_drawdown_breach_publishes_force_close():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    await bus.subscribe(EventType.FORCE_CLOSE, handler)

    risk = RiskManager(bus, RiskLimits(max_drawdown_pct=0.10))
    risk.record_equity(1000.0)
    risk.record_equity(800.0)
    await risk.tick()

    assert len(received) == 1
    assert received[0].payload["reason"] == "max_drawdown"


# -- paper trader ----------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_trader_opens_and_closes_on_the_stop():
    series = wave()
    analyzer = FakeAnalyzer(series)
    state = StrategyState(
        strategy="rsi_reversion", symbols=["BTCUSDT"], stop_pct=0.02, target_pct=0.05
    )
    trader = PaperTrader(EventBus(), analyzer, state)

    # Force an entry rather than depending on the synthetic series firing one.
    from claudebot.agents.paper_trader import PaperPosition

    trader.positions["BTCUSDT"] = PaperPosition(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=100.0,
        quantity=0.3,
        strategy="rsi_reversion",
    )

    analyzer.set_price(90.0)  # 10% below entry, well past the 2% stop
    await trader.tick()

    assert "BTCUSDT" not in trader.positions
    assert len(trader.results) == 1
    assert trader.results[0].reason == "stop"
    assert trader.results[0].pnl < 0


@pytest.mark.asyncio
async def test_paper_trader_closes_on_the_target():
    from claudebot.agents.paper_trader import PaperPosition

    series = wave()
    analyzer = FakeAnalyzer(series)
    state = StrategyState(
        strategy="rsi_reversion", symbols=["BTCUSDT"], stop_pct=0.02, target_pct=0.05
    )
    trader = PaperTrader(EventBus(), analyzer, state)
    trader.positions["BTCUSDT"] = PaperPosition(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=100.0,
        quantity=0.3,
        strategy="rsi_reversion",
    )

    analyzer.set_price(110.0)
    await trader.tick()

    assert trader.results[0].reason == "target"
    assert trader.results[0].pnl > 0


@pytest.mark.asyncio
async def test_paper_trader_respects_the_position_limit():
    series = wave()
    state = StrategyState(
        strategy="rsi_reversion", symbols=["BTCUSDT"], max_open_positions=0
    )
    trader = PaperTrader(EventBus(), FakeAnalyzer(series), state)
    await trader.tick()
    assert trader.positions == {}


@pytest.mark.asyncio
async def test_paper_trader_skips_symbols_without_a_snapshot():
    class Empty:
        def latest(self, _symbol):
            return None

    state = StrategyState(symbols=["BTCUSDT"])
    trader = PaperTrader(EventBus(), Empty(), state)
    await trader.tick()
    assert trader.positions == {}


def test_paper_trader_statistics():
    from claudebot.agents.paper_trader import PaperResult

    trader = PaperTrader(EventBus(), None, StrategyState())
    trader.results = [
        PaperResult("BTCUSDT", "BUY", 100, 106, 5.0, "macd_flip", "target", 0.0),
        PaperResult("BTCUSDT", "BUY", 100, 98, -2.0, "macd_flip", "stop", 0.0),
        PaperResult("ETHUSDT", "SELL", 100, 95, 4.0, "supertrend", "target", 0.0),
    ]
    stats = trader.stats()
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert stats["net_pnl"] == pytest.approx(7.0)

    per_strategy = trader.stats_by_strategy()
    assert per_strategy["macd_flip"]["trades"] == 2
    assert per_strategy["supertrend"]["win_rate"] == 100.0


def test_paper_trader_statistics_with_no_trades():
    trader = PaperTrader(EventBus(), None, StrategyState())
    assert trader.stats() == {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0}
