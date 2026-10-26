"""Tests for the event bus, the supervisor and persisted state."""

from __future__ import annotations

import asyncio
import json

import pytest

from claudebot.core.event_bus import Event, EventBus, EventType
from claudebot.core.state import StrategyState, TradeRecord
from claudebot.core.supervisor import Supervisor


# -- event bus -------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscriber_receives_published_event():
    bus = EventBus()
    received = []

    async def handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe(EventType.TRADE_SIGNAL, handler)
    await bus.publish(
        Event(type=EventType.TRADE_SIGNAL, source="test", payload={"symbol": "BTCUSDT"})
    )

    assert len(received) == 1
    assert received[0].payload["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_events_are_only_delivered_to_matching_subscribers():
    bus = EventBus()
    signals, risks = [], []

    async def on_signal(event):
        signals.append(event)

    async def on_risk(event):
        risks.append(event)

    await bus.subscribe(EventType.TRADE_SIGNAL, on_signal)
    await bus.subscribe(EventType.RISK_ALERT, on_risk)
    await bus.publish(Event(type=EventType.TRADE_SIGNAL, source="t"))

    assert len(signals) == 1
    assert risks == []


@pytest.mark.asyncio
async def test_a_failing_handler_does_not_stop_the_others():
    bus = EventBus()
    survived = []

    async def broken(_event):
        raise RuntimeError("boom")

    async def healthy(event):
        survived.append(event)

    await bus.subscribe(EventType.TRADE_SIGNAL, broken)
    await bus.subscribe(EventType.TRADE_SIGNAL, healthy)
    await bus.publish(Event(type=EventType.TRADE_SIGNAL, source="t"))

    assert len(survived) == 1


@pytest.mark.asyncio
async def test_a_handler_may_publish_without_deadlocking():
    """Dispatching while holding the lock would hang here forever."""
    bus = EventBus()
    seen = []

    async def republisher(_event):
        await bus.publish(Event(type=EventType.RISK_ALERT, source="handler"))

    async def on_risk(event):
        seen.append(event)

    await bus.subscribe(EventType.TRADE_SIGNAL, republisher)
    await bus.subscribe(EventType.RISK_ALERT, on_risk)

    await asyncio.wait_for(
        bus.publish(Event(type=EventType.TRADE_SIGNAL, source="t")), timeout=2.0
    )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    await bus.subscribe(EventType.TRADE_SIGNAL, handler)
    await bus.unsubscribe(EventType.TRADE_SIGNAL, handler)
    await bus.publish(Event(type=EventType.TRADE_SIGNAL, source="t"))

    assert received == []


@pytest.mark.asyncio
async def test_history_is_bounded():
    bus = EventBus(history_size=5)
    for i in range(20):
        await bus.publish(
            Event(type=EventType.INDICATOR_UPDATE, source="t", payload={"i": i})
        )
    recent = bus.recent()
    assert len(recent) == 5
    assert recent[-1].payload["i"] == 19


# -- supervisor ------------------------------------------------------------


class CrashingAgent:
    def __init__(self, name: str, crashes: int) -> None:
        self.name = name
        self.remaining = crashes
        self.runs = 0

    async def run(self) -> None:
        self.runs += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("scheduled crash")


@pytest.mark.asyncio
async def test_supervisor_restarts_a_crashing_agent():
    agent = CrashingAgent("flaky", crashes=2)
    supervisor = Supervisor(restart_delay=0.01)
    await asyncio.wait_for(supervisor.run([agent]), timeout=5.0)

    assert agent.runs == 3  # two crashes, then a clean finish
    assert supervisor.health["flaky"].restarts == 2


@pytest.mark.asyncio
async def test_supervisor_gives_up_after_too_many_crashes():
    agent = CrashingAgent("hopeless", crashes=1000)
    supervisor = Supervisor(restart_delay=0.001, max_restarts=3)
    await asyncio.wait_for(supervisor.run([agent]), timeout=5.0)

    assert supervisor.health["hopeless"].given_up is True
    assert supervisor.report()["hopeless"]["given_up"] is True


@pytest.mark.asyncio
async def test_cancellation_is_not_treated_as_a_crash():
    class Forever:
        name = "forever"

        async def run(self):
            await asyncio.sleep(3600)

    supervisor = Supervisor(restart_delay=0.01)
    task = asyncio.create_task(supervisor.run([Forever()]))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert supervisor.health.get("forever", None) is None or (
        supervisor.health["forever"].restarts == 0
    )


# -- state -----------------------------------------------------------------


def test_state_round_trips_through_disk(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = StrategyState(strategy="supertrend", leverage=4, symbols=["ETHUSDT"])
    state.save(path)

    loaded = StrategyState.load(path)
    assert loaded.strategy == "supertrend"
    assert loaded.leverage == 4
    assert loaded.symbols == ["ETHUSDT"]


def test_loading_a_missing_file_returns_defaults(tmp_path):
    state = StrategyState.load(tmp_path / "absent.json")
    assert state.strategy == "macd_flip"
    assert state.dry_run is True


def test_save_leaves_no_temporary_files_behind(tmp_path):
    path = tmp_path / "state.json"
    StrategyState().save(path)
    StrategyState().save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_saved_file_is_valid_json(tmp_path):
    path = tmp_path / "state.json"
    StrategyState().save(path)
    json.loads(path.read_text(encoding="utf-8"))


def test_unknown_keys_are_ignored_on_load(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"strategy": "supertrend", "legacy_field": 1}), encoding="utf-8"
    )
    state = StrategyState.load(path)
    assert state.strategy == "supertrend"


def test_per_symbol_strategy_overrides_the_global_one():
    state = StrategyState(strategy="macd_flip", strategy_by_symbol={"ETHUSDT": "supertrend"})
    assert state.strategy_for("ETHUSDT") == "supertrend"
    assert state.strategy_for("BTCUSDT") == "macd_flip"


def test_validate_accepts_defaults():
    assert StrategyState().validate() == []


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"leverage": 0}, "leverage"),
        ({"stop_pct": 0}, "stop_pct"),
        ({"target_pct": 1.5}, "target_pct"),
        ({"stop_pct": 0.10, "target_pct": 0.05}, "greater than"),
        ({"margin_per_trade": 0}, "margin_per_trade"),
        ({"symbols": []}, "symbol"),
    ],
)
def test_validate_rejects_bad_configuration(kwargs, fragment):
    problems = StrategyState(**kwargs).validate()
    assert any(fragment in p for p in problems), problems


def test_trade_record_round_trips():
    record = TradeRecord(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=100.0,
        exit_price=106.0,
        quantity=0.3,
        pnl=1.8,
        opened_at=1.0,
        closed_at=2.0,
        strategy="macd_flip",
    )
    assert TradeRecord.from_dict(record.to_dict()) == record
