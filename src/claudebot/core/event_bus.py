"""An asyncio publish/subscribe bus for loosely coupled agents.

The one subtlety worth reading the code for: subscribers are snapshotted under
the lock and then invoked *outside* it. Dispatching while holding the lock
deadlocks the moment any handler publishes an event of its own, which is a
normal thing for a handler to do and an unpleasant thing to debug, since the
process simply stops with no traceback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


class EventType(str, Enum):
    MARKET_ANALYSIS = "market_analysis"
    INDICATOR_UPDATE = "indicator_update"
    TRADE_SIGNAL = "trade_signal"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_ALERT = "risk_alert"
    FORCE_CLOSE = "force_close"
    STRATEGY_CHANGED = "strategy_changed"


@dataclass(frozen=True)
class Event:
    """An immutable message on the bus."""

    type: EventType
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Fan-out message bus with a bounded history for inspection."""

    def __init__(self, history_size: int = 1000) -> None:
        self._subscribers: dict[EventType, list[Handler]] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register ``handler`` for ``event_type``."""
        async with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    async def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a previously registered handler, ignoring unknown ones."""
        async with self._lock:
            handlers = self._subscribers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Deliver ``event`` to every subscriber of its type.

        A handler that raises is logged and does not prevent the remaining
        handlers from running: one broken consumer must not take down the bus.
        """
        async with self._lock:
            self._history.append(event)
            handlers = list(self._subscribers.get(event.type, []))

        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                log.exception(
                    "handler %r failed on %s from %s",
                    getattr(handler, "__qualname__", handler),
                    event.type.value,
                    event.source,
                )

    def recent(self, event_type: EventType | None = None, limit: int = 50) -> list[Event]:
        """Return the most recent events, newest last."""
        events = list(self._history)
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def subscriber_count(self, event_type: EventType) -> int:
        return len(self._subscribers.get(event_type, []))
