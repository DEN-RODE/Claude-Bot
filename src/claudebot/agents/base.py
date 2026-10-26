"""Base class for supervised agents."""

from __future__ import annotations

import asyncio
import logging

from ..core.event_bus import EventBus

log = logging.getLogger(__name__)


class Agent:
    """A named coroutine that wakes on an interval and does one unit of work.

    Subclasses implement :meth:`tick`. :meth:`run` owns the loop so that the
    supervisor has a single thing to restart, and so an exception inside one
    tick is never confused with the loop ending.
    """

    #: Seconds between ticks.
    interval: float = 60.0

    def __init__(self, name: str, bus: EventBus) -> None:
        self.name = name
        self.bus = bus
        self.ticks = 0

    async def tick(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def run(self) -> None:
        """Loop until cancelled, ticking on the configured interval."""
        log.info("agent %s started (interval %.0fs)", self.name, self.interval)
        while True:
            await self.tick()
            self.ticks += 1
            await asyncio.sleep(self.interval)
