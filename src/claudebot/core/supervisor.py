"""Supervision for long-running agent coroutines.

An agent that raises takes its coroutine down with it. Without supervision the
rest of the system carries on, unaware, and the failure surfaces hours later as
"why did it stop trading". The supervisor restarts a crashed agent after a
backoff and records what happened.

Two details that matter in practice. ``asyncio.CancelledError`` is re-raised
rather than treated as a crash, because cancellation is how shutdown works and
swallowing it produces a task that cannot be stopped. And an agent that crashes
repeatedly in quick succession is given up on rather than restarted forever,
since a restart loop on a misconfigured agent will happily burn an API rate
limit all night.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_RESTART_DELAY = 10.0
DEFAULT_MAX_RESTARTS = 20
RESTART_WINDOW_SECONDS = 3600.0


@dataclass
class AgentHealth:
    """Restart bookkeeping for one supervised agent."""

    name: str
    restarts: int = 0
    last_error: str | None = None
    last_restart_at: float | None = None
    recent_restarts: list[float] = field(default_factory=list)
    given_up: bool = False

    def record_restart(self, error: str) -> None:
        now = time.monotonic()
        self.restarts += 1
        self.last_error = error
        self.last_restart_at = now
        self.recent_restarts.append(now)
        cutoff = now - RESTART_WINDOW_SECONDS
        self.recent_restarts = [t for t in self.recent_restarts if t >= cutoff]


class Supervisor:
    """Run agents concurrently and restart the ones that crash."""

    def __init__(
        self,
        restart_delay: float = DEFAULT_RESTART_DELAY,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
    ) -> None:
        self.restart_delay = restart_delay
        self.max_restarts = max_restarts
        self.health: dict[str, AgentHealth] = {}
        self._tasks: list[asyncio.Task] = []

    async def _supervise(self, agent) -> None:
        health = self.health.setdefault(agent.name, AgentHealth(name=agent.name))
        while True:
            try:
                await agent.run()
                log.info("agent %s finished cleanly", agent.name)
                return
            except asyncio.CancelledError:
                # Shutdown, not failure. Must propagate.
                raise
            except Exception as exc:
                health.record_restart(f"{type(exc).__name__}: {exc}")
                if len(health.recent_restarts) > self.max_restarts:
                    health.given_up = True
                    log.error(
                        "agent %s crashed %d times within the restart window, "
                        "giving up",
                        agent.name,
                        len(health.recent_restarts),
                    )
                    return
                log.exception(
                    "agent %s crashed, restarting in %.0fs (restart %d)",
                    agent.name,
                    self.restart_delay,
                    health.restarts,
                )
                await asyncio.sleep(self.restart_delay)

    async def run(self, agents) -> None:
        """Run every agent until cancelled."""
        self._tasks = [
            asyncio.create_task(self._supervise(agent), name=agent.name)
            for agent in agents
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Cancel every supervised task and wait for them to unwind."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def report(self) -> dict[str, dict]:
        """Health summary, suitable for a status endpoint or a log line."""
        return {
            name: {
                "restarts": h.restarts,
                "last_error": h.last_error,
                "given_up": h.given_up,
            }
            for name, h in self.health.items()
        }
