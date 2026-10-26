"""Account-level risk limits, enforced independently of any strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.event_bus import Event, EventBus, EventType
from .base import Agent

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Hard limits. These exist to be boring and to never be relaxed at runtime.

    A strategy that is losing will always produce a reason to widen a stop or
    add to a position, and a system that lets it do so has no risk management,
    only risk opinions. These are checked by an agent that cannot see the
    strategy's reasoning and does not care about it.
    """

    #: Fraction of starting equity at which everything is closed.
    max_drawdown_pct: float = 0.20
    #: Consecutive losses before new entries are paused.
    max_consecutive_losses: int = 4
    #: Pause duration once that streak is hit, in seconds.
    cooldown_seconds: float = 6 * 3600
    #: Simultaneous open positions.
    max_open_positions: int = 3
    #: Largest acceptable leverage, regardless of what a strategy asks for.
    max_leverage: int = 5


class RiskManager(Agent):
    """Watches equity and the loss streak, and can veto or force-close."""

    interval = 60.0

    def __init__(
        self,
        bus: EventBus,
        limits: RiskLimits | None = None,
        starting_equity: float | None = None,
    ) -> None:
        super().__init__("risk_manager", bus)
        self.limits = limits or RiskLimits()
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.consecutive_losses = 0
        self.paused_until: float | None = None
        self._clock = __import__("time").monotonic

    # -- state updates ---------------------------------------------------

    def record_equity(self, equity: float) -> None:
        if self.starting_equity is None:
            self.starting_equity = equity
        self.current_equity = equity

    def record_trade_result(self, pnl: float) -> None:
        """Update the loss streak and start a cooldown when it is breached."""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.limits.max_consecutive_losses:
                self.paused_until = self._clock() + self.limits.cooldown_seconds
                log.warning(
                    "%d consecutive losses, pausing entries for %.0f minutes",
                    self.consecutive_losses,
                    self.limits.cooldown_seconds / 60,
                )
        else:
            self.consecutive_losses = 0

    # -- checks ----------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self.paused_until is not None and self._clock() < self.paused_until

    def drawdown_pct(self) -> float:
        """Current drawdown from starting equity, as a positive fraction."""
        if not self.starting_equity or self.current_equity is None:
            return 0.0
        if self.current_equity >= self.starting_equity:
            return 0.0
        return (self.starting_equity - self.current_equity) / self.starting_equity

    def can_open(self, open_positions: int, leverage: int) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a proposed new position."""
        if self.is_paused:
            return False, "cooldown after consecutive losses"
        if open_positions >= self.limits.max_open_positions:
            return False, "maximum open positions reached"
        if leverage > self.limits.max_leverage:
            return False, f"leverage {leverage} exceeds limit {self.limits.max_leverage}"
        if self.drawdown_pct() >= self.limits.max_drawdown_pct:
            return False, "maximum drawdown reached"
        return True, ""

    async def tick(self) -> None:
        if self.drawdown_pct() >= self.limits.max_drawdown_pct:
            log.error(
                "drawdown %.1f%% at or beyond limit, requesting force close",
                self.drawdown_pct() * 100,
            )
            await self.bus.publish(
                Event(
                    type=EventType.FORCE_CLOSE,
                    source=self.name,
                    payload={
                        "reason": "max_drawdown",
                        "drawdown_pct": round(self.drawdown_pct() * 100, 2),
                    },
                )
            )
