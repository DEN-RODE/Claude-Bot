"""Persistent strategy configuration and trade records.

The write path is the interesting part. An earlier version of this project had
several agents writing one JSON file directly, and a crash partway through a
write left a truncated file that the next start could not parse, taking the
configuration with it. :meth:`StrategyState.save` writes to a temporary file in
the same directory and then replaces the target atomically, so a reader either
sees the old file or the new one and never a half-written one.

This is still a single-writer design. Coordinating several processes needs a
real lock or a real database, and pretending otherwise with JSON is how the
earlier version lost its configuration.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_STOP_PCT = 0.024
DEFAULT_TARGET_PCT = 0.060


@dataclass
class TradeRecord:
    """One closed position, as recorded by the trader."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    opened_at: float
    closed_at: float
    strategy: str
    reason: str = "signal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StrategyState:
    """Live trading configuration, persisted between restarts."""

    strategy: str = "macd_flip"
    strategy_by_symbol: dict[str, str] = field(default_factory=dict)
    leverage: int = 3
    stop_pct: float = DEFAULT_STOP_PCT
    target_pct: float = DEFAULT_TARGET_PCT
    margin_per_trade: float = 10.0
    max_open_positions: int = 3
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    dry_run: bool = True

    def strategy_for(self, symbol: str) -> str:
        """Per-symbol strategy if one is set, otherwise the global one."""
        return self.strategy_by_symbol.get(symbol, self.strategy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "StrategyState":
        """Load state from ``path``, returning defaults if it does not exist."""
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        """Write state atomically: temp file in the same directory, then replace."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, p)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def validate(self) -> list[str]:
        """Return a list of configuration problems, empty when valid."""
        problems: list[str] = []
        if self.leverage < 1:
            problems.append("leverage must be at least 1")
        if not 0 < self.stop_pct < 1:
            problems.append("stop_pct must be between 0 and 1")
        if not 0 < self.target_pct < 1:
            problems.append("target_pct must be between 0 and 1")
        if self.target_pct <= self.stop_pct:
            problems.append("target_pct must be greater than stop_pct")
        if self.margin_per_trade <= 0:
            problems.append("margin_per_trade must be positive")
        if self.max_open_positions < 1:
            problems.append("max_open_positions must be at least 1")
        if not self.symbols:
            problems.append("at least one symbol is required")
        return problems
