"""Wire the whole runtime together and paper trade against live prices.

    export ASTERDEX_API_KEY=... ASTERDEX_API_SECRET=...
    python examples/run_paper_trading.py --strategy supertrend

No orders are sent. The client is used for market data only, every fill is
simulated at the current close, and the same fees a live trade would pay are
charged against the result.

This is the step between a backtest and real money, and skipping it is how a
strategy that looked fine on a clean historical series meets gaps, reconnects
and partially formed bars for the first time with capital behind it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from claudebot import strategies  # noqa: E402
from claudebot.agents import MarketAnalyzer, PaperTrader, RiskLimits, RiskManager  # noqa: E402
from claudebot.clients import AsterdexClient  # noqa: E402
from claudebot.core import EventBus, EventType, StrategyState, Supervisor  # noqa: E402

log = logging.getLogger("paper")


async def report_every(trader: PaperTrader, risk: RiskManager, seconds: float) -> None:
    """Print a running summary so the session is readable while it runs."""
    while True:
        await asyncio.sleep(seconds)
        stats = trader.stats()
        log.info(
            "paper: %d trades, %.1f%% win, net %.4f | open %d | %s",
            stats["trades"],
            stats["win_rate"],
            stats["net_pnl"],
            len(trader.positions),
            "paused" if risk.is_paused else "active",
        )


async def main_async(args) -> int:
    api_key = os.getenv("ASTERDEX_API_KEY", "")
    api_secret = os.getenv("ASTERDEX_API_SECRET", "")
    if not api_key or not api_secret:
        print(
            "ASTERDEX_API_KEY and ASTERDEX_API_SECRET must be set.\n"
            "Copy .env.example to .env and export them, or run "
            "examples/run_backtest.py, which needs no credentials.",
            file=sys.stderr,
        )
        return 2

    if not strategies.is_known(args.strategy):
        print(
            f"unknown strategy {args.strategy!r}. Available:\n  "
            + "\n  ".join(strategies.list_names()),
            file=sys.stderr,
        )
        return 2

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    state = StrategyState(
        strategy=args.strategy,
        symbols=symbols,
        leverage=args.leverage,
        margin_per_trade=args.margin,
        stop_pct=args.stop,
        target_pct=args.target,
        dry_run=True,
    )
    problems = state.validate()
    if problems:
        for problem in problems:
            print(f"configuration error: {problem}", file=sys.stderr)
        return 2

    bus = EventBus()
    client = AsterdexClient(api_key, api_secret)

    analyzer = MarketAnalyzer(bus, client, symbols)
    analyzer.interval = args.analyze_interval
    risk = RiskManager(bus, RiskLimits(max_open_positions=state.max_open_positions))
    trader = PaperTrader(bus, analyzer, state)
    trader.interval = args.trade_interval

    async def on_signal(event) -> None:
        log.info(
            "signal %s %s at %.4f via %s",
            event.payload["side"],
            event.payload["symbol"],
            event.payload["price"],
            event.payload["strategy"],
        )

    await bus.subscribe(EventType.TRADE_SIGNAL, on_signal)

    log.info(
        "paper trading %s on %s, leverage %dx, stop %.1f%%, target %.1f%%",
        state.strategy,
        ", ".join(symbols),
        state.leverage,
        state.stop_pct * 100,
        state.target_pct * 100,
    )

    # One eager fetch so the trader has data on its first tick instead of
    # idling through a whole analyzer interval.
    await analyzer.tick()

    supervisor = Supervisor()
    reporter = asyncio.create_task(report_every(trader, risk, args.report_interval))
    try:
        await supervisor.run([analyzer, risk, trader])
    except asyncio.CancelledError:
        pass
    finally:
        reporter.cancel()
        client.close()
        stats = trader.stats()
        print(
            f"\nFinal: {stats['trades']} trades, {stats['win_rate']}% win rate, "
            f"net {stats['net_pnl']}"
        )
        for name, per in trader.stats_by_strategy().items():
            print(f"  {name}: {per}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="macd_flip")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--stop", type=float, default=0.024)
    parser.add_argument("--target", type=float, default=0.060)
    parser.add_argument("--analyze-interval", type=float, default=300.0)
    parser.add_argument("--trade-interval", type=float, default=60.0)
    parser.add_argument("--report-interval", type=float, default=300.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
