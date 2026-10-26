"""Backtest every registered strategy against the sample series and rank them.

    python examples/run_backtest.py
    python examples/run_backtest.py --strategy supertrend --verbose

The numbers this prints describe a synthetic random walk. They say something
about whether the harness works and nothing whatsoever about whether any of
these strategies make money. Point it at real candles before drawing a
conclusion, and read the methodology notes in ``claudebot/backtest.py`` first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from claudebot import backtest, strategies  # noqa: E402
from claudebot.indicators import precompute  # noqa: E402
from generate_sample_data import generate, load_csv  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "sample_BTCUSDT_1h.csv"


def load_series(path: Path) -> dict[str, list]:
    if path.exists():
        return load_csv(path)
    rows = generate()
    return {
        "closes": [r["close"] for r in rows],
        "highs": [r["high"] for r in rows],
        "lows": [r["low"] for r in rows],
        "volumes": [r["volume"] for r in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--strategy", help="run only this strategy")
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--stop", type=float, default=backtest.DEFAULT_STOP_PCT)
    parser.add_argument("--target", type=float, default=backtest.DEFAULT_TARGET_PCT)
    parser.add_argument("--verbose", action="store_true", help="show regime breakdown")
    args = parser.parse_args()

    series = load_series(args.data)
    closes, highs = series["closes"], series["highs"]
    lows, volumes = series["lows"], series["volumes"]

    print(f"Series: {len(closes)} bars, {closes[0]:,.0f} to {closes[-1]:,.0f}")
    print(
        f"Settings: leverage {args.leverage}x, margin {args.margin:.0f}, "
        f"stop {args.stop:.1%}, target {args.target:.1%}\n"
    )

    # Computed once and shared: the indicators do not depend on the strategy.
    indicators = precompute(closes, highs, lows, volumes)
    regimes = backtest.label_regimes(closes, highs, lows, indicators)

    names = [args.strategy] if args.strategy else strategies.list_names()
    rows = []
    for name in names:
        strategy = strategies.get(name)
        result = backtest.run(
            strategy.signal_fn,
            closes,
            highs,
            lows,
            volumes,
            leverage=args.leverage,
            margin=args.margin,
            stop_pct=args.stop,
            target_pct=args.target,
            min_atr_pct=strategy.min_atr_pct,
            indicators=indicators,
            regime_labels=regimes,
        )
        rows.append((name, result))

    rows.sort(key=lambda r: r[1].net_pnl, reverse=True)

    header = f"{'strategy':<22} {'n':>4} {'win%':>6} {'net':>9} {'avg':>8} {'PF':>6} {'maxDD':>8}"
    print(header)
    print("-" * len(header))
    for name, result in rows:
        pf = result.profit_factor
        print(
            f"{name:<22} {result.count:>4} {result.win_rate:>6.1f} "
            f"{result.net_pnl:>9.2f} {result.avg_pnl:>8.3f} "
            f"{(f'{pf:.2f}' if pf is not None else '-'):>6} "
            f"{result.max_drawdown:>8.2f}"
        )

    traded = [r for _n, r in rows if r.count > 0]
    print(
        f"\n{len(traded)} of {len(rows)} strategies produced trades. "
        "Net and avg are in quote currency on the configured margin."
    )

    if args.verbose:
        for name, result in rows:
            if result.count == 0:
                continue
            print(f"\n{name} by regime:")
            for regime, stats in result.by_regime().items():
                if stats["n"]:
                    print(
                        f"  {regime:<12} n={stats['n']:<4} "
                        f"win%={stats['win_rate']:<6.1f} net={stats['net_pnl']:.2f}"
                    )
            print(f"  exits: {result.exit_reasons()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
