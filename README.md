# Claude-Bot

A strategy research and execution framework for crypto perpetual futures, in
plain Python. Twenty-three signal strategies built on an indicator library
written from scratch, a backtest harness that is careful about the things that
make backtests lie, and an asyncio runtime that runs the same strategy code live
that the backtester validated.

No numpy, no pandas, no TA-Lib. Two runtime dependencies, both for HTTP and
configuration.

```bash
git clone https://github.com/DEN-RODE/Claude-Bot.git
cd Claude-Bot
pip install -r requirements-dev.txt
python -m pytest                  # 99 tests, no network
python examples/run_backtest.py   # ranks every strategy, no credentials needed
```

---

## Why the backtest harness is the interesting part

Writing a backtester is easy. Writing one that does not flatter you is not. Four
decisions in `src/claudebot/backtest.py` exist because the naive version of each
produced results that were better than reality.

**Entry fills on the next bar.** A signal computed from bar `i` is filled at
`closes[i + 1]`, never at `closes[i]`. Filling at the close of the bar that
produced the signal means transacting at a price that was only knowable once
that bar had finished. This is look-ahead bias and it is the single most common
way a backtest invents profit that cannot be reproduced. There is a test that
fails if anyone reintroduces it.

**Round-trip fees are charged on every trade.** Notional times fee rate times
two. At 3x leverage and a 0.05% taker fee that is about 0.3% of margin per
round trip, which is the entire difference between a profitable-looking
high-frequency strategy and a real one. A test opens and closes a position at an
unchanged price and asserts the loss equals the fee exactly.

**Stops are checked before targets.** When one bar's range covers both the stop
and the target, OHLC data cannot say which was touched first. Resolving that
ambiguity in favour of the target inflates the win rate by an amount that scales
with volatility. Resolving it in favour of the stop is conservative and
reproducible, so that is what happens, and a test constructs exactly that bar to
prove it.

**Trades are attributed to a market regime.** A strategy at 60% overall may be
75% in trends and 40% in ranges, and the aggregate hides the only thing worth
knowing. Every bar is classified into one of six regimes and results are broken
down by it. The percentiles that drive the classification are computed over a
trailing window rather than the whole series, because full-series percentiles
would let a bar's label depend on volatility that had not happened yet, which is
look-ahead bias smuggled in through the labels.

None of this makes a strategy profitable. It makes the measurement honest, which
is a prerequisite rather than an achievement.

```
$ python examples/run_backtest.py

Series: 1200 bars, 42,186 to 44,501
Settings: leverage 3x, margin 10, stop 2.4%, target 6.0%

strategy                  n   win%       net      avg     PF    maxDD
---------------------------------------------------------------------
donchian_20              26   53.8     14.48    0.557   2.61    -4.50
adx_directional          25   52.0     14.24    0.570   2.62    -3.00
keltner_breakout         19   57.9     13.75    0.724   3.41    -2.72
...
cci_reversion            93   40.9    -23.55   -0.253   0.32   -24.07
williams_r               98   36.7    -26.41   -0.269   0.26   -26.41
```

Those numbers describe a **synthetic random walk**, generated with a fixed seed
by `scripts/generate_sample_data.py`. They demonstrate that the harness runs.
They say nothing about whether any of these strategies make money, and the fact
that breakout strategies rank above mean-reversion ones here is a property of
the generator, not a finding. Point it at real candles before drawing any
conclusion.

## The closed-bar rule

`src/claudebot/engine.py` drops the last element of every series before
evaluating anything, because in live trading the last bar is still forming.

This sounds pedantic and is not. An indicator computed mid-bar is not the
indicator the backtest saw. Supertrend and Donchian in particular flip several
times inside an hourly bar and then settle, so a bot polling every thirty
seconds fires a burst of entries and exits on a signal that never existed once
the bar closed. The backtest, which only ever sees completed bars, shows none of
this and cannot warn you about it.

There is a test that spikes the final bar by 50% and asserts that not one of the
twenty-three strategies changes its decision.

## One registry

Backtest, paper trading and live execution all resolve strategies through the
same `REGISTRY` in `src/claudebot/strategies.py`.

An earlier iteration of this project did not. The backtester had its own
strategy implementations, the live trader had a smaller set, and a lookup table
mapped one onto the other. The consequence was that a strategy validated under
one name ran different logic in production, and the live results were therefore
measuring something nobody had tested. A name that backtests now runs the same
code live, or it raises `KeyError` and refuses to run at all.

## Architecture

```
src/claudebot/
├── indicators.py     26 indicators on plain lists, plus precompute()
├── strategies.py     23 strategies, factories and the shared registry
├── backtest.py       the harness, the four decisions above
├── regime.py         six-regime classification
├── engine.py         closed-bar evaluation, shared by paper and live
├── core/
│   ├── event_bus.py  asyncio pub/sub
│   ├── supervisor.py restart-on-crash for agent coroutines
│   └── state.py      atomic JSON persistence
├── clients/
│   └── asterdex.py   signed REST client for a Binance-compatible venue
└── agents/
    ├── market_analyzer.py  one fetcher, one snapshot, shared by all
    ├── risk_manager.py     limits enforced independently of any strategy
    └── paper_trader.py     live prices, simulated fills, real fees
```

Three details worth opening the file for.

**The event bus snapshots its subscribers under the lock and dispatches outside
it.** Dispatching while holding the lock deadlocks the moment a handler
publishes an event of its own, which is a normal thing for a handler to do and
an unpleasant thing to debug, because the process simply stops with no
traceback. There is a test that would hang forever on the naive version.

**The client never retries an order.** Transient-error retries are correct for
reads and dangerous for writes: a gateway timeout returned *after* the exchange
has accepted an order turns one position into two. Reads use a session with a
retry adapter, writes use one without, and a test asserts a 504 on an order
raises rather than producing a second request.

**The signature covers exactly the bytes that are sent.** The query string is
built once, signed, and appended to the URL verbatim. Signing a re-sorted copy
of the parameters and handing the dict to the HTTP library is wrong whenever the
library serialises in a different order, and it fails silently: requests with two
parameters work and requests with six are rejected as badly signed, which looks
like an intermittent auth problem for as long as it takes to find. The test
recomputes the HMAC over the sent query and requires a match.

**Risk limits are enforced by an agent that cannot see the strategy's
reasoning.** A losing strategy will always produce a reason to widen a stop or
add to a position. A system that lets it do so has risk opinions, not risk
management.

## Usage

Backtest everything against the sample series:

```bash
python examples/run_backtest.py
python examples/run_backtest.py --strategy supertrend --verbose   # regime breakdown
python examples/run_backtest.py --leverage 2 --stop 0.03 --target 0.05
```

Paper trade against live prices, sending no orders:

```bash
cp .env.example .env     # then fill in and export, or use your own loader
python examples/run_paper_trading.py --strategy supertrend --symbols BTCUSDT
```

Use the library directly:

```python
from claudebot import backtest, strategies
from claudebot.indicators import precompute

strategy = strategies.get("donchian_20")
indicators = precompute(closes, highs, lows, volumes)
regimes = backtest.label_regimes(closes, highs, lows, indicators)

result = backtest.run(
    strategy.signal_fn, closes, highs, lows, volumes,
    leverage=strategy.leverage,
    min_atr_pct=strategy.min_atr_pct,
    indicators=indicators,
    regime_labels=regimes,
)
print(result.summary())
```

Evaluate a live signal on the last closed bar:

```python
from claudebot import engine

side = engine.evaluate_entry("supertrend", closes, highs, lows, volumes)
```

## What is deliberately not here

**Live order execution is not wired up.** `AsterdexClient` can place, stop and
cancel orders and is tested, but no agent calls those methods. Paper trading is
the end of the supplied path. Connecting the last wire is a short piece of work
and an entirely different level of responsibility, and a repository that ships
it invites people to run it before they have read any of the above.

**There is no machine learning.** An earlier version of this project had a class
called `MLLearningAgent`, and what it actually computed was a Pearson correlation
between six indicator values and trade outcomes. That is descriptive statistics,
it was useful, and calling it ML was an oversell that any reader would catch in
thirty seconds. It is not in this repository, and if it returns it will be called
what it is.

**No performance claims, and no account data.** Every number this repository can
print comes from a generated series. Nothing here reports what any real account
did, and the `.gitignore` is written to keep it that way.

**Persistence is JSON, single-writer.** `StrategyState.save()` writes to a
temporary file and replaces atomically, so a crash mid-write cannot truncate the
configuration, which it previously did. That is as far as it goes. Several
processes writing one file needs a real lock or a real database, and pretending
otherwise with JSON is how the earlier version lost its configuration.

## Before you point this at money

Perpetual futures with leverage can lose more than the margin posted. This is
research tooling written by one person and it has no track record. A backtest
result is a measurement of the past under assumptions, several of which are
listed above and several of which are certainly not. Paper trade first, for long
enough to see the strategy fail as well as succeed.

Credentials come from the environment only. Nothing here writes a credential to
disk. If you add a library that caches derived API credentials next to your code,
add that filename to `.gitignore` before you run it once, not after.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -v
python scripts/generate_sample_data.py --out data/sample_BTCUSDT_1h.csv
```

The suite is 99 tests and touches no network: the exchange client is tested
against a stubbed HTTP layer. Tests of the four methodology decisions live in
`tests/test_backtest.py` and are the ones to keep green.

## License

MIT. See [LICENSE](LICENSE).
