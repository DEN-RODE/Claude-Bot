"""claudebot: a strategy research and execution framework for crypto perpetuals.

The library is organised around one idea: the code that backtests a strategy is
the code that trades it. See :mod:`claudebot.backtest` for the measurement
methodology and :mod:`claudebot.engine` for the closed-bar rule that keeps live
execution consistent with it.
"""

__version__ = "0.1.0"

from . import backtest, engine, indicators, regime, strategies

__all__ = ["backtest", "engine", "indicators", "regime", "strategies", "__version__"]
