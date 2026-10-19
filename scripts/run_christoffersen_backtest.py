"""
Run Kupiec, Christoffersen independence, and joint conditional-coverage
backtests against the rolling-window historical VaR and EWMA VaR series.

Usage: PYTHONPATH=. python scripts/run_christoffersen_backtest.py
"""
from __future__ import annotations

import numpy as np

from src.backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    rolling_historical_var,
    violation_series,
)
from src.data_loader import compute_log_returns, download_adj_close
from src.var_models import ewma_var, portfolio_returns


def report(name: str, port_ret, var_series, alpha: float = 0.01) -> None:
    violations = violation_series(port_ret, var_series)
    n = len(violations)
    x = int(violations.sum())

    ind = christoffersen_independence_test(violations)
    cc = christoffersen_conditional_coverage_test(port_ret, var_series, alpha=alpha)

    print(f"\n=== {name} ===")
    print(f"Observations: {n}, Violations: {x}, Expected: {n * alpha:.2f}")
    print(f"Christoffersen independence LR: {ind.get('LR_ind')}, "
          f"p-value: {ind.get('p_value')}, reject H0 (independent): {ind.get('reject')}")
    print(f"Conditional coverage LR_cc: {cc.get('LR_cc')}, "
          f"p-value: {cc.get('p_value')}, reject H0 (correctly specified): {cc.get('reject')}")


def main() -> None:
    prices = download_adj_close(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")
    rets = compute_log_returns(prices)
    w = np.array([0.5, 0.5])
    p = portfolio_returns(rets, w)

    rolling_var = rolling_historical_var(p, window=60, alpha=0.01)
    ewma_var_series = ewma_var(p, alpha=0.01, lambda_=0.94)

    report("Rolling 60-day Historical VaR", p, rolling_var)
    report("EWMA VaR (lambda=0.94)", p, ewma_var_series)


if __name__ == "__main__":
    main()
