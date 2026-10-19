from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    count_violations,
    kupiec_test,
    rolling_historical_var,
    violation_series,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_count_violations_counts_losses_worse_than_var():
    returns = pd.Series([0.01, -0.05, -0.02, 0.00, -0.06], index=_dates(5))
    assert count_violations(returns, var_value=0.03) == 2


def test_kupiec_fails_to_reject_well_calibrated_model():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 2500), index=_dates(2500))
    var_value = 0.01
    result = kupiec_test(returns, var_value=0.01 * 2.326, alpha=0.01)
    assert result["reject"] is False
    assert abs(result["violations"] - result["expected"]) < 15


def test_kupiec_rejects_grossly_miscalibrated_model():
    # VaR set far too tight: ~50% of days should breach it, nowhere near alpha=1%.
    returns = pd.Series(np.full(500, -0.01), index=_dates(500))
    result = kupiec_test(returns, var_value=0.001, alpha=0.01)
    assert result["reject"] is True


def test_violation_series_aligns_on_var_series_index():
    returns = pd.Series([0.01, -0.05, -0.02, 0.00, -0.06], index=_dates(5))
    var_series = pd.Series([np.nan, 0.03, 0.03, 0.03, 0.03], index=_dates(5))
    violations = violation_series(returns, var_series)
    assert list(violations.index) == list(_dates(5)[1:])
    assert violations.tolist() == [True, False, False, True]


def test_christoffersen_independence_detects_clustering():
    # Violations clumped together back-to-back: strong positive autocorrelation.
    pattern = [0] * 20 + [1, 1, 1, 1, 1] + [0] * 20
    violations = pd.Series(pattern, index=_dates(len(pattern))).astype(bool)
    result = christoffersen_independence_test(violations)
    assert result["reject"] is True


def test_christoffersen_independence_accepts_scattered_violations():
    # Violations spread evenly through the sample, no clustering.
    n = 300
    pattern = [1 if i % 20 == 0 else 0 for i in range(n)]
    violations = pd.Series(pattern, index=_dates(n)).astype(bool)
    result = christoffersen_independence_test(violations)
    assert result["reject"] is False


def test_christoffersen_independence_handles_no_violations():
    violations = pd.Series(np.zeros(50, dtype=bool), index=_dates(50))
    result = christoffersen_independence_test(violations)
    assert result["p_value"] is None
    assert result["reject"] is None


def test_conditional_coverage_catches_clustering_kupiec_alone_misses():
    # Build a return series whose VaR violations have the right overall
    # frequency (Kupiec would pass) but are clustered in one block (Kupiec
    # cannot see this, Christoffersen's independence component can).
    n = 400
    returns = pd.Series(np.full(n, 0.001), index=_dates(n))
    cluster_start = 150
    cluster_len = 4
    returns.iloc[cluster_start:cluster_start + cluster_len] = -0.05

    var_series = pd.Series(np.full(n, 0.03), index=_dates(n))

    uc = kupiec_test(returns, var_value=0.03, alpha=0.01)
    cc = christoffersen_conditional_coverage_test(returns, var_series, alpha=0.01)

    assert uc["reject"] is False
    assert cc["reject"] is True


def test_rolling_historical_var_is_nan_before_window_fills():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100), index=_dates(100))
    rolling_var = rolling_historical_var(returns, window=60, alpha=0.01)
    assert rolling_var.iloc[:60].isna().all()
    assert rolling_var.iloc[60:].notna().all()
