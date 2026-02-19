from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def count_violations(port_ret: pd.Series, var_value: float) -> int:
    """
    Count number of VaR violations.
    A violation occurs when return < -VaR.
    """
    violations = port_ret < -var_value
    return int(violations.sum())


def kupiec_test(
    port_ret: pd.Series,
    var_value: float,
    alpha: float = 0.01
) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.
    
    H0: violation probability = alpha
    """

    n = len(port_ret)
    x = count_violations(port_ret, var_value)

    p_hat = x / n

    # Likelihood ratio statistic
    if x == 0 or x == n:
        return {
            "violations": x,
            "expected": n * alpha,
            "p_value": 0.0,
            "reject": True
        }

    LR = -2 * (
        (n - x) * np.log((1 - alpha) / (1 - p_hat)) +
        x * np.log(alpha / p_hat)
    )

    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "violations": x,
        "expected": n * alpha,
        "p_value": p_value,
        "reject": p_value < 0.05
    }

def rolling_historical_var(
    port_ret: pd.Series,
    window: int = 60,
    alpha: float = 0.01
) -> pd.Series:
    """
    Compute rolling historical VaR using a moving window.
    Returns a Series aligned with port_ret index.
    """

    rolling_var = []

    for i in range(len(port_ret)):
        if i < window:
            rolling_var.append(np.nan)
        else:
            window_data = port_ret.iloc[i - window:i]
            var = -np.quantile(window_data, alpha)
            rolling_var.append(var)

    return pd.Series(rolling_var, index=port_ret.index)
