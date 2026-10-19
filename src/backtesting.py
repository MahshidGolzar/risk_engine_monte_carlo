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
        "reject": bool(p_value < 0.05)
    }

def violation_series(port_ret: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Boolean violation indicator aligned with var_series's index.
    A violation occurs when the realised return is worse than -VaR.
    """
    aligned_returns = port_ret.loc[var_series.dropna().index]
    aligned_var = var_series.dropna()
    return aligned_returns < -aligned_var


def christoffersen_independence_test(violations: pd.Series) -> dict:
    """
    Christoffersen (1998) test of independence of VaR violations.

    H0: violations are independent over time (no clustering), i.e. the
    probability of a violation tomorrow does not depend on whether there
    was a violation today.

    Tests the null that the transition probabilities pi_01 and pi_11 are
    equal, against the Markov-chain alternative that they differ.
    """
    v = violations.astype(int).to_numpy()

    if len(v) < 2:
        return {"n00": 0, "n01": 0, "n10": 0, "n11": 0, "p_value": None, "reject": None}

    prev, curr = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n0 = n00 + n01
    n1 = n10 + n11

    # Degenerate: no violations, or every observation is a violation -
    # the independence test is not informative.
    if n01 + n11 == 0 or n0 == 0 or n1 == 0:
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "p_value": None, "reject": None,
        }

    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _log_term(count: int, prob: float) -> float:
        # 0 * log(0) is conventionally 0 in these likelihood terms.
        return count * np.log(prob) if count > 0 and prob > 0 else 0.0

    log_l_restricted = _log_term(n00, 1 - pi) + _log_term(n01, pi) \
        + _log_term(n10, 1 - pi) + _log_term(n11, pi)
    log_l_unrestricted = _log_term(n00, 1 - pi01) + _log_term(n01, pi01) \
        + _log_term(n10, 1 - pi11) + _log_term(n11, pi11)

    LR_ind = -2 * (log_l_restricted - log_l_unrestricted)
    p_value = 1 - chi2.cdf(LR_ind, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "LR_ind": LR_ind,
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
    }


def christoffersen_conditional_coverage_test(
    port_ret: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """
    Christoffersen (1998) joint test of conditional coverage.

    Combines the Kupiec unconditional-coverage LR statistic (correct average
    violation rate) with the Christoffersen independence LR statistic (no
    clustering of violations) into a single joint test:

        LR_cc = LR_uc + LR_ind ~ chi2(2)

    A model can pass Kupiec (right number of violations overall) while
    failing this test, if violations cluster in time instead of being
    spread out - exactly the failure mode conditional coverage is designed
    to catch.
    """
    violations = violation_series(port_ret, var_series)
    n = len(violations)
    x = int(violations.sum())

    # kupiec_test takes a single scalar VaR value, but here the VaR threshold
    # varies over time (rolling/EWMA), so the unconditional-coverage LR is
    # computed directly against the violation series instead.
    p_hat = x / n if n else 0.0
    if x == 0 or x == n:
        LR_uc = None
    else:
        LR_uc = -2 * (
            (n - x) * np.log((1 - alpha) / (1 - p_hat)) +
            x * np.log(alpha / p_hat)
        )

    ind = christoffersen_independence_test(violations)
    LR_ind = ind.get("LR_ind")

    if LR_uc is None or LR_ind is None:
        return {
            "violations": x, "expected": n * alpha,
            "LR_uc": LR_uc, "LR_ind": LR_ind,
            "p_value": None, "reject": None,
        }

    LR_cc = LR_uc + LR_ind
    p_value = 1 - chi2.cdf(LR_cc, df=2)

    return {
        "violations": x,
        "expected": n * alpha,
        "LR_uc": LR_uc,
        "LR_ind": LR_ind,
        "LR_cc": LR_cc,
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
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
