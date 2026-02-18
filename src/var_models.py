from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Convert asset return matrix (T x N) into portfolio return series (T,).
    weights must sum to 1.
    """
    w = np.asarray(weights, dtype=float)
    if returns.shape[1] != w.shape[0]:
        raise ValueError("weights length must match number of assets")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return pd.Series(returns.values @ w, index=returns.index, name="portfolio_return")


def historical_var(port_ret: pd.Series, alpha: float = 0.01) -> float:
    """
    Historical VaR at level alpha (e.g., 0.01 = 99% VaR).
    Returns a positive loss number.
    """
    q = np.quantile(port_ret.dropna().values, alpha)
    return float(-q)


def parametric_var(port_ret: pd.Series, alpha: float = 0.01) -> float:
    """
    Parametric (Gaussian) VaR.
    Assumes portfolio returns are normally distributed.
    Returns a positive loss number.
    """
    r = port_ret.dropna().values
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = float(norm.ppf(alpha))
    return float(-(mu + z * sigma))
