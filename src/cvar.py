from __future__ import annotations

import numpy as np
import pandas as pd


def historical_cvar(port_ret: pd.Series, alpha: float = 0.01) -> float:
    """
    Historical CVaR / Expected Shortfall at level alpha.
    Returns a positive loss number (average loss in the worst alpha tail).
    """
    r = port_ret.dropna().values
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(-np.mean(tail))
