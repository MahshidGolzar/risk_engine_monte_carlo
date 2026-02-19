from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_var(
    port_ret: pd.Series,
    alpha: float = 0.01,
    n_sim: int = 10000,
    horizon: int = 1,
) -> float:
    """
    Monte Carlo VaR assuming Gaussian returns (GBM approximation).
    
    port_ret: historical portfolio return series
    alpha: confidence level (0.01 = 99%)
    n_sim: number of simulations
    horizon: number of days ahead
    
    Returns positive loss number.
    """

    r = port_ret.dropna().values
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    np.random.seed(42)


    # simulate future returns
    simulated = np.random.normal(
        loc=mu * horizon,
        scale=sigma * np.sqrt(horizon),
        size=n_sim,
    )

    var = -np.quantile(simulated, alpha)
    return float(var)
