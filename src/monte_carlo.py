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

def monte_carlo_var_student_t(
    port_ret: pd.Series,
    alpha: float = 0.01,
    n_sim: int = 10000,
    horizon: int = 1,
    df: float = 5.0,
    seed: int = 42,
) -> float:
    """
    Monte Carlo VaR using Student-t innovations (fat tails).

    We scale the t-distribution so the simulated returns have approximately the
    same standard deviation as the historical series.

    df: degrees of freedom (lower = fatter tails). Must be > 2 for finite variance.
    Returns positive loss number.
    """
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance")

    r = port_ret.dropna().values
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))

    # Student-t with df has variance df/(df-2). Choose scale so Var matches sigma^2.
    scale = sigma * np.sqrt((df - 2.0) / df)

    rng = np.random.default_rng(seed)

    # Sample t innovations then scale + shift
    t_samples = rng.standard_t(df, size=n_sim)
    simulated_1d = mu + scale * t_samples

    # Horizon scaling (approx): sum of iid daily returns
    if horizon != 1:
        simulated = np.sum(
            rng.standard_t(df, size=(n_sim, horizon)), axis=1
        ) * scale + mu * horizon
    else:
        simulated = simulated_1d

    var = -np.quantile(simulated, alpha)
    return float(var)
