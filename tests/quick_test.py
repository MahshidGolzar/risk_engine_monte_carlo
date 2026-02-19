import numpy as np

from src.data_loader import download_adj_close, compute_log_returns
from src.var_models import portfolio_returns, historical_var, parametric_var
from src.cvar import historical_cvar
from src.monte_carlo import monte_carlo_var

prices = download_adj_close(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")
rets = compute_log_returns(prices)

w = np.array([0.5, 0.5])
p = portfolio_returns(rets, w)

var_hist_99 = historical_var(p, alpha=0.01)
var_param_99 = parametric_var(p, alpha=0.01)
cvar_99 = historical_cvar(p, alpha=0.01)
var_mc_99 = monte_carlo_var(p, alpha=0.01, n_sim=20000)
print("Monte Carlo VaR 99% (1-day):", var_mc_99)


print("Parametric VaR 99% (1-day):", var_param_99)
print("Historical VaR 99% (1-day):", var_hist_99)
print("Historical CVaR 99% (1-day):", cvar_99)
