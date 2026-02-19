import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import download_adj_close, compute_log_returns
from src.var_models import portfolio_returns, historical_var, parametric_var
from src.monte_carlo import monte_carlo_var
from src.cvar import historical_cvar
from src.monte_carlo import monte_carlo_var, monte_carlo_var_student_t


# 1. Load data
prices = download_adj_close(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")
rets = compute_log_returns(prices)

# 2. Portfolio
w = np.array([0.5, 0.5])
p = portfolio_returns(rets, w)

# 3. Risk metrics
var_hist = historical_var(p, alpha=0.01)
var_param = parametric_var(p, alpha=0.01)
var_mc = monte_carlo_var(p, alpha=0.01, n_sim=20000)
var_t = monte_carlo_var_student_t(p, alpha=0.01, n_sim=20000, df=5)
cvar_hist = historical_cvar(p, alpha=0.01)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.hist(p, bins=40, density=True, alpha=0.6, color="steelblue")

plt.axvline(-var_hist, linestyle="--", color="red",
            label=f"Historical VaR: {var_hist:.4f}")

plt.axvline(-var_param, linestyle="--", color="green",
            label=f"Parametric VaR: {var_param:.4f}")

plt.axvline(-var_mc, linestyle="--", color="purple",
            label=f"Monte Carlo VaR: {var_mc:.4f}")

plt.axvline(-cvar_hist, linestyle="--", color="black",
            label=f"Historical CVaR: {cvar_hist:.4f}")
plt.axvline(-var_t, linestyle="--", label=f"Student-t MC VaR: {var_t:.4f}")

plt.title("Portfolio Return Distribution (1-Day, 99% Risk)")
plt.xlabel("Daily Log Return")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("risk_distribution.png", dpi=300)
plt.show()
