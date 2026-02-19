import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import download_adj_close, compute_log_returns
from src.var_models import portfolio_returns
from src.backtesting import rolling_historical_var
from src.var_models import ewma_var

# Load data
prices = download_adj_close(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")
rets = compute_log_returns(prices)

# Portfolio
w = np.array([0.5, 0.5])
p = portfolio_returns(rets, w)

# Rolling VaR
rolling_var = rolling_historical_var(p, window=60, alpha=0.01)


ewma_var_series = ewma_var(p, alpha=0.01, lambda_=0.94)

# Violations
violations = p < -rolling_var

# Plot
plt.figure(figsize=(12, 6))
plt.plot(p.index, p, label="Portfolio Return", alpha=0.5)
plt.plot(rolling_var.index, -rolling_var, label="Rolling Historical VaR (99%)", color="red")

plt.scatter(p.index[violations], p[violations], color="black", label="Violations")

plt.title("Rolling Historical VaR Backtest")
plt.legend()
plt.grid(True, alpha=0.3)


plt.plot(ewma_var_series.index, -ewma_var_series,
         label="EWMA VaR (99%)", color="green")

plt.tight_layout()
plt.savefig("rolling_var_backtest.png", dpi=300)
plt.show()


# Summary statistics
total_obs = len(p.dropna())
valid_var = rolling_var.dropna()

aligned_returns = p.loc[valid_var.index]
violations = aligned_returns < -valid_var

n_violations = violations.sum()
expected = len(valid_var) * 0.01
violation_rate = n_violations / len(valid_var)

print("Total Observations:", len(valid_var))
print("Violations:", int(n_violations))
print("Expected Violations:", round(expected, 2))
print("Violation Rate:", round(violation_rate, 4))


# --- EWMA Backtest ---
ewma_valid = ewma_var_series.dropna()
aligned_returns_ewma = p.loc[ewma_valid.index]
ewma_violations = aligned_returns_ewma < -ewma_valid

n_violations_ewma = ewma_violations.sum()
expected_ewma = len(ewma_valid) * 0.01
violation_rate_ewma = n_violations_ewma / len(ewma_valid)

print("\nEWMA VaR Backtest:")
print("Total Observations:", len(ewma_valid))
print("Violations:", int(n_violations_ewma))
print("Expected Violations:", round(expected_ewma, 2))
print("Violation Rate:", round(violation_rate_ewma, 4))

# Kupiec for EWMA
from scipy.stats import chi2
import numpy as np

n = len(ewma_valid)
x = int(n_violations_ewma)
alpha = 0.01
p_hat = x / n

if x == 0 or x == n:
    print("Kupiec test undefined.")
else:
    LR = -2 * (
        (n - x) * np.log((1 - alpha) / (1 - p_hat)) +
        x * np.log(alpha / p_hat)
    )
    p_value = 1 - chi2.cdf(LR, df=1)

    print("Kupiec LR statistic:", round(LR, 4))
    print("Kupiec p-value:", round(p_value, 6))
    print("Reject model at 5% level:", p_value < 0.05)





from scipy.stats import chi2
import numpy as np

# Kupiec POF test for rolling VaR
n = len(valid_var)
x = int(n_violations)
alpha = 0.01
p_hat = x / n

if x == 0 or x == n:
    print("Kupiec test undefined for edge case.")
else:
    LR = -2 * (
        (n - x) * np.log((1 - alpha) / (1 - p_hat)) +
        x * np.log(alpha / p_hat)
    )
    p_value = 1 - chi2.cdf(LR, df=1)

    print("Kupiec LR statistic:", round(LR, 4))
    print("Kupiec p-value:", round(p_value, 6))
    print("Reject model at 5% level:", p_value < 0.05)
