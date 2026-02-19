
# Risk Engine Report
## Monte Carlo VaR & CVaR Implementation

---

## 1. Objective

This project implements portfolio risk estimation using:

- Historical Value-at-Risk (VaR)
- Parametric (Gaussian) VaR
- Monte Carlo VaR
- Historical Conditional Value-at-Risk (CVaR / Expected Shortfall)

The goal is to build a structured, reusable risk engine suitable for quantitative risk analysis.

---

## 2. Data Description

### Assets Used
- AAPL (Apple)
- MSFT (Microsoft)

### Data Source
Yahoo Finance via `yfinance`

### Data Type
- Adjusted Close prices
- Daily frequency
- Date range: 2023-01-01 to 2024-01-01
- ~249 trading days

Adjusted Close is used to properly account for dividends and stock splits.

---

## 3. Return Calculation

Daily log returns were computed using:

r_t = ln(P_t / P_{t-1})

Log returns are used because:
- They aggregate additively across time
- They are standard in financial modeling
- They simplify Monte Carlo simulation assumptions

---

## 4. Portfolio Construction

A 50% / 50% portfolio was constructed:

- 50% AAPL
- 50% MSFT

Portfolio returns were computed as:

R_p,t = w1 * r1,t + w2 * r2,t

---

## 5. Risk Measures Implemented

### 5.1 Historical VaR

Method:
- Empirical 1% quantile of portfolio return distribution
- No distributional assumptions

Interpretation:
The 99% 1-day VaR represents the loss threshold that is exceeded only 1% of the time.

---

### 5.2 Parametric (Gaussian) VaR

Method:
- Assume portfolio returns are normally distributed
- Estimate mean (μ) and standard deviation (σ)
- Compute:

VaR = -(μ + z_α σ)

where z_α is the standard normal quantile.

Limitation:
Normality assumption may underestimate extreme tail risk.

---

### 5.3 Monte Carlo VaR

Method:
- Simulate 20,000 future portfolio returns
- Returns drawn from Normal(μ, σ)
- Compute empirical 1% quantile of simulated distribution

Advantage:
Flexible framework — can extend to fat tails, multi-factor models, correlated assets.

---

### 5.4 Historical CVaR (Expected Shortfall)

Method:
- Identify worst 1% of returns
- Compute average loss of those tail observations

Interpretation:
Average loss given that losses exceed the VaR threshold.

CVaR is a coherent risk measure and captures tail severity better than VaR.

---

## 6. Results (1-Day, 99%)

Monte Carlo VaR ≈ 2.67%  
Parametric VaR ≈ 2.72%  
Historical VaR ≈ 2.35%  
Historical CVaR ≈ 2.66%

---

## 7. Interpretation

- Parametric and Monte Carlo VaR are close because both assume normal returns.
- Historical VaR is lower, reflecting the empirical distribution over the sample period.
- CVaR exceeds Historical VaR, as expected, since it averages the worst tail losses.

The differences highlight how distributional assumptions impact risk estimates.

---

## 8. Current Limitations

- No backtesting yet
- Gaussian assumption in Monte Carlo
- No volatility clustering modeling
- No fat-tail modeling
- Short historical window

---

## 9. Future Improvements

- Backtesting using Kupiec test
- Student-t Monte Carlo simulation
- GARCH volatility modeling
- Multi-asset covariance matrix simulation
- Stress testing scenarios
- 10-day VaR estimation

---

## 10. Conclusion

This project establishes a clean, modular quantitative risk engine capable of estimating portfolio VaR and CVaR under multiple modeling assumptions.

The structure supports extension into production-grade risk analytics.
