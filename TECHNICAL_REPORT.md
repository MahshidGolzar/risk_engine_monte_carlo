# Quantitative Risk Engine – Technical Report

## 1. Objective

The objective of this project is to design and implement a modular quantitative risk engine capable of estimating portfolio risk using multiple methodologies and evaluating model performance through statistical backtesting.

The project includes:

- Historical VaR
- Parametric (Gaussian) VaR
- Monte Carlo VaR (Gaussian)
- Monte Carlo VaR (Student-t)
- Historical CVaR (Expected Shortfall)
- Rolling VaR estimation
- EWMA volatility-based VaR
- Kupiec backtesting
- Distribution visualization

---

## 2. Data Description

Assets:
- AAPL
- MSFT

Frequency:
- Daily

Sample Period:
- 2023-01-01 to 2024-01-01
- ~249 trading observations

Data Type:
- Adjusted Close prices (split and dividend adjusted)

Returns:
- Daily log returns

r_t = ln(P_t / P_{t-1})

---

## 3. Portfolio Construction

Portfolio weights:
- 50% AAPL
- 50% MSFT

Portfolio returns computed as:

R_p,t = w1 * r1,t + w2 * r2,t

---

## 4. Risk Measures Implemented

### 4.1 Historical VaR (99%)

Definition:
Empirical 1% quantile of portfolio return distribution.

Result:
≈ 2.35%

Interpretation:
Based on historical data, a 1-day loss exceeding 2.35% occurs approximately 1% of the time.

---

### 4.2 Parametric (Gaussian) VaR (99%)

Assumption:
Returns are normally distributed.

Result:
≈ 2.72%

Interpretation:
Gaussian assumption produces slightly higher tail estimate than historical sample.

---

### 4.3 Monte Carlo VaR (Gaussian)

Method:
Simulated 20,000 returns from Normal(μ, σ).

Result:
≈ 2.77%

Interpretation:
Consistent with parametric VaR since both rely on Gaussian assumptions.

---

### 4.4 Student-t Monte Carlo VaR (df = 5)

Method:
Simulated 20,000 returns from Student-t distribution with df=5.

Result:
≈ 3.05%

Interpretation:
Fat-tail assumption materially increases tail risk estimate.
Demonstrates sensitivity of VaR to distributional assumptions.

---

### 4.5 Historical CVaR (99%)

Definition:
Average loss beyond the 1% VaR threshold.

Result:
≈ 2.66%

Interpretation:
Captures severity of extreme losses beyond VaR.

CVaR > VaR as expected.

---

## 5. Rolling Risk Analysis

Rolling 60-day Historical VaR was computed.

Results:
- Observations: 189
- Violations: 6
- Expected: 1.89
- Violation Rate: 3.17%

Kupiec Test:
- p-value ≈ 0.0166
- Model rejected at 5% level

Conclusion:
Rolling historical VaR underestimates tail risk in this sample.

---

## 6. EWMA Volatility VaR

EWMA parameter:
λ = 0.94

Results:
- Observations: 249
- Violations: 1
- Expected: 2.49
- Violation Rate: 0.4%

Kupiec Test:
- p-value ≈ 0.28
- Model not rejected

Conclusion:
EWMA VaR adapts to volatility clustering and performs better statistically.

---

## 7. Student-t Distribution Fitting

Student-t distribution was fitted via Maximum Likelihood Estimation.

Estimated parameters:
df ≈ 1.3e8
loc ≈ 0.0018
scale ≈ 0.0124

Interpretation:
Estimated degrees of freedom extremely high → distribution statistically close to Gaussian over this sample period.

Implication:
This sample period does not contain sufficiently extreme events to statistically justify heavy-tail modeling.

---

## 8. Model Risk Insights

Key findings:

1. Static VaR passes backtesting.
2. Rolling historical VaR fails statistical validation.
3. EWMA improves calibration.
4. Tail assumptions materially affect VaR magnitude.
5. Sample window strongly influences fitted tail behavior.

---

## 9. Limitations of Current Analysis

- Only ~1 year of data.
- No major crisis events included.
- Limited cross-asset diversification.
- Short sample insufficient for robust tail estimation.
- Student-t fit highly sensitive to sample period.

---

## 10. Why a Broader Sample Is Needed

Heavy tails are typically observed over:

- Crisis periods (2008, 2020)
- Multi-regime environments
- Long-term datasets

Short calm samples bias tail estimation toward Gaussian behavior.

Extending the dataset to 2007–2024 would:

- Capture crisis regimes
- Provide more reliable tail parameter estimation
- Improve robustness of backtesting
- Allow regime comparison

---

## 11. Next Phase

Extend dataset to long-horizon (e.g., 2007–2024) and:

- Re-estimate Student-t parameters
- Re-evaluate VaR comparisons
- Compare rolling and EWMA performance
- Evaluate structural breaks
