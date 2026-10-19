# Monte Carlo VaR & CVaR Risk Engine

A modular Python risk engine implementing multiple Value-at-Risk (VaR) methodologies and Expected Shortfall (CVaR), validated through Kupiec unconditional-coverage and Christoffersen independence / conditional-coverage backtesting.

## Why this project exists

Any VaR number is only as good as the backtest behind it. This project isn't just "compute a VaR" — it implements several VaR methodologies side by side (so their assumptions can be compared directly) and then checks each one against two different failure modes:

- **Kupiec test** — are there the *right number* of violations overall?
- **Christoffersen independence test** — are violations *clustered in time* instead of spread out? A model can pass Kupiec (correct average violation rate) while still being unsafe in practice, if its breaches arrive in bursts rather than independently.
- **Christoffersen conditional-coverage test** — the joint test combining both (`LR_cc = LR_uc + LR_ind`), which is what actually determines whether a VaR model is well specified.

## Methods implemented

- Historical VaR
- Parametric (Gaussian) VaR
- Monte Carlo VaR (Gaussian and Student-t)
- Historical CVaR / Expected Shortfall
- Rolling-window historical VaR
- EWMA volatility-based VaR
- Kupiec, Christoffersen independence, and conditional-coverage backtests

## Results (AAPL/MSFT 50/50 portfolio, 2023-01-01 to 2024-01-01, 99% 1-day VaR)

| Method | VaR estimate |
|---|---|
| Historical | 2.35% |
| Parametric (Gaussian) | 2.72% |
| Monte Carlo (Gaussian) | 2.77% |
| Monte Carlo (Student-t, df=5) | 3.05% |
| Historical CVaR | 2.66% |

Fatter tail assumptions (Student-t) push the VaR estimate up materially relative to Gaussian-based methods — expected, and a useful sanity check that the simulation responds correctly to its distributional assumption.

### Backtesting the time-varying VaR series

| VaR series | Obs. | Violations | Kupiec reject? | Christoffersen independence reject? | Conditional coverage reject? |
|---|---|---|---|---|---|
| Rolling 60-day historical VaR | 189 | 6 (expected 1.89) | Yes (p ≈ 0.017) | No (p ≈ 0.53) | Yes (p ≈ 0.047) |
| EWMA VaR (λ=0.94) | 249 | 1 (expected 2.49) | No (p ≈ 0.28) | No (p ≈ 0.93) | No (p ≈ 0.56) |

The rolling historical VaR fails both the simple Kupiec test and the joint conditional-coverage test — it under-counts tail risk on this sample. The independence component on its own does *not* reject here (the violations in this particular window happen not to cluster), so in this case conditional coverage agrees with Kupiec rather than catching an additional clustering failure — but the two tests measure genuinely different things, and the repo's test suite (`tests/test_backtesting.py`) includes a synthetic example where violations are deliberately clustered, which Kupiec alone misses and conditional coverage catches. The EWMA VaR passes all three tests, consistent with it adapting to volatility clustering better than a fixed rolling window.

Full derivation and interpretation: [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

## Repo layout

- `src/` — core risk engine: data loading, VaR/CVaR models, Monte Carlo simulation, backtesting
- `tests/` — pytest unit tests for the backtesting module (synthetic data, known-answer and edge cases)
- `scripts/` — runnable entry points: `demo_summary.py` (point-in-time VaR/CVaR summary), `run_christoffersen_backtest.py` (full backtest report)
- `notebooks/` — exploration and the rolling-VaR plotting script
- `TECHNICAL_REPORT.md` / `REPORT.md` — full write-up of methodology, data, and results

## Known limitations

- Two-asset (AAPL/MSFT) portfolio over a single, fairly calm ~1-year sample — no crisis period (2008, 2020) is included, so tail estimates here likely understate risk in a genuine stress regime.
- Monte Carlo and parametric VaR assume i.i.d. Gaussian or Student-t returns with no volatility clustering (GARCH) or cross-asset correlation modelling beyond the fixed portfolio weights.
- Rolling and EWMA VaR are univariate on the already-combined portfolio return series, not re-estimated per-asset.

## Running it

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Unit tests (no network access required)
pytest tests/ -v

# Full demo and backtest reports (downloads AAPL/MSFT prices via yfinance)
PYTHONPATH=. python scripts/demo_summary.py
PYTHONPATH=. python scripts/run_christoffersen_backtest.py
PYTHONPATH=. python notebooks/rolling_backtest.py
```
