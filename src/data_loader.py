from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download Adjusted Close prices for tickers between start and end dates.
    Returns DataFrame indexed by date with tickers as columns.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # For multiple tickers, yfinance returns columns like ('Adj Close', 'AAPL'), ...
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data[["Adj Close"]] if "Adj Close" in data.columns else data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    r_t = ln(P_t / P_{t-1})
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns
