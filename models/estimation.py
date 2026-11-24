"""Inputs for the optimizer: returns, expected returns, and covariance."""
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Rows with any missing value are dropped, so the sample starts
    at the youngest fund. Pick the columns you need before calling (for example, leave out
    ESG funds when testing 2008)."""
    return prices.pct_change().dropna()


def expected_returns(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Historical mean return, annualized. Noisy: see the Black-Litterman alternative."""
    return returns.mean() * periods


def covariance(returns: pd.DataFrame, periods: int = TRADING_DAYS, shrink: bool = True) -> pd.DataFrame:
    """Annualized covariance matrix. shrink=True uses Ledoit-Wolf shrinkage."""
    if shrink:
        cov = LedoitWolf().fit(returns.values).covariance_
    else:
        cov = returns.cov().values
    return pd.DataFrame(cov * periods, index=returns.columns, columns=returns.columns)
