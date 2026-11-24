"""Benchmark strategies. Each takes a window of daily returns and returns target weights,
so they plug straight into backtest.walk_forward."""
import pandas as pd

from models.estimation import covariance, expected_returns
from models.optimizer import mean_variance, min_variance


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """1/N across all funds in the window."""
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


def sixty_forty(returns: pd.DataFrame, equity: str = "SPY", bond: str = "AGG") -> pd.Series:
    """60% equity fund, 40% bond fund, nothing else."""
    w = pd.Series(0.0, index=returns.columns)
    w[equity], w[bond] = 0.6, 0.4
    return w


def markowitz(returns: pd.DataFrame, risk_aversion: float = 5.0, max_weight: float = 1.0) -> pd.Series:
    """Plain mean-variance with historical mean and sample covariance (the textbook benchmark)."""
    mu = expected_returns(returns)
    cov = covariance(returns, shrink=False)
    return mean_variance(mu, cov, risk_aversion, max_weight)


def minimum_variance(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """Minimum-variance portfolio with shrinkage covariance."""
    return min_variance(covariance(returns), max_weight)
