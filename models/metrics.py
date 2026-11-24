"""Performance statistics for a series of daily portfolio returns."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_summary(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> pd.Series:
    """Annual return, annual volatility, Sharpe ratio, max drawdown, and 95% CVaR (daily)."""
    wealth = (1 + returns).cumprod()
    ann_return = wealth.iloc[-1] ** (periods / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(periods)
    sharpe = (returns.mean() * periods - rf) / ann_vol
    max_drawdown = (wealth / wealth.cummax() - 1).min()
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    return pd.Series(
        {
            "Annual return": ann_return,
            "Annual volatility": ann_vol,
            "Sharpe ratio": sharpe,
            "Max drawdown": max_drawdown,
            "CVaR 95% (daily)": cvar_95,
        }
    )
