"""Walk-forward backtest with no look-ahead."""
import numpy as np
import pandas as pd


def walk_forward(
    returns: pd.DataFrame,
    strategy,
    lookback: int = 504,
    rebalance_every: int = 21,
    cost_bps: float = 5.0,
):
    """Backtest a strategy on daily returns.

    On each rebalance day t, strategy(window) sees only the previous `lookback` rows
    (strictly before day t) and returns target weights as a pd.Series. Those weights are
    held, drifting with market returns, until the next rebalance. Each rebalance costs
    cost_bps per unit of absolute weight change (the first one buys the whole portfolio).

    Returns (portfolio_returns, target_weights, turnover):
        portfolio_returns: daily returns net of costs, indexed from day `lookback` on
        target_weights: DataFrame of targets, one row per rebalance date
        turnover: Series of absolute weight change at each rebalance date
    """
    n_assets = returns.shape[1]
    dates = returns.index
    current = np.zeros(n_assets)
    port, targets, turnover = [], {}, {}

    for t in range(lookback, len(dates)):
        cost = 0.0
        if (t - lookback) % rebalance_every == 0:
            window = returns.iloc[t - lookback : t]
            target = strategy(window).reindex(returns.columns).fillna(0.0).values
            change = np.abs(target - current).sum()
            cost = change * cost_bps / 1e4
            targets[dates[t]] = target
            turnover[dates[t]] = change
            current = target
        day_returns = returns.iloc[t].values
        gross = current @ day_returns
        port.append(gross - cost)
        current = current * (1 + day_returns) / (1 + gross)  # weights drift with the market

    port = pd.Series(port, index=dates[lookback:], name="portfolio")
    targets = pd.DataFrame(targets, index=returns.columns).T
    turnover = pd.Series(turnover, name="turnover")
    return port, targets, turnover
