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


def reference_weights(universe: pd.DataFrame, bucket_mix: dict[str, float]) -> pd.Series:
    """Takes the universe (indexed by ticker, with a `bucket` column, restricted to the funds
    we may hold) and the reference share of each bucket; returns reference weights.

    Within a bucket, funds are weighted by the optional `ref_weight` column (e.g. market cap),
    or equally if it is missing. Buckets with no funds are dropped and the rest rescaled."""
    if "ref_weight" in universe.columns:
        raw = universe["ref_weight"].fillna(0.0).astype(float)
    else:
        raw = pd.Series(1.0, index=universe.index)
    w = pd.Series(0.0, index=universe.index)
    for bucket, share in bucket_mix.items():
        members = universe.index[universe["bucket"] == bucket]
        if len(members) and raw[members].sum() > 0:
            w[members] = share * raw[members] / raw[members].sum()
    if w.sum() == 0:
        raise ValueError("No funds in any reference bucket.")
    return w / w.sum()


def equilibrium_returns(cov: pd.DataFrame, w_ref: pd.Series, risk_aversion: float = 2.5) -> pd.Series:
    """Takes an annualized covariance, reference weights and the market's risk aversion;
    returns the expected (excess) returns that would make the reference portfolio exactly
    optimal: pi = risk_aversion * cov @ w_ref.

    This is the first step of Black-Litterman with no views. A fund's expected return is
    high when it adds a lot of risk to the reference portfolio, so the estimates are stable
    and never chase recent winners."""
    w = w_ref.reindex(cov.index).fillna(0.0)
    return risk_aversion * cov @ w
