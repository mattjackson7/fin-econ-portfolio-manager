"""Build a client's portfolio from fund returns and their PortfolioSpec.

    profile --(spec.profile_to_spec)--> PortfolioSpec --(build_portfolio)--> weights + explanation

make_client_strategy wraps this in the standard strategy shape (past returns in, weights
out), so a client's portfolio can be run through backtest.walk_forward like any benchmark.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from models.estimation import covariance, equilibrium_returns, reference_weights
from models.optimizer import mean_variance
from models.spec import MARKET_RISK_AVERSION, PortfolioSpec

UNIVERSE_FILE = Path(__file__).resolve().parents[1] / "data" / "universe.csv"

# The reference portfolio: a standard balanced mix. A client with the market's risk
# aversion and no binding limits ends up holding exactly this.
REFERENCE_MIX = {"equity": 0.60, "bonds": 0.35, "real_assets": 0.05}

TOL = 1e-4  # how close to a limit counts as "at the limit" when explaining


class InfeasibleSpec(ValueError):
    """Raised when no portfolio can satisfy all of a client's rules at once."""


@dataclass
class PortfolioResult:
    weights: pd.Series  # by ticker, sums to 1, only funds with weight > 0
    expected_returns: pd.Series  # annual, the equilibrium estimates used
    explanation: list[str]  # spec.reasons, then which limits shaped the result


def load_universe(path=UNIVERSE_FILE) -> pd.DataFrame:
    """Reads universe.csv; returns it indexed by ticker."""
    return pd.read_csv(path, index_col="ticker")


def eligible_funds(universe: pd.DataFrame, spec: PortfolioSpec, available) -> pd.DataFrame:
    """Takes the universe, a spec and the tickers that have data; returns the universe rows
    the client may hold."""
    keep = [t for t in universe.index if t in set(available) and t not in spec.excluded_tickers]
    return universe.loc[keep]


def check_feasible(funds: pd.DataFrame, spec: PortfolioSpec) -> None:
    """Takes the eligible funds and a spec; raises InfeasibleSpec with a plain-English reason
    if no fully-invested, long-only portfolio meets every rule. Simple checks first (they give
    clearer messages), then a linear program that looks for any portfolio at all."""
    n, cap = len(funds), spec.max_weight
    if n == 0:
        raise InfeasibleSpec("No eligible funds left after exclusions.")
    if n * cap < 1 - 1e-9:
        raise InfeasibleSpec(f"Only {n} funds at max {cap:.0%} each cannot reach 100%.")
    for bucket, (lo, _) in spec.bucket_bounds.items():
        room = (funds["bucket"] == bucket).sum() * cap
        if room < lo - 1e-9:
            raise InfeasibleSpec(f"Need at least {lo:.0%} in {bucket}, but its funds can hold only {room:.0%}.")
    esg_room = min((funds["esg_label"] > 0).sum() * cap, 1.0)
    if esg_room < spec.min_esg_share - 1e-9:
        raise InfeasibleSpec(f"Need {spec.min_esg_share:.0%} in ESG funds, but they can hold only {esg_room:.0%}.")

    a_ub, b_ub = [], []
    for bucket, (lo, hi) in spec.bucket_bounds.items():
        member = (funds["bucket"] == bucket).astype(float).values
        a_ub += [member, -member]
        b_ub += [hi, -lo]
    if spec.min_esg_share > 0:
        a_ub.append(-(funds["esg_label"] > 0).astype(float).values)
        b_ub.append(-spec.min_esg_share)
    lp = linprog(np.zeros(n), A_ub=np.array(a_ub) if a_ub else None, b_ub=b_ub or None,
                 A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, cap)] * n)
    if lp.status != 0:
        raise InfeasibleSpec("The asset-class, ESG and fund-size limits conflict with each other.")


def explain_limits(w: pd.Series, funds: pd.DataFrame, spec: PortfolioSpec) -> list[str]:
    """Takes final weights, the eligible funds and the spec; returns a sentence for each
    limit the portfolio is pressed against (the limits that shaped it)."""
    notes = []
    w = w.reindex(funds.index, fill_value=0.0)
    for bucket, (lo, hi) in spec.bucket_bounds.items():
        share = w[funds.index[funds["bucket"] == bucket]].sum()
        if hi < 1 and share >= hi - TOL:
            notes.append(f"{bucket} is at its {hi:.0%} maximum; the client would otherwise hold more.")
        elif lo > 0 and share <= lo + TOL:
            notes.append(f"{bucket} is at its {lo:.0%} minimum.")
    if spec.min_esg_share > 0:
        esg = w[funds.index[funds["esg_label"] > 0]].sum()
        if esg <= spec.min_esg_share + TOL:
            notes.append(f"ESG funds are at the client's {spec.min_esg_share:.0%} minimum.")
    capped = w.index[w >= spec.max_weight - TOL].tolist()
    if capped:
        notes.append(f"{', '.join(capped)} at the {spec.max_weight:.0%} single-fund cap.")
    if not notes:
        notes.append("No limit is binding: the mix reflects the client's risk aversion alone.")
    return notes


def build_portfolio(returns: pd.DataFrame, universe: pd.DataFrame, spec: PortfolioSpec) -> PortfolioResult:
    """Takes daily fund returns (past window only), the universe and a client's spec;
    returns the client's portfolio with an explanation.

    Funds with any missing return in the window (not yet launched) are left out, so funds
    with different start dates can share one universe."""
    returns = returns.dropna(axis=1, how="any")
    funds = eligible_funds(universe, spec, returns.columns)
    check_feasible(funds, spec)

    tickers = funds.index
    hedge = None
    if spec.liability_hedge:
        missing = [t for t in spec.liability_hedge if t not in returns.columns]
        if missing:
            raise InfeasibleSpec(f"No return data for liability proxy {missing}.")
        tickers = tickers.union(pd.Index(spec.liability_hedge), sort=False)
        hedge = pd.Series(spec.liability_hedge)

    cov = covariance(returns[tickers])
    w_ref = reference_weights(funds, REFERENCE_MIX)
    mu = equilibrium_returns(cov, w_ref, MARKET_RISK_AVERSION)

    # The liability proxy stays in the covariance even if the client may not hold it, so the
    # hedge can still be measured (see _project_hedge).
    weights = mean_variance(
        mu[funds.index], cov.loc[funds.index, funds.index], spec.risk_aversion,
        max_weight=spec.max_weight,
        esg_scores=(funds["esg_label"] > 0).astype(float).values,
        min_esg=spec.min_esg_share,
        buckets=funds["bucket"], bucket_bounds=spec.bucket_bounds,
        hedge=_project_hedge(hedge, cov, funds.index),
    )
    weights = weights[weights > TOL]
    weights /= weights.sum()
    return PortfolioResult(weights, mu[funds.index], list(spec.reasons) + explain_limits(weights, funds, spec))


def _project_hedge(hedge, cov, investable):
    """If the liability proxy is investable, the hedge is just that weight. If not, replace it
    with the investable mix that tracks it best (regression on the covariance), so a client
    who excludes TLT still gets a sensible liability hedge."""
    if hedge is None:
        return None
    outside = [t for t in hedge.index if t not in investable]
    if not outside:
        return hedge
    inside = hedge.reindex(investable).fillna(0.0)
    s_ii = cov.loc[investable, investable].values
    for t in outside:
        beta = np.linalg.solve(s_ii, cov.loc[investable, t].values)
        inside += hedge[t] * pd.Series(beta, index=investable)
    return inside


def make_client_strategy(universe: pd.DataFrame, spec: PortfolioSpec):
    """Takes the universe and a spec; returns strategy(window) -> weights for walk_forward."""

    def strategy(window: pd.DataFrame) -> pd.Series:
        return build_portfolio(window, universe, spec).weights

    return strategy
