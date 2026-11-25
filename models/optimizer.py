"""Portfolio optimizers. Weights are long-only and sum to 1."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _linear_constraints(n, esg_scores, min_esg, groups):
    """Builds SLSQP constraints: weights sum to 1, ESG share >= min_esg, and for each group
    (membership vector, lo, hi) lo <= group share <= hi."""
    ones = np.ones(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones}]
    if esg_scores is not None and min_esg > 0:
        esg = np.asarray(esg_scores, dtype=float)
        cons.append({"type": "ineq", "fun": lambda w: esg @ w - min_esg, "jac": lambda w: esg})
    for member, lo, hi in groups or []:
        m = np.asarray(member, dtype=float)
        if lo > 0:
            cons.append({"type": "ineq", "fun": lambda w, m=m, lo=lo: m @ w - lo, "jac": lambda w, m=m: m})
        if hi < 1:
            cons.append({"type": "ineq", "fun": lambda w, m=m, hi=hi: hi - m @ w, "jac": lambda w, m=m: -m})
    return cons


def _solve(objective, n, max_weight, esg_scores, min_esg, jac=None, groups=None):
    """Minimize objective(w) subject to sum(w) = 1, 0 <= w <= max_weight, optionally
    esg_scores @ w >= min_esg, and group bounds (see _linear_constraints).
    jac, if given, is the gradient of objective (much faster with many funds)."""
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=_linear_constraints(n, esg_scores, min_esg, groups),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise ValueError(f"Optimizer failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def _groups(index, buckets, bucket_bounds):
    """Takes fund tickers, a ticker -> bucket Series and {bucket: (lo, hi)}; returns
    (membership vector, lo, hi) triples."""
    if not bucket_bounds:
        return []
    labels = buckets.reindex(index).values
    return [((labels == b).astype(float), lo, hi) for b, (lo, hi) in bucket_bounds.items()]


def mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float,
    max_weight: float = 1.0,
    esg_scores=None,
    min_esg: float = 0.0,
    buckets: pd.Series | None = None,
    bucket_bounds: dict | None = None,
    hedge: pd.Series | None = None,
) -> pd.Series:
    """Maximize mu'w - (risk_aversion / 2) (w - h)'Sigma(w - h). mu and cov must be annualized.

    h (hedge) defaults to zero, giving plain mean-variance. For a pension, h is the portfolio
    that mimics its liabilities, so the risk term is the variance of assets minus
    liabilities (surplus risk). The same form with h = a benchmark is tracking error.
    buckets/bucket_bounds limit the share in each asset class, e.g. {"equity": (0, 0.6)}.
    """
    m, s = mu.values, cov.values
    h = np.zeros(len(m)) if hedge is None else hedge.reindex(mu.index).fillna(0.0).values

    def objective(w):
        d = w - h
        return -(m @ w) + 0.5 * risk_aversion * (d @ s @ d)

    def gradient(w):
        return -m + risk_aversion * (s @ (w - h))

    weights = _solve(objective, len(m), max_weight, esg_scores, min_esg, jac=gradient,
                     groups=_groups(mu.index, buckets, bucket_bounds))
    return pd.Series(weights, index=mu.index)


def min_variance(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    esg_scores=None,
    min_esg: float = 0.0,
) -> pd.Series:
    """Minimize portfolio variance, ignoring expected returns."""
    s = cov.values
    weights = _solve(lambda w: w @ s @ w, len(s), max_weight, esg_scores, min_esg,
                     jac=lambda w: 2 * s @ w)
    return pd.Series(weights, index=cov.columns)
