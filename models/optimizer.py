"""Portfolio optimizers. Weights are long-only and sum to 1."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve(objective, n, max_weight, esg_scores, min_esg):
    """Minimize objective(w) subject to sum(w) = 1, 0 <= w <= max_weight, and
    (optionally) esg_scores @ w >= min_esg. Needs max_weight * n >= 1."""
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if esg_scores is not None and min_esg > 0:
        esg = np.asarray(esg_scores, dtype=float)
        constraints.append({"type": "ineq", "fun": lambda w: esg @ w - min_esg})
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Optimizer failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float,
    max_weight: float = 1.0,
    esg_scores=None,
    min_esg: float = 0.0,
) -> pd.Series:
    """Maximize mu'w - (risk_aversion / 2) w'Sigma w. mu and cov must be annualized."""
    m, s = mu.values, cov.values
    weights = _solve(
        lambda w: -(m @ w) + 0.5 * risk_aversion * (w @ s @ w),
        len(m),
        max_weight,
        esg_scores,
        min_esg,
    )
    return pd.Series(weights, index=mu.index)


def min_variance(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    esg_scores=None,
    min_esg: float = 0.0,
) -> pd.Series:
    """Minimize portfolio variance, ignoring expected returns."""
    s = cov.values
    weights = _solve(lambda w: w @ s @ w, len(s), max_weight, esg_scores, min_esg)
    return pd.Series(weights, index=cov.columns)
