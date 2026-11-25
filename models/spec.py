"""Turn a client profile into a PortfolioSpec: the fixed set of numbers the optimizer needs.

Every rule is a short function of a few named constants (the POLICY section below), so each
number in a client's spec can be explained in one sentence, and spec.reasons records those
sentences for the report and the dashboard.
"""
from dataclasses import dataclass, field

import numpy as np

from models import goals as g
from models.clients import IndividualProfile, InstitutionalProfile

# ---- POLICY: the team's judgment calls. Change them here, nowhere else. ----

# Risk aversion of "the market". A client with this risk aversion holds the reference
# portfolio (see estimation.equilibrium_returns). Common textbook value.
MARKET_RISK_AVERSION = 2.5
# Survey tolerance (1-10) that corresponds to the market's risk aversion.
MIDPOINT_TOLERANCE = 5.5
# Every this many points of tolerance halves (or doubles) risk aversion.
TOLERANCE_POINTS_PER_HALVING = 2.0

# Horizon sets the most the client may hold in equity: a straight line from
# EQUITY_CAP_SHORT at horizon 0 to EQUITY_CAP_LONG at LONG_HORIZON years or more.
EQUITY_CAP_SHORT = 0.20
EQUITY_CAP_LONG = 0.90
LONG_HORIZON = 30
REAL_ASSETS_CAP = 0.15  # gold + real estate, for everyone without a policy range

# Risk capacity: funded ratio (resources / cost of essential goals) mapped linearly to a
# 1-10 tolerance between these points. The client's tolerance used is the lower of what
# they said (willingness) and this (capacity).
CAPACITY_FUNDED_LOW, CAPACITY_AT_LOW = 0.5, 3.0
CAPACITY_FUNDED_HIGH, CAPACITY_AT_HIGH = 1.2, 10.0

# Endowments keep this many years of spending in bonds so they never sell stocks in a crash
# to make payouts.
ENDOWMENT_LIQUIDITY_YEARS = 3


@dataclass
class PortfolioSpec:
    """Everything the optimizer needs to know about a client.

    risk_aversion: lambda in "maximize expected return - lambda/2 * variance".
    bucket_bounds: {asset class: (min share, max share)} using universe.csv's bucket column.
    max_weight: cap on any single fund.
    min_esg_share: minimum share in ESG-labelled funds.
    excluded_tickers: funds the client will not hold.
    liability_hedge: {ticker: weight} of a portfolio that behaves like the client's
        liabilities. If set, the optimizer minimizes the risk of assets *minus* liabilities
        (surplus risk) instead of the risk of assets alone. Used for pensions.
    reasons: one plain-English sentence per rule applied.
    stats: numbers worth showing (horizon, funded ratio, required return).
    """

    risk_aversion: float
    bucket_bounds: dict[str, tuple[float, float]]
    max_weight: float = 0.25
    min_esg_share: float = 0.0
    excluded_tickers: frozenset[str] = frozenset()
    liability_hedge: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float | None] = field(default_factory=dict)


def tolerance_to_risk_aversion(tolerance: float) -> float:
    """Takes a 1-10 tolerance; returns a risk aversion. The midpoint client gets the market's
    risk aversion, and every TOLERANCE_POINTS_PER_HALVING points halves or doubles it
    (tolerance 1 is about 12, 10 is about 0.5)."""
    steps = (MIDPOINT_TOLERANCE - tolerance) / TOLERANCE_POINTS_PER_HALVING
    return MARKET_RISK_AVERSION * 2.0 ** steps


def equity_cap(horizon_years: float) -> float:
    """Takes a horizon in years; returns the maximum equity share (straight line, capped)."""
    share = min(max(horizon_years, 0.0), LONG_HORIZON) / LONG_HORIZON
    return EQUITY_CAP_SHORT + share * (EQUITY_CAP_LONG - EQUITY_CAP_SHORT)


def capacity_tolerance(funded: float) -> float:
    """Takes a funded ratio; returns the 1-10 tolerance the client can afford."""
    return float(np.interp(funded, [CAPACITY_FUNDED_LOW, CAPACITY_FUNDED_HIGH],
                           [CAPACITY_AT_LOW, CAPACITY_AT_HIGH]))


def _default_buckets(eq_cap: float) -> dict[str, tuple[float, float]]:
    """Takes an equity cap; returns bucket bounds with bonds taking whatever is left."""
    return {"equity": (0.0, eq_cap), "bonds": (0.0, 1.0), "real_assets": (0.0, REAL_ASSETS_CAP)}


def individual_spec(p: IndividualProfile) -> PortfolioSpec:
    """Takes an IndividualProfile; returns its PortfolioSpec."""
    reasons = []
    savings_years = max((p.savings_until_age or p.age) - p.age, 0)

    if p.goals:
        hz = g.horizon(p.goals, p.age)
        reasons.append(f"Horizon {hz:.1f} years: the value-weighted average time until the goals are paid.")
    elif p.horizon_years is not None:
        hz = float(p.horizon_years)
        reasons.append(f"Horizon {hz:.0f} years, as stated (no goals given).")
    else:
        raise ValueError(f"{p.name}: give either goals or horizon_years.")

    funded = g.funded_ratio(p.wealth, p.annual_savings, savings_years, p.goals, p.age)
    capacity = capacity_tolerance(funded)
    tolerance = min(p.risk_tolerance, capacity)
    if capacity < p.risk_tolerance:
        reasons.append(
            f"Risk tolerance lowered from {p.risk_tolerance} to {tolerance:.1f}: resources cover "
            f"{funded:.0%} of essential goals, so a large loss could not be made up."
        )
    else:
        reasons.append(f"Risk tolerance {p.risk_tolerance} as stated; essential goals are {funded:.0%} funded.")

    req = g.required_return(p.wealth, p.annual_savings, savings_years, p.goals, p.age) if p.goals else None
    if p.goals and req is None:
        reasons.append("Warning: the goals cannot be met at any reasonable return. Revise goals or savings.")

    eq_cap = equity_cap(hz)
    reasons.append(f"Equity capped at {eq_cap:.0%} because of the {hz:.0f}-year horizon.")
    return PortfolioSpec(
        risk_aversion=tolerance_to_risk_aversion(tolerance),
        bucket_bounds=_default_buckets(eq_cap),
        max_weight=p.max_weight,
        min_esg_share=p.min_esg_share,
        excluded_tickers=p.excluded_tickers,
        reasons=reasons,
        stats={"horizon_years": hz, "funded_ratio": funded, "tolerance_used": tolerance,
               "required_real_return": req},
    )


def institutional_spec(p: InstitutionalProfile) -> PortfolioSpec:
    """Takes an InstitutionalProfile; returns its PortfolioSpec."""
    reasons, stats, hedge = [], {}, {}
    tolerance = p.risk_tolerance

    if p.kind == "pension":
        if not p.liabilities:
            raise ValueError(f"{p.name}: a pension needs liabilities.")
        hz = g.horizon(p.liabilities, 0)
        funded = p.assets / g.present_value(p.liabilities, 0)
        capacity = capacity_tolerance(funded)
        if capacity < tolerance:
            reasons.append(f"Risk tolerance lowered from {tolerance} to {capacity:.1f}: plan is {funded:.0%} funded.")
            tolerance = capacity
        hedge = {p.liability_proxy: 1.0 / funded}
        reasons.append(
            f"Risk measured against liabilities: each $1 of assets backs ${1 / funded:.2f} of promised "
            f"benefits, which move like {p.liability_proxy}, so holding {p.liability_proxy} reduces risk."
        )
        buckets = _default_buckets(equity_cap(hz))
        reasons.append(f"Equity capped at {equity_cap(hz):.0%}: benefits are paid {hz:.1f} years out on average.")
        stats.update(horizon_years=hz, funded_ratio=funded)
    elif p.kind == "endowment":
        hz = float("inf")
        buckets = _default_buckets(equity_cap(hz))
        bond_floor = min(ENDOWMENT_LIQUIDITY_YEARS * p.spending_rate, 1.0)
        buckets["bonds"] = (bond_floor, 1.0)
        reasons.append(f"Perpetual horizon, so equity can go up to {equity_cap(hz):.0%}.")
        reasons.append(
            f"At least {bond_floor:.0%} in bonds: {ENDOWMENT_LIQUIDITY_YEARS} years of "
            f"{p.spending_rate:.1%} spending, so payouts never force selling stocks in a crash."
        )
        stats.update(horizon_years=hz, required_real_return=p.spending_rate)
    else:
        raise ValueError(f"Unknown institution kind: {p.kind!r}")

    for bucket, bounds in p.policy_ranges.items():
        buckets[bucket] = bounds
        reasons.append(f"{bucket} kept between {bounds[0]:.0%} and {bounds[1]:.0%} by the investment policy.")

    stats["tolerance_used"] = tolerance
    return PortfolioSpec(
        risk_aversion=tolerance_to_risk_aversion(tolerance),
        bucket_bounds=buckets,
        max_weight=p.max_weight,
        min_esg_share=p.min_esg_share,
        excluded_tickers=p.excluded_tickers,
        liability_hedge=hedge,
        reasons=reasons,
        stats=stats,
    )


def profile_to_spec(profile) -> PortfolioSpec:
    """Takes any client profile; returns its PortfolioSpec."""
    if isinstance(profile, IndividualProfile):
        return individual_spec(profile)
    if isinstance(profile, InstitutionalProfile):
        return institutional_spec(profile)
    raise TypeError(f"Unsupported profile type: {type(profile).__name__}")
