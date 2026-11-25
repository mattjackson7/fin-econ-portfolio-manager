"""Client goals as cash flows, and the numbers we derive from them.

All amounts are in today's dollars, so every rate here is a real (after-inflation) rate.
Times are in years from now.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

REAL_DISCOUNT_RATE = 0.03  # used to value goals today; a policy choice, see models/spec.py


@dataclass
class Goal:
    """One thing the money is for.

    amount: the payment size, in today's dollars. For a one-off goal this is the whole amount;
        for a recurring goal it is each payment (e.g. 5000 with payments_per_year=12 is
        $5,000 a month).
    start: age at the first payment (for institutions, years from now, with current age 0).
    end: age at which payments stop (exclusive), or None for a one-off goal.
    essential: True for must-have goals (retirement income), False for nice-to-have
        (a bequest, a bigger house). Only essential goals set the client's risk capacity.
    """

    name: str
    amount: float
    start: float
    end: float | None = None
    payments_per_year: int = 1
    essential: bool = True

    def cash_flows(self, current_age: float) -> tuple[np.ndarray, np.ndarray]:
        """Takes the client's current age; returns (times in years from now, amounts).
        Payments that would fall in the past are dropped."""
        if self.end is None:
            times = np.array([self.start - current_age], dtype=float)
        else:
            n = int(round((self.end - self.start) * self.payments_per_year))
            times = self.start - current_age + np.arange(n) / self.payments_per_year
        amounts = np.full(len(times), float(self.amount))
        keep = times >= 0
        return times[keep], amounts[keep]


def _all_flows(goals, current_age, essential_only=False):
    """Takes goals and an age; returns all their cash flows stacked as (times, amounts)."""
    chosen = [g for g in goals if g.essential or not essential_only]
    if not chosen:
        return np.array([]), np.array([])
    parts = [g.cash_flows(current_age) for g in chosen]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def present_value(goals, current_age, rate=REAL_DISCOUNT_RATE, essential_only=False) -> float:
    """Takes goals, current age and an annual discount rate; returns what the goals cost today."""
    t, a = _all_flows(goals, current_age, essential_only)
    return float((a / (1 + rate) ** t).sum())


def savings_value(annual_savings, years, rate=REAL_DISCOUNT_RATE) -> float:
    """Takes yearly savings, number of years and a rate; returns the value today of
    saving that amount at the end of each year."""
    t = np.arange(1, int(years) + 1)
    return float((annual_savings / (1 + rate) ** t).sum())


def funded_ratio(wealth, annual_savings, savings_years, goals, current_age,
                 rate=REAL_DISCOUNT_RATE, essential_only=True) -> float:
    """Takes the client's resources and goals; returns (wealth + value of future savings)
    divided by the value of the goals. Above 1 means the goals are covered at this rate;
    returns inf if there are no goals."""
    cost = present_value(goals, current_age, rate, essential_only)
    if cost == 0:
        return float("inf")
    return (wealth + savings_value(annual_savings, savings_years, rate)) / cost


def horizon(goals, current_age, rate=REAL_DISCOUNT_RATE) -> float:
    """Takes goals and an age; returns the investment horizon in years: the average time
    until the money is needed, weighted by each payment's value today (Macaulay duration).
    A house in 5 years and retirement income from 35 to 65 years out gives something in
    between, closer to whichever goal costs more."""
    t, a = _all_flows(goals, current_age)
    if len(t) == 0:
        raise ValueError("No future goals to compute a horizon from.")
    pv = a / (1 + rate) ** t
    return float((t * pv).sum() / pv.sum())


def required_return(wealth, annual_savings, savings_years, goals, current_age,
                    essential_only=False, low=-0.05, high=0.30) -> float | None:
    """Takes the client's resources and goals; returns the constant real annual return
    that makes resources exactly cover the goals.

    Returns `low` if goals are covered even at that return, and None if not even `high`
    is enough (the goals are out of reach and should be revised)."""

    def gap(r):
        return (wealth + savings_value(annual_savings, savings_years, r)
                - present_value(goals, current_age, r, essential_only))

    if gap(low) >= 0:
        return low
    if gap(high) < 0:
        return None
    return brentq(gap, low, high)
