import numpy as np
import pandas as pd
import pytest

from models import goals as g
from models.backtest import walk_forward
from models.clients import EXAMPLE_INDIVIDUALS, EXAMPLE_INSTITUTIONS, IndividualProfile, InstitutionalProfile
from models.estimation import covariance, equilibrium_returns, reference_weights
from models.optimizer import mean_variance
from models.portfolio import (
    REFERENCE_MIX, InfeasibleSpec, build_portfolio, load_universe, make_client_strategy,
)
from models.spec import MARKET_RISK_AVERSION, PortfolioSpec, equity_cap, profile_to_spec

UNIVERSE = load_universe()


def fake_universe_returns(n=800, seed=1):
    """Correlated fake daily returns for every fund in universe.csv: an equity factor and a
    rate factor, with bonds loading on rates and equities on equity."""
    rng = np.random.default_rng(seed)
    eq, rates = rng.normal(0, 0.010, n), rng.normal(0, 0.004, n)
    cols = {}
    for t, row in UNIVERSE.iterrows():
        b_eq, b_rate = {"equity": (1.0, 0.0), "bonds": (0.05, 1.0), "real_assets": (0.5, 0.5)}[row["bucket"]]
        if t == "TLT":
            b_rate = 3.0
        cols[t] = b_eq * eq + b_rate * rates + rng.normal(0, 0.004, n)
    return pd.DataFrame(cols, index=pd.bdate_range("2018-01-01", periods=n))


R = fake_universe_returns()
ALL_PROFILES = EXAMPLE_INDIVIDUALS + EXAMPLE_INSTITUTIONS


# ---- goals ----

def test_monthly_goal_has_right_number_of_payments():
    t, a = g.Goal("Retirement", 4000, start=60, end=90, payments_per_year=12).cash_flows(25)
    assert len(t) == 360 and t[0] == 35 and a.sum() == 4000 * 360


def test_one_off_goal_horizon_and_required_return():
    house = [g.Goal("House", 200_000, start=35)]
    assert g.horizon(house, 25) == pytest.approx(10)
    # 100k must double in 10 years
    assert g.required_return(100_000, 0, 0, house, 25) == pytest.approx(2 ** 0.1 - 1, abs=1e-6)


def test_unreachable_goals_have_no_required_return():
    assert g.required_return(1_000, 0, 0, [g.Goal("Yacht", 1e9, start=26)], 25) is None


# ---- spec rules ----

def test_longer_horizon_never_lowers_equity_cap():
    caps = [equity_cap(h) for h in range(0, 50)]
    assert all(a <= b for a, b in zip(caps, caps[1:]))


def test_underfunded_client_gets_lower_tolerance():
    goals = [g.Goal("Retirement", 5_000, start=65, end=90, payments_per_year=12)]
    rich = IndividualProfile("Rich", age=50, wealth=5_000_000, risk_tolerance=9, goals=goals)
    poor = IndividualProfile("Poor", age=50, wealth=200_000, risk_tolerance=9, goals=goals)
    assert profile_to_spec(poor).risk_aversion > profile_to_spec(rich).risk_aversion


def test_endowment_holds_liquidity_buffer():
    endow = EXAMPLE_INSTITUTIONS[1]
    spec = profile_to_spec(endow)
    w = build_portfolio(R, UNIVERSE, spec).weights
    bonds = w[[t for t in w.index if UNIVERSE.loc[t, "bucket"] == "bonds"]].sum()
    assert bonds >= 3 * endow.spending_rate - 1e-6


# ---- portfolio construction ----

@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.name)
def test_every_example_client_respects_all_rules(profile):
    spec = profile_to_spec(profile)
    res = build_portfolio(R, UNIVERSE, spec)
    w = res.weights
    assert abs(w.sum() - 1) < 1e-8 and (w >= 0).all()
    assert w.max() <= spec.max_weight + 1e-6
    assert not set(w.index) & spec.excluded_tickers
    buckets = UNIVERSE.loc[w.index, "bucket"]
    for b, (lo, hi) in spec.bucket_bounds.items():
        assert lo - 1e-6 <= w[buckets == b].sum() <= hi + 1e-6
    assert w[UNIVERSE.loc[w.index, "esg_label"] > 0].sum() >= spec.min_esg_share - 1e-6
    assert res.explanation  # always explains itself


def test_market_risk_aversion_with_no_limits_gives_reference_portfolio():
    """The anchor of the whole design: equilibrium returns are chosen so that an average
    client with no binding limits holds the reference portfolio."""
    cov = covariance(R)
    w_ref = reference_weights(UNIVERSE, REFERENCE_MIX)
    mu = equilibrium_returns(cov, w_ref, MARKET_RISK_AVERSION)
    w = mean_variance(mu, cov, MARKET_RISK_AVERSION)
    assert np.allclose(w, w_ref.reindex(w.index), atol=1e-3)


def test_more_risk_aversion_means_less_volatility():
    cov = covariance(R)
    vols = []
    for lam in (0.5, 2.5, 10):
        spec = PortfolioSpec(risk_aversion=lam, bucket_bounds={}, max_weight=1.0)
        w = build_portfolio(R, UNIVERSE, spec).weights
        vols.append(float(w @ cov.loc[w.index, w.index] @ w))
    assert vols[0] > vols[1] > vols[2]


def test_pension_hedge_increases_liability_proxy():
    pension = EXAMPLE_INSTITUTIONS[0]
    with_hedge = profile_to_spec(pension)
    without = PortfolioSpec(with_hedge.risk_aversion, with_hedge.bucket_bounds, with_hedge.max_weight)
    tlt = lambda s: build_portfolio(R, UNIVERSE, s).weights.get("TLT", 0.0)
    assert tlt(with_hedge) > tlt(without) + 0.01


def test_pension_that_excludes_proxy_still_hedges_with_other_bonds():
    p = InstitutionalProfile("No TLT", kind="pension", assets=5e8, risk_tolerance=5,
                             liabilities=EXAMPLE_INSTITUTIONS[0].liabilities,
                             excluded_tickers=frozenset({"TLT"}))
    w = build_portfolio(R, UNIVERSE, profile_to_spec(p)).weights
    assert "TLT" not in w.index


def test_impossible_spec_raises_clear_error():
    # 4 bond funds x 20% cap = 80%: enough for an 80% bond floor, not once LQD is excluded
    spec = PortfolioSpec(risk_aversion=3, bucket_bounds={"bonds": (0.8, 1.0)}, max_weight=0.2)
    build_portfolio(R, UNIVERSE, spec)
    spec.excluded_tickers = frozenset({"LQD"})
    with pytest.raises(InfeasibleSpec, match="bonds"):
        build_portfolio(R, UNIVERSE, spec)


def test_fund_without_full_history_is_skipped():
    r = R.copy()
    r.iloc[:100, r.columns.get_loc("ESGU")] = np.nan  # launched mid-window
    spec = profile_to_spec(EXAMPLE_INDIVIDUALS[0])
    assert "ESGU" not in build_portfolio(r, UNIVERSE, spec).weights.index


def test_client_strategy_runs_in_backtest():
    strat = make_client_strategy(UNIVERSE, profile_to_spec(EXAMPLE_INDIVIDUALS[1]))
    port, targets, _ = walk_forward(R, strat, lookback=300, rebalance_every=63)
    assert len(port) == len(R) - 300 and np.isfinite(port).all()
    assert np.allclose(targets.sum(axis=1), 1)
