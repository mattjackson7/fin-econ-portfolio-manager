import numpy as np
import pandas as pd

from models.backtest import walk_forward
from models.benchmarks import equal_weight, markowitz, minimum_variance, sixty_forty
from models.clients import EXAMPLE_CLIENTS
from models.estimation import covariance, expected_returns
from models.metrics import performance_summary
from models.optimizer import mean_variance, min_variance

COLUMNS = ["SPY", "AGG", "GLD", "ESGU"]


def fake_returns(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal([5e-4, 2e-4, 3e-4, 5e-4], [0.010, 0.003, 0.008, 0.010], size=(n, 4))
    return pd.DataFrame(data, index=pd.bdate_range("2015-01-01", periods=n), columns=COLUMNS)


def test_weights_sum_to_one_and_are_long_only():
    r = fake_returns()
    for w in (
        mean_variance(expected_returns(r), covariance(r), risk_aversion=5),
        min_variance(covariance(r)),
    ):
        assert abs(w.sum() - 1) < 1e-8
        assert (w >= -1e-12).all()


def test_min_variance_has_lower_variance_than_equal_weight():
    r = fake_returns()
    cov = covariance(r)
    w_min = min_variance(cov)
    w_eq = equal_weight(r)
    assert w_min @ cov @ w_min <= w_eq @ cov @ w_eq


def test_max_weight_cap():
    r = fake_returns()
    w = mean_variance(expected_returns(r), covariance(r), risk_aversion=1, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-8


def test_esg_constraint():
    r = fake_returns()
    esg = [0, 0, 0, 1]
    w = min_variance(covariance(r), esg_scores=esg, min_esg=0.5)
    assert w["ESGU"] >= 0.5 - 1e-6


def test_example_clients_have_sensible_risk_aversion():
    values = [c.risk_aversion() for c in EXAMPLE_CLIENTS]
    assert values[0] < values[2] < values[1]  # growth < ESG < near-retiree


def test_backtest_uses_only_past_data():
    r = fake_returns()
    seen = []

    def strategy(window):
        seen.append(window.index[-1])
        return equal_weight(window)

    lookback = 300
    port, targets, turnover = walk_forward(r, strategy, lookback=lookback, rebalance_every=21)
    assert targets.index[0] == r.index[lookback]
    assert seen[0] == r.index[lookback - 1]  # last row the strategy saw is before the first trade
    assert all(s < d for s, d in zip(seen, targets.index))
    assert len(port) == len(r) - lookback


def test_equal_weight_backtest_runs_and_summary_is_finite():
    r = fake_returns()
    for strat in (equal_weight, sixty_forty, minimum_variance, markowitz):
        port, _, _ = walk_forward(r, strat, lookback=300, rebalance_every=21)
        stats = performance_summary(port)
        assert np.isfinite(stats.values).all()
