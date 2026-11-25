# Models

| File | What it does | Owner |
| --- | --- | --- |
| `clients.py` | Survey outputs: individual and institutional client profiles | [Name] |
| `goals.py` | Client goals as cash flows; horizon, funded ratio, required return | [Name] |
| `spec.py` | Policy rules turning a profile into a `PortfolioSpec` (all judgment calls in one place) | [Name] |
| `portfolio.py` | Builds a client's portfolio from returns and its spec, with an explanation | [Name] |
| `estimation.py` | Returns, expected returns (historical and equilibrium), shrinkage covariance | [Name] |
| `optimizer.py` | Mean-variance (optionally vs. liabilities) and minimum-variance with weight cap, ESG minimum and asset-class bounds | [Name] |
| `benchmarks.py` | Equal weight, 60/40, Markowitz, minimum variance | [Name] |
| `backtest.py` | Walk-forward backtest with transaction costs and no look-ahead | [Name] |
| `metrics.py` | Return, volatility, Sharpe, max drawdown, CVaR | [Name] |

## How a client's portfolio is built

```
survey -> IndividualProfile / InstitutionalProfile      (clients.py, goals.py)
       -> PortfolioSpec                                  (spec.py: profile_to_spec)
       -> weights + explanation                          (portfolio.py: build_portfolio)
```

`make_client_strategy(universe, spec)` wraps this for `backtest.walk_forward`.

These are working starting points, not final choices. The final models will follow the course material. Each file exposes plain functions with documented inputs and outputs, so the dashboard can call them without knowing the internals.

## Running the tests

Run this full sequence from the project folder in the Codespace terminal:

```
cd /workspaces/YOUR-REPO-NAME
python3 -m pytest
```

Use your actual repo name in place of `YOUR-REPO-NAME`. All tests should pass before you commit changes to `models/`.
