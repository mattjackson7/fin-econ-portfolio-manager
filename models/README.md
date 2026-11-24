# Models

| File | What it does | Owner |
| --- | --- | --- |
| `clients.py` | Client profile and the mapping from risk tolerance to risk aversion | [Name] |
| `estimation.py` | Returns, expected returns, shrinkage covariance | [Name] |
| `optimizer.py` | Mean-variance and minimum-variance with weight cap and ESG minimum | [Name] |
| `benchmarks.py` | Equal weight, 60/40, Markowitz, minimum variance | [Name] |
| `backtest.py` | Walk-forward backtest with transaction costs and no look-ahead | [Name] |
| `metrics.py` | Return, volatility, Sharpe, max drawdown, CVaR | [Name] |

These are working starting points, not final choices. The final models will follow the course material. Each file exposes plain functions with documented inputs and outputs, so the dashboard can call them without knowing the internals.

## Running the tests

Run this full sequence from the project folder in the Codespace terminal:

```
cd /workspaces/YOUR-REPO-NAME
python3 -m pytest
```

Use your actual repo name in place of `YOUR-REPO-NAME`. All tests should pass before you commit changes to `models/`.
