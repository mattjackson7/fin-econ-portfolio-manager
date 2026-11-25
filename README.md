# AI Portfolio Manager

An automated portfolio manager that turns a client's profile (wealth, risk tolerance, horizon, ESG preference, constraints) into an allocation over a universe of ETFs, rebalances after market shocks, and is compared with benchmark portfolios. Built for [course name] at Western University. Here "AI" means an automated, rule-driven program that makes and explains investment decisions.

## Team

| Name | GitHub | Owns |
| --- | --- | --- |
| [Name] | [username] | [e.g. research and survey] |
| [Name] | [username] | [e.g. client profiles and objective] |
| [Name] | [username] | [e.g. inputs and optimizer] |
| [Name] | [username] | [e.g. backtesting and stress tests] |
| Mason Walton | H4xWow | Dashboard |

## Project layout

- `app/` Streamlit dashboard. `Home.py` is the entry page. To add pages, create `app/pages/` and put one file per page in it.
- `models/` Clients, estimation, optimizer, benchmarks, backtest, metrics. See `models/README.md`.
- `data/` Fund universe (`universe.csv`), the download script, and the shared price file.
- `tests/` Automated checks for the model code.
- `report/` Figures and text for the written report.
- `.devcontainer/` Settings that set up the Codespace automatically.

## Getting started (Codespaces)

1. Accept the repo invitation you received by email.
2. On the repo page click **Code**, then the **Codespaces** tab, then **Create codespace on main**.
3. Wait for the editor to open. The dashboard preview should appear. If not, open the **Ports** tab and click the globe icon next to port 8501.
4. Edit your own files. Saving a file refreshes the dashboard.

## Saving your work

1. Click the **Source Control** icon in the left sidebar.
2. Type a short message describing what you changed.
3. Click **Commit**, then **Sync Changes**.

Do this before you start working and again before you close the Codespace.

## Team rules

- Only edit the files you own. If you need a change in someone else's file or in a shared file such as `requirements.txt`, message the owner.
- If Sync Changes reports a conflict, stop and message [your name]. Do not try to fix it yourself.
- Never put passwords or API keys in files.
- Every function needs a short docstring saying what it takes in and what it returns.

## Function conventions

A strategy is a function that takes a DataFrame of daily returns (the past window only) and returns target weights as a `pd.Series` indexed by fund ticker, long-only and summing to 1. Benchmarks and the AI manager all follow this shape, so `models/backtest.py` can run any of them. Expected returns and covariances passed to the optimizer must be annualized.

[Update this section as the team agrees on signatures.]

## Planned: portfolio chat assistant

Not started. The design is still to be decided as a team; it depends on the dashboard being built first.

The idea is a small chat panel in the dashboard, connected to the Claude API, where a client (or we, during the presentation) can ask questions such as "why am I only 40% in stocks?" or "what happens if I retire at 65 instead?".

- **It explains; it does not decide.** The portfolio always comes from our models. The assistant is given the client's profile, `PortfolioSpec` (including `spec.reasons`) and the explanation from `build_portfolio`, and answers from those. It should not invent its own allocations or recommend specific trades.
- **What-if questions call our code.** If we let it change inputs (retirement age, risk tolerance, goals), it should re-run `profile_to_spec` and `build_portfolio` through tool use, so every number it quotes comes from the model.
- **API key.** Each person keeps the key in `.streamlit/secrets.toml` (already ignored by git) or the `ANTHROPIC_API_KEY` environment variable. Never commit it. Agree as a team whose key is used and set a spending limit on it.
- **Open questions:** who builds it; whether it can run what-ifs or only explain; how it is presented in the report (it is a front end to the model, not part of the "AI" that makes decisions).

## Data

Prices come from Yahoo Finance through the `yfinance` package, adjusted for dividends and splits. The shared file `data/prices.csv` is committed so nobody has to download it. See `data/README.md`.
