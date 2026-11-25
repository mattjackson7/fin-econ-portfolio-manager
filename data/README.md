# Data

## Fund universe

`universe.csv` lists the funds: ticker, name, asset class, bucket (`equity`, `bonds` or `real_assets`, used for client asset-class limits), and an ESG label. An optional `ref_weight` column (e.g. market cap) sets each fund's share of its bucket in the reference portfolio; without it, funds in a bucket share equally (1 for ESG-labelled funds). This list is a draft. Verify the tickers and names, and change the file to change the universe everywhere.

ESG-labelled funds are a proxy, since detailed ESG ratings are often paid. They were launched around 2016 to 2018 [confirm inception dates], so they have no data for 2008. A 2008 stress test must use a universe without them (pick the columns before calling `daily_returns`).

## Downloading prices

`load_prices.py` downloads adjusted daily prices into `prices.csv`. One person runs it once, then commits `prices.csv` so everyone shares identical data. Run this full sequence from the project folder in the Codespace terminal (Terminal > New Terminal):

```
cd /workspaces/YOUR-REPO-NAME
python3 data/load_prices.py
```

Use your actual repo name in place of `YOUR-REPO-NAME`. Then click Source Control, commit `data/prices.csv`, and Sync Changes.

## Rules

- Do not edit `prices.csv` by hand.
- If you re-download, tell the team, because results can change slightly when prices are revised.
- Record the download date here: [date].
