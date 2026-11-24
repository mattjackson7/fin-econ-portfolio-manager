"""Download adjusted daily prices for the funds in data/universe.csv.

Run once from the repo root, then commit data/prices.csv so everyone on the team
shares the same data:
    python3 data/load_prices.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
START = "2000-01-01"


def main() -> None:
    universe = pd.read_csv(ROOT / "data" / "universe.csv")
    tickers = universe["ticker"].tolist()
    prices = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    prices = prices[tickers]  # keep the universe order
    out = ROOT / "data" / "prices.csv"
    prices.to_csv(out)
    print(f"Saved {out}")
    print("First available date per fund:")
    print(prices.apply(lambda s: s.first_valid_index()).to_string())


if __name__ == "__main__":
    main()
