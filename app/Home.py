"""Dashboard entry page. Run with: streamlit run app/Home.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # lets pages import from models/

st.set_page_config(page_title="AI Portfolio Manager", layout="wide")
st.title("AI Portfolio Manager")
st.write("Dashboard placeholder. Edit this file and save to see it update.")

prices_file = ROOT / "data" / "prices.csv"
if prices_file.exists():
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    st.subheader("Price data loaded")
    st.write(
        f"{prices.shape[1]} funds, "
        f"{prices.index.min().date()} to {prices.index.max().date()}"
    )
    first_valid = prices.apply(lambda s: s.dropna().iloc[0])
    st.line_chart(prices / first_valid)  # each fund rebased to 1
else:
    st.info("No price data yet. Run data/load_prices.py once and commit data/prices.csv.")
