
"""
Download the Ticker Universe
Top 20 S&P 500 constituents per year
VIX for regime modelling and Sensitivity Analysis
SP500 as a Standardbenchmark
"""


import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")


OUTPUT_PATH = Path(r"C:\Users\benel\Coding\Python\Thesis\_database")

START_DATE  = '1990-01-01'
END_DATE    = '2025-12-31'

tickers             = [
    'AAPL', 'AIG', 'AMGN', 'AMZN', 'AVGO', 
    'BAC', 'BMY', 'BRK-B', 'C', 'COST', 
    'CSCO', 'CVX', 'DIS', 'GE', 'GOOGL', 
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 
    'KO', 'LLY', 'MA', 'META', 'MRK', 
    'MSFT', 'NVDA', 'ORCL', 'PEP', 'PFE', 
    'PG', 'PM', 'PYPL', 'QCOM', 'T', 
    'TSLA', 'UNH', 'UPS', 'V', 'VZ', 
    'WFC', 'WMT', 'XOM',
]

benchmark_ticker    = '^GSPC'
vix_ticker          = '^VIX'

all_tickers         = [
    'AAPL', 'AIG', 'AMGN', 'AMZN', 'AVGO', 
    'BAC', 'BMY', 'BRK-B', 'C', 'COST', 
    'CSCO', 'CVX', 'DIS', 'GE', 'GOOGL', 
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 
    'KO', 'LLY', 'MA', 'META', 'MRK', 
    'MSFT', 'NVDA', 'ORCL', 'PEP', 'PFE', 
    'PG', 'PM', 'PYPL', 'QCOM', 'T', 
    'TSLA', 'UNH', 'UPS', 'V', 'VZ', 
    'WFC', 'WMT', 'XOM', '^VIX', '^GSPC',
]

# Start
if __name__ == "__main__":

    DATA_PATH = OUTPUT_PATH
    DATA_PATH.mkdir(exist_ok=True)

    database = DATA_PATH / "database.parquet"

    if not database.exists():
        print(f"Downloading {len(all_tickers)} tickers from 1990 to 2025")
        raw = yf.download(
            all_tickers,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            actions=True,
            progress=True,
        )

        prices = raw[['Open', 'High', 'Low', 'Close', 'Adj Close',
              'Volume', 'Dividends', 'Stock Splits']]
        prices.to_parquet(database)

        print(f"Saved to {database}")
    else:
        print("Data already exists, loading from disk")

    prices = pd.read_parquet(database)