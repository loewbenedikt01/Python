
'''
Download the Ticker Universe
Top 20 S&P 500 constituents per year
VIX for regime modelling and Sensitivity Analysis
SP500 as a Standardbenchmark
'''


import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')


OUTPUT_PATH = Path(r'C:\Users\benel\Coding\Python\Paper\_database')

START_DATE  = '1988-01-01'
END_DATE    = '2025-12-31'

tickers         = [
    'AAPL', 'AIG', 'AMGN', 
    'AMZN', 'AVGO', 'BA', 
    'BAC', 'BMY', 'BRK-B', 
    'C', 'COST', 'CSCO', 
    'CVX', 'DIS', 'FNMA', 
    'GE', 'GOOGL', 'HD', 
    'IBM', 'INTC', 'JNJ', 
    'JPM', 'KO', 'LLY', 
    'MA', 'MCD', 'META', 
    'MMM','MRK', 'MSFT', 
    'MO', 'NVDA', 'ORCL', 
    'PEP', 'PFE', 'PG', 
    'PM', 'PYPL', 'QCOM', 
    'SLB', 'T', 'TSLA', 
    'UNH', 'UPS', 'V', 
    'VZ', 'WFC', 'WMT', 
    'XOM',
]

benchmark_ticker    = '^GSPC'
vix_ticker          = '^VIX'
t_bill_3_month      = 'DTB3'
FRED_API            = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={t_bill_3_month}'

all_tickers         = [
    'AAPL', 'AIG', 'AMGN', 
    'AMZN', 'AVGO', 'BA', 
    'BAC', 'BMY', 'BRK-B', 
    'C', 'COST', 'CSCO', 
    'CVX', 'DIS', 'FNMA', 
    'GE', 'GOOGL', 'HD', 
    'IBM', 'INTC', 'JNJ', 
    'JPM', 'KO', 'LLY', 
    'MA', 'MCD', 'META', 
    'MMM','MRK', 'MSFT', 
    'MO', 'NVDA', 'ORCL', 
    'PEP', 'PFE', 'PG', 
    'PM', 'PYPL', 'QCOM', 
    'SLB', 'T', 'TSLA', 
    'UNH', 'UPS', 'V', 
    'VZ', 'WFC', 'WMT', 
    'XOM', '^VIX', '^GSPC',
]

def download_risk_free_rate(path:Path) -> pd.DataFrame:
    rf = pd.read_csv(FRED_API, index_col=0, parse_dates=True, na_values='.')
    rf.index.name = 'date'
    rf = rf[[t_bill_3_month]].apply(pd.to_numeric, errors='coerce').dropna()
    rf = rf.loc[START_DATE:END_DATE]
    rf.to_parquet(path)
    return rf

if __name__ == '__main__':

    DATA_PATH = OUTPUT_PATH
    DATA_PATH.mkdir(exist_ok=True)

    database = DATA_PATH / 'database.parquet'

    if not database.exists():
        print(f'Downloading {len(all_tickers)} tickers from 1990 to 2025')
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

        print(f'Saved to {database}')
    else:
        print('Data already exists, loading from disk')

    risk_free_rate = DATA_PATH / 'risk_free_rate.parquet'
    if not risk_free_rate.exists():
        rf = download_risk_free_rate(risk_free_rate)
        print(f'Saved {t_bill_3_month} {rf.index.min().date()}..{rf.index.max().date()} '
              f'({len(rf)} days) to {risk_free_rate}')
    else:
        print('Risk-free rate already exists, loading from disk.')
    
    prices = pd.read_parquet(database)