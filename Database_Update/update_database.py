
import sys
import os
import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# --------------
# PARAMS
# --------------

START_DATE = '2000-01-01'
END_DATE   = datetime.now().strftime('%Y-%m-%d')
DATABASE_DIR = os.path.join(_ROOT, 'Database')

# --------------
# API KEYS
# --------------

FRED_API_KEY = os.getenv('FRED_API_KEY', '8ce0ff451f8503c5fcc7aa27b9ecb3e0')


os.makedirs(DATABASE_DIR, exist_ok=True)



# --------------
# TICKERS
# --------------

ticker_bonds = {
    # --- Pure CBOE Yield Indices ---
    '^IRX': 'CBOE 13-Week Treasury Bill Yield',
    '^FVX': 'CBOE 5-Year Treasury Note Yield',
    '^TNX': 'CBOE 10-Year Treasury Note Yield',
    '^TYX': 'CBOE 30-Year Treasury Bond Yield',

    # --- Liquid Duration ETFs (Execution Proxies) ---
    'BIL': 'SPDR Bloomberg 1-3 Month T-Bill ETF',
    'SHV': 'iShares Short Treasury Bond ETF',
    'SHY': 'iShares 1-3 Year Treasury Bond ETF',
    'IEI': 'iShares 3-7 Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'TLH': 'iShares 10-20 Year Treasury Bond ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'GOVT': 'iShares Core U.S. Treasury ETF',
    'TIP': 'iShares TIPS Bond ETF',
    'STIP': 'iShares Short-Term TIPS ETF',
    'TMF': 'Direxion Daily 20+ Year Treasury Bull 3X Shares',
    'TMV': 'Direxion Daily 20+ Year Treasury Bear 3X Shares'
}



# --------------
# HELPERS
# --------------

def fetch_yfinance(
        tickers,
        start,
        end
):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ['Field', 'Ticker']
    else:
        ticker = tickers[0] if isinstance(tickers, list) else tickers
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, ticker) for col in raw.columns], names=['Field', 'Ticker']
        )

    try:
        df = raw.stack(level='Ticker', future_stack=True)
    except TypeError:
        df = raw.stack(level='Ticker')

    df.index.names = ['Date', 'Ticker']
    return df.sort_index()


def get_last_date(path):
    if not os.path.exists(path):
        return None
    df  = pd.read_parquet(path, columns=[])
    idx = df.index.get_level_values('Date')
    return pd.Timestamp(idx.max())


def update_yfinance_group(tickers, filename):
    ticker_list = list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)
    path        = os.path.join(DATABASE_DIR, filename)

    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last is not None else START_DATE

    if start > END_DATE:
        print(f'  {filename}: already up to date (last = {last.date()}).')
        return

    print(f'  {filename}: fetching {len(ticker_list)} tickers from {start} to {END_DATE} ...')
    new_df = fetch_yfinance(ticker_list, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows -> {filename} (total {len(combined)} rows)')


# --------------
# MAIN
# --------------

def main():
    print(f'=== Database Update — {END_DATE} ===')
    update_yfinance_group(ticker_bonds, 'bonds.parquet')
    print('=== Done ===')


if __name__ == '__main__':
    main()






