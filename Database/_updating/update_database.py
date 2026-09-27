import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_ROOT)
sys.path.insert(0, _PROJECT)

from Financial.Metrics.equity_metrics import add_all_indicators


# ----
# DIRECTORIES
# ----

DATABASE_DIR = os.path.join(_ROOT, '_database')
os.makedirs(DATABASE_DIR, exist_ok=True)


# ----
# API KEYS
# ----

def load_api_keys(path=os.path.join(_PROJECT, 'api_keys.txt')):
    """
    Read KEYs from api_keys.txt
    """
    keys = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    keys[k.strip()] = v.strip()
    return keys


KEYS = load_api_keys()

CLINICAL_TRIALS_API = 'https://clinicaltrials.gov/api/v2/studies'
FRED_API_KEY        = KEYS.get('FRED_API_KEY')
ALPACA_KEY_ID       = KEYS.get('ALPACA_KEY_ID')
ALPACA_SECRET       = KEYS.get('ALPACA_SECRET')
NAME                = KEYS.get('NAME')
EMAIL               = KEYS.get('EMAIL')
SEC_EDGAR           = 'https://data.sec.gov/submissions/'

# ----
# PARAMETERS
# ----

START_DATE = '1995-01-01'
END_DATE   = datetime.now().strftime('%Y-%m-%d')


# ----
# TICKERS
# ----

from Database._tickers.bond import ticker_bonds
from Database._tickers.commodities import ticker_commodities
from Database._tickers.crypto import ticker_crypto
from Database._tickers.equities import ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw
from Database._tickers.forex import ticker_forex
from Database._tickers.indices import ticker_indices
#from Database._tickers.macro import ticker_macro
#from Database._tickers.sectors import ticker_sectors
#from Database._tickers.sentiment import ticker_sentiment


# ----
# HELPERS
# ----

def fetch_yfinance(tickers, start, end, auto_adjust=True):
    """
    auto_adjust=True : OHLC adjusted for splits and dividends.
    auto_adjust=False: OHLC split-adjusted only, plus Adj_Close (splits + dividends).
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
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
    df.columns.name = None
    df = df.rename(columns={'Adj Close': 'Adj_Close'})
    df = df[[c for c in ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'] if c in df.columns]]
    return df.sort_index()


def get_last_date(path):
    if not os.path.exists(path):
        return None
    df  = pd.read_parquet(path, columns=[])
    idx = df.index.get_level_values('Date')
    return pd.Timestamp(idx.max())


def get_start_date(path):
    """
    Day after the last stored date, or START_DATE for a fresh file.
    """
    last = get_last_date(path)
    return (last + timedelta(days=1)).strftime('%Y-%m-%d') if last is not None else START_DATE


def fill_short_gaps(df, limit=2):
    """
    Forward-fill NaNs per ticker, but only across gaps of up to `limit` rows.
    """
    return (
        df.sort_index()
          .groupby(level='Ticker', group_keys=False)
          .apply(lambda g: g.ffill(limit=limit))
    )


def split_yield_column(df, yield_prefix='^'):
    """
    Move Close into a Yield column for tickers starting with `yield_prefix`
    (CBOE yield indices), zeroing their Close. ETF tickers get Yield = 0.
    """
    df = df.copy()
    is_yield = df.index.get_level_values('Ticker').str.startswith(yield_prefix)
    df['Yield'] = 0.0
    df.loc[is_yield, 'Yield'] = df.loc[is_yield, 'Close']
    df.loc[is_yield, 'Close'] = 0.0
    return df


def add_name_column(df, name_map):
    """
    Add a Name column looked up from a {ticker: name} dict.
    """
    df = df.copy()
    df['Name'] = df.index.get_level_values('Ticker').map(name_map)
    return df


def _finalize_and_save(new_df, path, filename):
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined = fill_short_gaps(combined, limit=2)

    core_cols = [c for c in ('Open', 'High', 'Low', 'Close', 'Volume') if c in combined.columns]
    before = len(combined)
    combined = combined.dropna(how='any', subset=core_cols)
    dropped = before - len(combined)

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows -> {filename} (total {len(combined)} rows, dropped {dropped} NaN rows)')


# ----
# UPDATES
# ----

def update_yfinance_group(tickers, filename, yield_prefix=None):
    ticker_list = list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)
    name_map    = tickers if isinstance(tickers, dict) else None
    path        = os.path.join(DATABASE_DIR, filename)
    start       = get_start_date(path)

    if start > END_DATE:
        print(f'  {filename}: already up to date.')
        return

    print(f'  {filename}: fetching {len(ticker_list)} tickers from {start} to {END_DATE} ...')
    new_df = fetch_yfinance(ticker_list, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    if yield_prefix is not None:
        new_df = split_yield_column(new_df, yield_prefix)

    if name_map is not None:
        new_df = add_name_column(new_df, name_map)

    _finalize_and_save(new_df, path, filename)


def update_equities_group(ticker_groups, filename, auto_adjust=False):
    """Fetch several {ticker: name} dicts (one per region) separately — so one
    slow/failing region doesn't block the rest — then combine into one file.
    Close is split-adjusted only (needed for market cap); Adj_Close also
    includes dividends. Per-company metadata lives in equities_mapped.parquet
    — see indexing/corporate_info.py."""
    path  = os.path.join(DATABASE_DIR, filename)
    start = get_start_date(path)

    if start > END_DATE:
        print(f'  {filename}: already up to date.')
        return

    frames = []
    for group in ticker_groups:
        ticker_list = list(group.keys())
        print(f'  {filename}: fetching {len(ticker_list)} tickers from {start} to {END_DATE} ...')
        df = fetch_yfinance(ticker_list, start, END_DATE, auto_adjust=auto_adjust)
        if not df.empty:
            frames.append(df)

    if not frames:
        print('  No new data returned.')
        return

    _finalize_and_save(pd.concat(frames), path, filename)


def update_equity_indicators(filename='equities_financial.parquet', price_col='Close'):
    """Recompute technical indicators (SMA/EMA/RSI/MACD/Bollinger/...) over the
    full price history of every ticker and save them back into the file."""
    path = os.path.join(DATABASE_DIR, filename)
    if not os.path.exists(path):
        print('  equities file not found — skipping indicators.')
        return

    df = pd.read_parquet(path)
    df = add_all_indicators(df, price_col=price_col)
    df.to_parquet(path)
    print(f'  Indicators updated in {filename}.')


# ----
# MAIN
# ----

def main():
    from Database._updating.indexing.corporate_info import update_corporate_info
    from Database._updating.indexing.financial_info import update_shares, update_financial_info

    equity_groups = [ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw]
    equity_names  = {t: n for group in equity_groups for t, n in group.items()}

    print(f'=== Database Update — {END_DATE} ===')
    update_yfinance_group(ticker_bonds, 'bonds.parquet', yield_prefix='^')
    update_yfinance_group(ticker_commodities, 'commodities.parquet')
    update_yfinance_group(ticker_crypto, 'crypto.parquet')
    update_yfinance_group(ticker_forex, 'forex.parquet')          # needed for USD prices
    update_yfinance_group(ticker_indices, 'indices.parquet')
    update_equities_group(equity_groups, 'equities_financial.parquet')
    update_corporate_info(equity_names)                            # -> equities_mapped.parquet
    update_shares(list(equity_names))                              # -> equities_shares.parquet
    update_financial_info()                                        # corporate_id, USD, market cap
    update_equity_indicators()
    print('=== Done ===')


if __name__ == '__main__':
    main()
