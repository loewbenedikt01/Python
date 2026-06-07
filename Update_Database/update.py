import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

# ─── Configuration ────────────────────────────────────────────────────────────
START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATABASE_DIR = os.path.join(_ROOT, 'Database')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

os.makedirs(DATABASE_DIR, exist_ok=True)

# ─── Ticker imports ───────────────────────────────────────────────────────────
from US_equities import ticker_us
from ASIA_equities import ticker_asia
from EU_equities import ticker_eu
from DE_equities import ticker_de
from bond import ticker_bonds
from commodities import ticker_commodities
from crypto import ticker_crypto
from forex import ticker_forex
from indices import ticker_indices
from macro import ticker_macro
from sectors import ticker_sectors
from sentiment import ticker_sentiment
from ROTW_equities import ticker_rotw


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_list(tickers):
    return list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)


def get_last_date(path):
    """Return the latest date stored in a parquet file, or None if absent."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        idx = df.index
        dates = idx.get_level_values('Date') if isinstance(idx, pd.MultiIndex) else idx
        return pd.Timestamp(dates.max())
    except Exception:
        return None


def fetch_yfinance(tickers, start, end):
    """
    Download OHLCV data for a list of tickers.
    Returns a DataFrame with (Date, Ticker) MultiIndex and OHLCV columns.
    """
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
        # Single ticker — yfinance returns flat columns
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


def fetch_fred(series_ids, start, end):
    """
    Fetch FRED observations for each series ID.
    Returns a wide DataFrame with Date as index and series IDs as columns.
    """
    if not FRED_API_KEY:
        print('  [WARN] FRED_API_KEY not set — skipping macro.')
        return pd.DataFrame()

    frames = []
    for sid in series_ids:
        url = (
            'https://api.stlouisfed.org/fred/series/observations'
            f'?series_id={sid}&api_key={FRED_API_KEY}'
            f'&file_type=json&observation_start={start}&observation_end={end}'
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            obs = r.json().get('observations', [])
            s = pd.Series(
                {o['date']: float(o['value']) for o in obs if o['value'] != '.'},
                name=sid, dtype=float,
            )
            s.index = pd.to_datetime(s.index)
            frames.append(s)
        except Exception as e:
            print(f'  [WARN] FRED {sid}: {e}')

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _append_and_save(path, new_df):
    """Concat new rows onto existing parquet, dedup by index, save."""
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()
    combined.to_parquet(path)
    return len(new_df)


# ─── Group updaters ───────────────────────────────────────────────────────────

def update_yfinance_group(tickers, filename):
    tickers = to_list(tickers)
    path = os.path.join(DATABASE_DIR, filename)

    last = get_last_date(path)
    if last is not None:
        start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  Last date: {last.date()} → fetching from {start}')
    else:
        start = START_DATE
        print(f'  No existing data → fetching from {start}')

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Downloading {len(tickers)} tickers ...')
    new_df = fetch_yfinance(tickers, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    n = _append_and_save(path, new_df)
    print(f'  +{n} rows → {filename}')


def update_fred_group(tickers, filename):
    tickers = to_list(tickers)
    path = os.path.join(DATABASE_DIR, filename)

    last = get_last_date(path)
    if last is not None:
        start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  Last date: {last.date()} → fetching from {start}')
    else:
        start = START_DATE
        print(f'  No existing data → fetching from {start}')

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Fetching {len(tickers)} FRED series ...')
    new_df = fetch_fred(tickers, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    n = _append_and_save(path, new_df)
    print(f'  +{n} rows → {filename}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'=== Database Update — {END_DATE} ===\n')

    yf_groups = [
        ('US equities',   ticker_us,         'US_equities.parquet'),
        ('Asia equities', ticker_asia,        'ASIA_equities.parquet'),
        ('EU equities',   ticker_eu,          'EU_equities.parquet'),
        ('DE equities',   ticker_de,          'DE_equities.parquet'),
        ('ROTW equities', ticker_rotw,        'ROTW_equities.parquet'),
        ('Bonds',         ticker_bonds,       'bonds.parquet'),
        ('Commodities',   ticker_commodities, 'commodities.parquet'),
        ('Crypto',        ticker_crypto,      'crypto.parquet'),
        ('Forex',         ticker_forex,       'forex.parquet'),
        ('Indices',       ticker_indices,     'indices.parquet'),
        ('Sectors',       ticker_sectors,     'sectors.parquet'),
        ('Sentiment',     ticker_sentiment,   'sentiment.parquet'),
    ]

    for name, tickers, filename in yf_groups:
        print(f'[{name}]')
        try:
            update_yfinance_group(tickers, filename)
        except Exception as e:
            print(f'  [ERROR] {e}')

    print('\n[Macro (FRED)]')
    try:
        update_fred_group(ticker_macro, 'macro.parquet')
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
