import os
import sys
import time
import concurrent.futures as cf

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Database._updating.update_database import DATABASE_DIR, START_DATE
from Database._updating.indexing.generate_id import add_corporate_id
from Database._updating.indexing.corporate_info import CORPORATE_FILENAME


# ----
# PARAMETERS
# ----

FINANCIAL_FILENAME = 'equities_financial.parquet'
SHARES_FILENAME    = 'equities_shares.parquet'
FOREX_FILENAME     = 'forex.parquet'
MAX_WORKERS        = 10
RETRIES            = 3


# ----
# CURRENCY
# ----

# currency -> (yfinance forex ticker, quote direction).
# 'direct'  = ticker is CCYUSD=X (1 CCY = rate USD) -> multiply.
# 'inverse' = ticker is USDCCY=X (1 USD = rate CCY) -> divide.
FOREX_RATE_INFO = {
    'EUR': ('EURUSD=X', 'direct'),
    'GBP': ('GBPUSD=X', 'direct'),
    'AUD': ('AUDUSD=X', 'direct'),
    'JPY': ('USDJPY=X', 'inverse'),
    'CAD': ('USDCAD=X', 'inverse'),
    'CHF': ('USDCHF=X', 'inverse'),
    'MXN': ('USDMXN=X', 'inverse'),
    'BRL': ('USDBRL=X', 'inverse'),
    'CNY': ('USDCNY=X', 'inverse'),
    'HKD': ('USDHKD=X', 'inverse'),
    'INR': ('USDINR=X', 'inverse'),
    'KRW': ('USDKRW=X', 'inverse'),
    'SGD': ('USDSGD=X', 'inverse'),
    'TWD': ('USDTWD=X', 'inverse'),
    'THB': ('USDTHB=X', 'inverse'),
    'MYR': ('USDMYR=X', 'inverse'),
    'IDR': ('USDIDR=X', 'inverse'),
    'PLN': ('USDPLN=X', 'inverse'),
    'SEK': ('USDSEK=X', 'inverse'),
    'DKK': ('USDDKK=X', 'inverse'),
}


def usd_rates(forex_close):
    """
    {currency: daily Series of USD per 1 unit of currency}.
    """
    rates = {}
    available = forex_close.index.get_level_values('Ticker').unique()
    for currency, (fx_ticker, direction) in FOREX_RATE_INFO.items():
        if fx_ticker not in available:
            continue
        series = forex_close.xs(fx_ticker, level='Ticker').sort_index()
        series = (1.0 / series) if direction == 'inverse' else series
        rates[currency] = series.ffill().bfill()
    return rates


def usd_factor(df, currency_map, rates):
    """
    Per-row multiplier from local currency to USD (1.0 for USD rows).
    """
    tickers    = df.index.get_level_values('Ticker')
    dates      = df.index.get_level_values('Date')
    currencies = pd.Series(tickers.map(currency_map), index=df.index).fillna('USD')
    factor     = pd.Series(1.0, index=df.index)

    for currency in currencies.unique():
        if currency == 'USD':
            continue
        mask = (currencies == currency).values
        rate = rates.get(currency)
        if rate is None:
            print(f'  [WARN] no USD rate for {currency} — leaving prices unconverted')
            continue
        factor.values[mask] = rate.reindex(dates[mask], method='ffill').bfill().values
    return factor


# ----
# SHARES OUTSTANDING
# ----

def _clean_shares(s, window=11, band=(0.9, 1.1)):
    """
    Drop Yahoo glitches (one-day spikes, short blocks at a wrong level):
    points outside `band` of their centred rolling median are removed and
    the previous valid count carries forward instead.
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    ratio = s / med
    return s[(ratio > band[0]) & (ratio < band[1])]


def _split_adjust_shares(s, splits, lag_days=180):
    """
    Put every share count on today's split basis. Counts dated before a split
    are scaled by its ratio. Yahoo often keeps reporting the old count for a
    while after the split, so counts in the `lag_days` after it are scaled too
    when that brings them closer to the post-lag level.
    """
    s = s.copy()
    for split_date, ratio in splits.sort_index(ascending=False).items():
        if ratio <= 0 or ratio == 1:
            continue
        s.loc[s.index < split_date] *= ratio

        window = (s.index >= split_date) & (s.index < split_date + pd.Timedelta(days=lag_days))
        after  = s[s.index >= split_date + pd.Timedelta(days=lag_days)]
        ref    = after.iloc[:5].median() if not after.empty else s[window].max()
        if pd.isna(ref) or not window.any():
            continue
        raw    = s[window]
        scaled = raw * ratio
        closer = (np.log(scaled / ref).abs() < np.log(raw / ref).abs())
        s.loc[raw.index[closer]] = scaled[closer]
    return s


def _fetch_shares(ticker):
    """
    Daily shares outstanding on today's split basis, so it matches the
    split-adjusted Close. yfinance history only goes back to ~2015; if it has
    none, the current share count is used as a single point.
    """
    for attempt in range(RETRIES):
        try:
            tk = yf.Ticker(ticker)
            s  = tk.get_shares_full(start=START_DATE)
            if s is None or s.empty:
                current = tk.fast_info.get('shares')
                if not current:
                    return ticker, None
                s = pd.Series([float(current)], index=[pd.Timestamp.today().normalize()])
            else:
                s.index = s.index.tz_localize(None).normalize()
                s = s.groupby(level=0).last().astype(float)

            splits = tk.splits
            if splits is not None and not splits.empty:
                splits.index = splits.index.tz_localize(None).normalize()
                s = _split_adjust_shares(s, splits)
            return ticker, _clean_shares(s)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return ticker, None


def update_shares(tickers, filename=SHARES_FILENAME):
    """
    Fetch the shares-outstanding history for every ticker into its own file
    (index Date, Ticker; column Shares). Refetched in full each run.
    """
    print(f'  Fetching shares outstanding for {len(tickers)} tickers ...')
    frames, missing = [], []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (ticker, s) in enumerate(ex.map(_fetch_shares, tickers), 1):
            if s is None or s.empty:
                missing.append(ticker)
            else:
                frames.append(pd.DataFrame({'Date': s.index, 'Ticker': ticker, 'Shares': s.values}))
            if i % 100 == 0:
                print(f'    ...{i}/{len(tickers)}')

    df = pd.concat(frames).set_index(['Date', 'Ticker']).sort_index()
    df.to_parquet(os.path.join(DATABASE_DIR, filename))
    print(f'  {filename}: {len(df)} rows for {len(tickers) - len(missing)} tickers ({len(missing)} without data).')


def market_cap_mapping(df, shares, backfill=False):
    """
    Shares outstanding per (Date, Ticker) row of `df`: the latest reported
    count on or before each date. Dates before the first report stay NaN
    unless `backfill` (then the earliest count is used — approximate).
    """
    left = pd.DataFrame({
        'Date':   pd.to_datetime(df.index.get_level_values('Date')).astype('datetime64[ns]'),
        'Ticker': df.index.get_level_values('Ticker').astype(str),
        'pos':    range(len(df)),
    }).sort_values('Date')
    right = shares.reset_index()
    right['Date']   = pd.to_datetime(right['Date']).astype('datetime64[ns]')
    right['Ticker'] = right['Ticker'].astype(str)

    merged = pd.merge_asof(left, right.sort_values('Date'), on='Date', by='Ticker', direction='backward')
    if backfill:
        merged = merged.sort_values(['Ticker', 'Date'])
        merged['Shares'] = merged.groupby('Ticker')['Shares'].bfill()

    return pd.Series(merged.set_index('pos')['Shares'].sort_index().values, index=df.index)


# ----
# UPDATE
# ----

def update_financial_info(filename=FINANCIAL_FILENAME, backfill_shares=False):
    """
    Add corporate_id, USD prices, shares outstanding and market cap to the
    daily equities price file.
    """
    path     = os.path.join(DATABASE_DIR, filename)
    fx_path  = os.path.join(DATABASE_DIR, FOREX_FILENAME)
    map_path = os.path.join(DATABASE_DIR, CORPORATE_FILENAME)
    sh_path  = os.path.join(DATABASE_DIR, SHARES_FILENAME)

    for label, p in ((filename, path), (FOREX_FILENAME, fx_path), (CORPORATE_FILENAME, map_path)):
        if not os.path.exists(p):
            print(f'  {label} not found — skipping financial info.')
            return

    df = add_corporate_id(pd.read_parquet(path))

    currency_map = pd.read_parquet(map_path, columns=['Currency'])['Currency']
    rates        = usd_rates(pd.read_parquet(fx_path, columns=['Close'])['Close'])
    factor       = usd_factor(df, currency_map, rates)

    df['Close_USD'] = df['Close'] * factor
    if 'Adj_Close' in df.columns:
        df['Adj_Close_USD'] = df['Adj_Close'] * factor

    if os.path.exists(sh_path):
        df['Shares']         = market_cap_mapping(df, pd.read_parquet(sh_path), backfill_shares)
        df['Market_Cap']     = df['Close'] * df['Shares']
        df['Market_Cap_USD'] = df['Market_Cap'] * factor
    else:
        print(f'  {SHARES_FILENAME} not found — skipping market cap.')

    df.to_parquet(path)
    n_mcap = int(df['Market_Cap'].notna().sum()) if 'Market_Cap' in df.columns else 0
    print(f'  {filename}: corporate_id, USD prices and market cap updated '
          f'({n_mcap:,} of {len(df):,} rows with market cap).')


if __name__ == '__main__':
    update_financial_info()
