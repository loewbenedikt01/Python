
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

from equity_metrics import add_all_indicators

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

from ticker_files.bond import ticker_bonds
from ticker_files.commodities import ticker_commodities
from ticker_files.crypto import ticker_crypto
from ticker_files.forex import ticker_forex
from ticker_files.indices import ticker_indices
from ticker_files.equities import ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw





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
    df.columns.name = None
    df = df[[c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]]
    return df.sort_index()


def get_last_date(path):
    if not os.path.exists(path):
        return None
    df  = pd.read_parquet(path, columns=[])
    idx = df.index.get_level_values('Date')
    return pd.Timestamp(idx.max())


def fill_short_gaps(df, limit=2):
    """Forward-fill NaNs per ticker, but only across gaps of up to `limit` rows."""
    return (
        df.sort_index()
          .groupby(level='Ticker', group_keys=False)
          .apply(lambda g: g.ffill(limit=limit))
    )


def split_yield_column(df, yield_prefix='^'):
    """Move Close into a Yield column for tickers starting with `yield_prefix`
    (CBOE yield indices), zeroing their Close. ETF tickers get Yield = 0."""
    df = df.copy()
    is_yield = df.index.get_level_values('Ticker').str.startswith(yield_prefix)
    df['Yield'] = 0.0
    df.loc[is_yield, 'Yield'] = df.loc[is_yield, 'Close']
    df.loc[is_yield, 'Close'] = 0.0
    return df


def add_name_column(df, name_map):
    """Add a Name column looked up from a {ticker: name} dict."""
    df = df.copy()
    df['Name'] = df.index.get_level_values('Ticker').map(name_map)
    return df


# Exchange suffix -> local currency, for equities priced outside the US.
SUFFIX_TO_CURRENCY = {
    '.DE': 'EUR', '.PA': 'EUR', '.AS': 'EUR', '.MI': 'EUR', '.MC': 'EUR',
    '.HE': 'EUR', '.BR': 'EUR', '.LS': 'EUR', '.VI': 'EUR',
    '.ST': 'SEK', '.CO': 'DKK', '.WA': 'PLN',
    '.L': 'GBP', '.SW': 'CHF',
    '.T': 'JPY', '.HK': 'HKD', '.SS': 'CNY', '.SZ': 'CNY', '.NS': 'INR',
    '.TW': 'TWD', '.KS': 'KRW', '.SI': 'SGD', '.JK': 'IDR', '.KL': 'MYR', '.BK': 'THB',
    '.TO': 'CAD', '.AX': 'AUD', '.SA': 'BRL', '.MX': 'MXN',
}

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

MAPPING = {
    # Asia-Pacific
    '.T':  ('Japan',        'Asia',          'JPY'),
    '.HK': ('Hong Kong',    'Asia',          'HKD'),
    '.SS': ('China',        'Asia',          'CNY'),
    '.SZ': ('China',        'Asia',          'CNY'),
    '.NS': ('India',        'Asia',          'INR'),
    '.BO': ('India',        'Asia',          'INR'),
    '.TW': ('Taiwan',       'Asia',          'TWD'),
    '.KS': ('South Korea',  'Asia',          'KRW'),
    '.KQ': ('South Korea',  'Asia',          'KRW'),
    '.SI': ('Singapore',    'Asia',          'SGD'),
    '.JK': ('Indonesia',    'Asia',          'IDR'),
    '.KL': ('Malaysia',     'Asia',          'MYR'),
    '.BK': ('Thailand',     'Asia',          'THB'),
    '.MN': ('Philippines',  'Asia',          'PHP'),
    '.VN': ('Vietnam',      'Asia',          'VND'),
    '.SR': ('Saudi Arabia', 'Asia',          'SAR'),
    '.TA': ('Israel',       'Asia',          'ILS'),
    # Europe
    '.DE': ('Germany',       'Europe',       'EUR'),
    '.PA': ('France',        'Europe',       'EUR'),
    '.AS': ('Netherlands',   'Europe',       'EUR'),
    '.MI': ('Italy',         'Europe',       'EUR'),
    '.MC': ('Spain',         'Europe',       'EUR'),
    '.ST': ('Sweden',        'Europe',       'SEK'),
    '.CO': ('Denmark',       'Europe',       'DKK'),
    '.HE': ('Finland',       'Europe',       'EUR'),
    '.BR': ('Belgium',       'Europe',       'EUR'),
    '.VI': ('Austria',       'Europe',       'EUR'),
    '.LS': ('Portugal',      'Europe',       'EUR'),
    '.WA': ('Poland',        'Europe',       'PLN'),
    '.L':  ('United Kingdom','Europe',       'GBP'),
    '.SW': ('Switzerland',   'Europe',       'CHF'),
    '.IR': ('Ireland',       'Europe',       'EUR'),
    '.OL': ('Norway',        'Europe',       'NOK'),
    # Americas
    '.TO': ('Canada',        'North America','CAD'),
    '.SA': ('Brazil',        'South America','BRL'),
    '.MX': ('Mexico',        'North America','MXN'),
    # Oceania
    '.AX': ('Australia',     'Oceania',      'AUD'),
    '.NZ': ('New Zealand',   'Oceania',      'NZD'),
    # Africa
    '.JO': ('South Africa',  'Africa',       'ZAR'),
    '.EG': ('Egypt',         'Africa',       'EGP'),
}


def get_currency(ticker):
    dot = ticker.rfind('.')
    if dot > 0:
        return SUFFIX_TO_CURRENCY.get(ticker[dot:], 'USD')
    return 'USD'


def add_currency_column(df):
    df = df.copy()
    df['Currency'] = [get_currency(t) for t in df.index.get_level_values('Ticker')]
    return df


def get_country_continent(ticker):
    dot = ticker.rfind('.')
    if dot > 0:
        country, continent, _ = MAPPING.get(ticker[dot:], ('Unknown', 'Unknown', None))
        return country, continent
    return 'United States', 'North America'


def add_country_continent_column(df):
    df = df.copy()
    countries, continents = zip(*(get_country_continent(t) for t in df.index.get_level_values('Ticker')))
    df['Country']   = countries
    df['Continent'] = continents
    return df


def compute_close_usd(equities_filename='equities.parquet', forex_filename='forex.parquet'):
    """Add a Close_USD column to the equities file by converting each row's
    Close using the matching daily FX rate from the forex file."""
    eq_path = os.path.join(DATABASE_DIR, equities_filename)
    fx_path = os.path.join(DATABASE_DIR, forex_filename)

    if not os.path.exists(eq_path):
        print('  equities file not found — skipping Close_USD.')
        return
    if not os.path.exists(fx_path):
        print('  forex file not found — skipping Close_USD.')
        return

    df = pd.read_parquet(eq_path)
    if 'Currency' not in df.columns:
        df = add_currency_column(df)

    forex_close = pd.read_parquet(fx_path, columns=['Close'])['Close']

    rates = {}
    available = forex_close.index.get_level_values('Ticker').unique()
    for currency, (fx_ticker, direction) in FOREX_RATE_INFO.items():
        if fx_ticker not in available:
            continue
        series = forex_close.xs(fx_ticker, level='Ticker').sort_index()
        series = (1.0 / series) if direction == 'inverse' else series
        rates[currency] = series.ffill().bfill()

    dates      = df.index.get_level_values('Date')
    currencies = df['Currency']
    close_usd  = df['Close'].copy()

    for currency in currencies.unique():
        if currency == 'USD':
            continue
        mask = (currencies == currency).values
        rate_series = rates.get(currency)
        if rate_series is None:
            print(f'  [WARN] no USD rate for {currency} — leaving Close as-is')
            continue
        aligned = rate_series.reindex(dates[mask], method='ffill').bfill().values
        close_usd.values[mask] = df['Close'].values[mask] * aligned

    df['Close_USD'] = close_usd
    df.to_parquet(eq_path)
    n_converted = int((currencies != 'USD').sum())
    print(f'  Close_USD added to {equities_filename} ({n_converted:,} non-USD rows converted).')


def backfill_country_continent(filename='equities.parquet'):
    """One-off: stamp Country/Continent columns onto an already-saved equities
    file that predates add_country_continent_column()."""
    path = os.path.join(DATABASE_DIR, filename)
    if not os.path.exists(path):
        print('  equities file not found — skipping Country/Continent backfill.')
        return

    df = pd.read_parquet(path)
    df = add_country_continent_column(df)
    df.to_parquet(path)
    print(f'  Country/Continent backfilled in {filename}.')


def update_equity_indicators(filename='equities.parquet', price_col='Close'):
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


def _finalize_and_save(new_df, path, filename):
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined = fill_short_gaps(combined, limit=2)

    # Only require the actual price columns to be present — a newly added
    # metadata column (Name, Currency, ...) being NaN on old rows must not
    # wipe out real price history.
    core_cols = [c for c in ('Open', 'High', 'Low', 'Close', 'Volume') if c in combined.columns]
    before = len(combined)
    combined = combined.dropna(how='any', subset=core_cols)
    dropped = before - len(combined)

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows -> {filename} (total {len(combined)} rows, dropped {dropped} NaN rows)')


def update_yfinance_group(tickers, filename, yield_prefix=None):
    ticker_list = list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)
    name_map    = tickers if isinstance(tickers, dict) else None
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

    if yield_prefix is not None:
        new_df = split_yield_column(new_df, yield_prefix)

    if name_map is not None:
        new_df = add_name_column(new_df, name_map)

    _finalize_and_save(new_df, path, filename)


def update_equities_group(ticker_groups, filename):
    """Fetch several {ticker: name} dicts (one per region) separately — so one
    slow/failing region doesn't block the rest — then combine into one file."""
    combined_map = {}
    for group in ticker_groups:
        combined_map.update(group)

    path  = os.path.join(DATABASE_DIR, filename)
    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last is not None else START_DATE

    if start > END_DATE:
        print(f'  {filename}: already up to date (last = {last.date()}).')
        return

    frames = []
    for group in ticker_groups:
        ticker_list = list(group.keys())
        print(f'  {filename}: fetching {len(ticker_list)} tickers from {start} to {END_DATE} ...')
        df = fetch_yfinance(ticker_list, start, END_DATE)
        if not df.empty:
            frames.append(df)

    if not frames:
        print('  No new data returned.')
        return

    new_df = pd.concat(frames)
    new_df = add_name_column(new_df, combined_map)
    new_df = add_currency_column(new_df)
    new_df = add_country_continent_column(new_df)

    _finalize_and_save(new_df, path, filename)


# --------------
# MAIN
# --------------

def main():
    print(f'=== Database Update — {END_DATE} ===')
    #update_yfinance_group(ticker_bonds, 'bonds.parquet', yield_prefix='^')
    #update_yfinance_group(ticker_commodities, 'commodities.parquet')
    #update_yfinance_group(ticker_crypto, 'crypto.parquet')
    update_yfinance_group(ticker_forex, 'forex.parquet')  # needed by compute_close_usd()
    #update_yfinance_group(ticker_indices, 'indices.parquet')
    update_equities_group([ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw], 'equities.parquet')
    compute_close_usd()
    update_equity_indicators()

    print('=== Done ===')


if __name__ == '__main__':
    main()






