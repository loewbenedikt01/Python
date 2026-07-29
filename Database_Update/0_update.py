import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import yfinance as yf
import pandas as pd
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
START_DATE   = '2000-01-01'
END_DATE     = datetime.now().strftime('%Y-%m-%d')
DATABASE_DIR = os.path.join(_ROOT, 'Database')
FRED_API_KEY = os.getenv('FRED_API_KEY', '8ce0ff451f8503c5fcc7aa27b9ecb3e0')

os.makedirs(DATABASE_DIR, exist_ok=True)

# ─── Ticker imports ───────────────────────────────────────────────────────────
from Database_Update.equities   import ticker_us
from equities import ticker_asia
from equities   import ticker_europe
from equities   import ticker_de
from equities import ticker_rotw
from bond          import ticker_bonds
from commodities   import ticker_commodities
from crypto        import ticker_crypto
from forex         import ticker_forex
from indices       import ticker_indices
from macro         import ticker_macro
from sectors       import ticker_sectors
from sentiment     import ticker_sentiment

# ─── Geography & currency mappings ───────────────────────────────────────────
# Exchange suffix → (country, continent, currency)
SUFFIX_MAP = {
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

NO NEEED
DIRECT_MAP = {
    '^GSPC': ('United States', 'North America', 'USD'),
    '^IXIC': ('United States', 'North America', 'USD'),
    '^NDX':  ('United States', 'North America', 'USD'),
    '^DJI':  ('United States', 'North America', 'USD'),
    '^RUT':  ('United States', 'North America', 'USD'),
    '^GDAXI':('Germany',       'Europe',        'EUR'),
    '^FTSE': ('United Kingdom','Europe',        'GBP'),
    '^FCHI': ('France',        'Europe',        'EUR'),
    '^N225': ('Japan',         'Asia',          'JPY'),
    '^HSI':  ('Hong Kong',     'Asia',          'HKD'),
    '^AXJO': ('Australia',     'Oceania',       'AUD'),
    '^GSPTSE':('Canada',       'North America', 'CAD'),
    '^BVSP': ('Brazil',        'South America', 'BRL'),
    'BTC-USD':('Global',       'Global',        'USD'),
    'ETH-USD':('Global',       'Global',        'USD'),
}


def get_geo(ticker, default_country, default_continent, default_currency='USD'):
    if ticker in DIRECT_MAP:
        row = DIRECT_MAP[ticker]
        return row[0], row[1], row[2]
    dot = ticker.rfind('.')
    if dot > 0:
        suffix = ticker[dot:]
        if suffix in SUFFIX_MAP:
            return SUFFIX_MAP[suffix]
    return default_country, default_continent, default_currency


def add_geo_columns(df, default_country, default_continent, default_currency='USD'):
    tickers  = df.index.get_level_values('Ticker')
    df       = df.copy()
    geo      = [get_geo(t, default_country, default_continent, default_currency) for t in tickers]
    df['Country']   = [g[0] for g in geo]
    df['Continent'] = [g[1] for g in geo]
    df['Currency']  = [g[2] for g in geo]
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_list(tickers):
    return list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)


def get_last_date(path):
    if not os.path.exists(path):
        return None
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        # Row-group stats give us min/max without reading data
        dates = []
        for rg in range(meta.num_row_groups):
            rg_meta = meta.row_group(rg)
            for col in range(rg_meta.num_columns):
                col_meta = rg_meta.column(col)
                if col_meta.path_in_schema in ('__index_level_0__', 'Date'):
                    stats = col_meta.statistics
                    if stats and stats.max is not None:
                        dates.append(pd.Timestamp(stats.max))
        if dates:
            return max(dates)
        # Fallback: read only the index
        df = pd.read_parquet(path, columns=[])
        idx = df.index
        date_vals = idx.get_level_values('Date') if isinstance(idx, pd.MultiIndex) else idx
        return pd.Timestamp(date_vals.max())
    except Exception:
        return None


def fetch_yfinance(tickers, start, end):
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


def fetch_fred(series_ids, start, end):
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
            r   = requests.get(url, timeout=15)
            r.raise_for_status()
            obs = r.json().get('observations', [])
            s   = pd.Series(
                {o['date']: float(o['value']) for o in obs if o['value'] != '.'},
                name=sid, dtype=float,
            )
            s.index = pd.to_datetime(s.index)
            frames.append(s)
        except Exception as e:
            print(f'  [WARN] FRED {sid}: {e}')
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# ─── Sector / Industry via yfinance (per ticker) ──────────────────────────────

def fetch_sector_industry(tickers: list, max_workers: int = 20) -> dict:
    """
    Fetch Sector and Industry per ticker using yfinance.Ticker.info.
    Uses a thread pool to parallelise the HTTP requests.
    """
    def _fetch_one(t):
        try:
            info = yf.Ticker(t).info
            return t, {
                'Sector':   info.get('sector')   or None,
                'Industry': info.get('industry') or None,
            }
        except Exception:
            return t, {'Sector': None, 'Industry': None}

    meta_map = {}
    print(f'  Fetching sector/industry for {len(tickers)} tickers '
          f'({max_workers} workers) ...')

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            t, meta = f.result()
            meta_map[t] = meta
            done += 1
            if done % 100 == 0 or done == len(tickers):
                print(f'    {done}/{len(tickers)} done ...')

    covered = sum(1 for v in meta_map.values() if v['Sector'])
    print(f'  Sector/Industry available for {covered}/{len(tickers)} tickers.')
    return meta_map


def apply_sector_industry(path: str, meta_map: dict) -> None:
    df = pd.read_parquet(path)
    t  = df.index.get_level_values('Ticker')
    df = df.copy()
    df['Sector']   = [meta_map.get(tk, {}).get('Sector')   for tk in t]
    df['Industry'] = [meta_map.get(tk, {}).get('Industry') for tk in t]
    df.to_parquet(path)
    unique   = t.unique()
    covered  = sum(1 for tk in unique if meta_map.get(tk, {}).get('Sector'))
    print(f'  Applied to {os.path.basename(path)} — {covered}/{len(unique)} tickers have sector.')


# ─── Non-equity updater (bonds, crypto, forex, etc.) ─────────────────────────

def update_yfinance_group(tickers, filename,
                          default_country='United States',
                          default_continent='North America',
                          default_currency='USD'):
    tickers = to_list(tickers)
    path    = os.path.join(DATABASE_DIR, filename)

    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last else START_DATE

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Downloading {len(tickers)} tickers from {start} ...')
    new_df = fetch_yfinance(tickers, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    new_df = add_geo_columns(new_df, default_country, default_continent, default_currency)

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows -> {filename}')


def update_fred_group(tickers, filename):
    tickers = to_list(tickers)
    path    = os.path.join(DATABASE_DIR, filename)

    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last else START_DATE

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Fetching {len(tickers)} FRED series from {start} ...')
    new_df = fetch_fred(tickers, start, END_DATE)
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
    print(f'  +{len(new_df)} rows -> {filename}')


# ─── Equity updater — single equities.parquet ─────────────────────────────────

# (ticker_list, default_country, default_continent, default_currency)
EQUITY_GROUPS = [
    (ticker_us,     'United States', 'North America', 'USD'),
    (ticker_asia,   'Asia',          'Asia',           'USD'),
    (ticker_europe, 'Europe',        'Europe',         'EUR'),
    (ticker_de,     'Germany',       'Europe',         'EUR'),
    (ticker_rotw,   'Global',        'Global',         'USD'),
]

def update_equities():
    path  = os.path.join(DATABASE_DIR, 'equities.parquet')
    last  = get_last_date(path)
    start = (last + timedelta(days=1)).strftime('%Y-%m-%d') if last else START_DATE

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Fetching from {start} ...')
    new_frames = []
    for tickers, def_country, def_cont, def_cur in EQUITY_GROUPS:
        tl = to_list(tickers)
        df = fetch_yfinance(tl, start, END_DATE)
        if df.empty:
            continue
        df = add_geo_columns(df, def_country, def_cont, def_cur)
        new_frames.append(df)
        print(f'    {def_country}/{def_cont}: +{len(df)} rows')

    if not new_frames:
        print('  No new data returned.')
        return

    new_df = pd.concat(new_frames)
    new_df = new_df[~new_df.index.duplicated(keep='last')].sort_index()

    # Keep only the columns we want (Sector/Industry will be added after)
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Country', 'Continent', 'Currency']
    new_df = new_df[[c for c in base_cols if c in new_df.columns]]

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        # New rows won't have Sector, Industry, or Close_USD yet — add as NaN
        # so concat doesn't drop those columns from existing rows
        for col in ('Sector', 'Industry', 'Close_USD'):
            if col in existing.columns and col not in new_df.columns:
                new_df[col] = None
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df

    combined.to_parquet(path)
    tickers_in_file = combined.index.get_level_values('Ticker').nunique()
    print(f'  equities.parquet: {combined.shape[0]:,} rows, {tickers_in_file:,} tickers')
    return combined


# ─── USD conversion ──────────────────────────────────────────────────────────
# Maps local currency code → yfinance forex ticker (gives USD per 1 unit of currency)
CURRENCY_TO_FOREX = {
    'AUD': 'AUDUSD=X',
    'BRL': 'BRLUSD=X',
    'CAD': 'CADUSD=X',
    'CHF': 'CHFUSD=X',
    'CNY': 'CNYUSD=X',
    'DKK': 'DKKUSD=X',
    'EGP': 'EGPUSD=X',
    'EUR': 'EURUSD=X',
    'GBP': 'GBPUSD=X',
    'HKD': 'HKDUSD=X',
    'IDR': 'IDRUSD=X',
    'ILS': 'ILSUSD=X',
    'INR': 'INRUSD=X',
    'JPY': 'JPYUSD=X',
    'KRW': 'KRWUSD=X',
    'MXN': 'MXNUSD=X',
    'MYR': 'MYRUSD=X',
    'NOK': 'NOKUSD=X',
    'NZD': 'NZDUSD=X',
    'PHP': 'PHPUSD=X',
    'PLN': 'PLNUSD=X',
    'SAR': 'SARUSD=X',
    'SEK': 'SEKUSD=X',
    'SGD': 'SGDUSD=X',
    'THB': 'THBUSD=X',
    'TWD': 'TWDUSD=X',
    'VND': 'VNDUSD=X',
    'ZAR': 'ZARUSD=X',
    'USD': None,   # no conversion needed
}


def compute_close_usd():
    """
    Add a Close_USD column to equities.parquet by multiplying Close by the
    daily USD exchange rate from forex.parquet.
    USD-denominated tickers get Close_USD = Close directly.
    Currencies without a rate fall back to Close (with a warning).
    """
    eq_path    = os.path.join(DATABASE_DIR, 'equities.parquet')
    forex_path = os.path.join(DATABASE_DIR, 'forex.parquet')

    if not os.path.exists(eq_path):
        print('  equities.parquet not found — skipping.')
        return

    df = pd.read_parquet(eq_path)
    if 'Currency' not in df.columns:
        print('  Currency column missing — skipping Close_USD.')
        return

    # ── Load daily rates from forex.parquet ───────────────────────────────────
    rates: dict[str, pd.Series] = {}   # currency -> Date-indexed Series of USD rate
    if os.path.exists(forex_path):
        try:
            forex_df = pd.read_parquet(forex_path, columns=['Close'])
            available = forex_df.index.get_level_values('Ticker').unique()
            for currency, fticker in CURRENCY_TO_FOREX.items():
                if fticker is None or fticker not in available:
                    continue
                rates[currency] = (
                    forex_df.xs(fticker, level='Ticker')['Close']
                    .sort_index()
                )
        except Exception as e:
            print(f'  [WARN] Could not load forex rates: {e}')
    else:
        print('  [WARN] forex.parquet not found — USD tickers only will be converted.')

    # ── Download any pairs still missing directly from yfinance ──────────────
    needed_currencies = df['Currency'].dropna().unique()
    missing = [
        c for c in needed_currencies
        if c != 'USD' and c not in rates and CURRENCY_TO_FOREX.get(c) is not None
    ]
    if missing:
        missing_tickers = [CURRENCY_TO_FOREX[c] for c in missing]
        print(f'  Fetching {len(missing_tickers)} missing forex pairs from yfinance ...')
        try:
            raw = yf.download(
                missing_tickers,
                start=str(df.index.get_level_values('Date').min().date()),
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    close_raw = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw.xs('Close', axis=1, level=0)
                else:
                    close_raw = raw[['Close']]
                    close_raw.columns = [missing_tickers[0]]
                for currency, fticker in zip(missing, missing_tickers):
                    if fticker in close_raw.columns:
                        rates[currency] = close_raw[fticker].dropna().sort_index()
        except Exception as e:
            print(f'  [WARN] Could not fetch missing forex pairs: {e}')

    # ── Only process rows where Close_USD is missing ─────────────────────────
    df         = df.copy()
    currencies = df['Currency'].fillna('USD')

    if 'Close_USD' not in df.columns:
        df['Close_USD'] = float('nan')

    todo = df['Close_USD'].isna()
    if not todo.any():
        print('  Close_USD already complete — nothing to do.')
        return

    close_usd = df['Close_USD'].copy()

    for currency in currencies[todo].unique():
        mask = todo & (currencies == currency)

        if currency == 'USD':
            close_usd[mask] = df.loc[mask, 'Close'].values
            continue

        rate_series = rates.get(currency)
        if rate_series is None:
            print(f'  [WARN] No USD rate for {currency} — using Close as-is')
            close_usd[mask] = df.loc[mask, 'Close'].values
            continue

        dates        = df.index.get_level_values('Date')[mask.values]
        rate_aligned = rate_series.reindex(dates, method='ffill').values
        close_usd[mask] = df.loc[mask, 'Close'].values * rate_aligned

    df['Close_USD'] = close_usd
    df.to_parquet(eq_path)

    n_filled  = int(todo.sum())
    n_missing = int(df['Close_USD'].isna().sum())
    print(f'  Close_USD: filled {n_filled:,} rows, '
          f'{n_missing} still NaN (no rate available).')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'=== Database Update — {END_DATE} ===\n')

    # ── Equities (combined into one file) ─────────────────────────────────────
    print('[Equities]')
    try:
        update_equities()
    except Exception as e:
        print(f'  [ERROR] {e}')

    # ── Sector / Industry (yfinance per ticker) ────────────────────────────────
    print('\n[Sector / Industry]')
    try:
        all_tickers = list({t for tl, *_ in EQUITY_GROUPS for t in to_list(tl)})
        meta_map    = fetch_sector_industry(all_tickers)
        apply_sector_industry(os.path.join(DATABASE_DIR, 'equities.parquet'), meta_map)
    except Exception as e:
        print(f'  [ERROR] {e}')

    # ── Non-equity asset classes ───────────────────────────────────────────────
    other_groups = [
        (ticker_bonds,       'bonds.parquet',       'United States', 'North America', 'USD'),
        (ticker_commodities, 'commodities.parquet', 'United States', 'North America', 'USD'),
        (ticker_crypto,      'crypto.parquet',      'Global',        'Global',        'USD'),
        (ticker_forex,       'forex.parquet',       'Global',        'Global',        'USD'),
        (ticker_indices,     'indices.parquet',     'United States', 'North America', 'USD'),
        (ticker_sectors,     'sectors.parquet',     'Global',        'Global',        'USD'),
        (ticker_sentiment,   'sentiment.parquet',   'United States', 'North America', 'USD'),
    ]
    labels = ['Bonds', 'Commodities', 'Crypto', 'Forex', 'Indices', 'Sectors', 'Sentiment']

    for label, (tickers, filename, country, continent, currency) in zip(labels, other_groups):
        print(f'\n[{label}]')
        try:
            update_yfinance_group(tickers, filename, country, continent, currency)
        except Exception as e:
            print(f'  [ERROR] {e}')
        time.sleep(3)   # avoid Yahoo rate limiting between groups

    print('\n[Macro (FRED)]')
    try:
        update_fred_group(ticker_macro, 'macro.parquet')
    except Exception as e:
        print(f'  [ERROR] {e}')

    # ── USD conversion (runs after forex is updated) ───────────────────────────
    print('\n[Close_USD Conversion]')
    try:
        compute_close_usd()
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
