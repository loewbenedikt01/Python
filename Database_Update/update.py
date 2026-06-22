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
FRED_API_KEY = os.getenv('FRED_API_KEY', '8ce0ff451f8503c5fcc7aa27b9ecb3e0')

os.makedirs(DATABASE_DIR, exist_ok=True)

# ─── Ticker imports ───────────────────────────────────────────────────────────
from US_equities import ticker_us
from ASIA_equities import ticker_asia
from EU_equities import ticker_europe
from DE_equities import ticker_de
from ROTW_equities import ticker_rotw
from bond import ticker_bonds
from commodities import ticker_commodities
from crypto import ticker_crypto
from forex import ticker_forex
from indices import ticker_indices
from macro import ticker_macro
from sectors import ticker_sectors
from sentiment import ticker_sentiment

# ─── Geography mappings ───────────────────────────────────────────────────────

# Exchange suffix ->(country, continent)
SUFFIX_MAP = {
    # Asia-Pacific
    '.T':   ('Japan',         'Asia'),
    '.HK':  ('Hong Kong',     'Asia'),
    '.SS':  ('China',         'Asia'),
    '.SZ':  ('China',         'Asia'),
    '.NS':  ('India',         'Asia'),
    '.BO':  ('India',         'Asia'),
    '.TW':  ('Taiwan',        'Asia'),
    '.KS':  ('South Korea',   'Asia'),
    '.KQ':  ('South Korea',   'Asia'),
    '.SI':  ('Singapore',     'Asia'),
    '.JK':  ('Indonesia',     'Asia'),
    '.KL':  ('Malaysia',      'Asia'),
    '.BK':  ('Thailand',      'Asia'),
    '.MN':  ('Philippines',   'Asia'),
    '.VN':  ('Vietnam',       'Asia'),
    '.SR':  ('Saudi Arabia',  'Asia'),
    '.TA':  ('Israel',        'Asia'),
    # Europe
    '.DE':  ('Germany',       'Europe'),
    '.PA':  ('France',        'Europe'),
    '.AS':  ('Netherlands',   'Europe'),
    '.MI':  ('Italy',         'Europe'),
    '.MC':  ('Spain',         'Europe'),
    '.ST':  ('Sweden',        'Europe'),
    '.CO':  ('Denmark',       'Europe'),
    '.HE':  ('Finland',       'Europe'),
    '.BR':  ('Belgium',       'Europe'),
    '.VI':  ('Austria',       'Europe'),
    '.LS':  ('Portugal',      'Europe'),
    '.WA':  ('Poland',        'Europe'),
    '.L':   ('United Kingdom','Europe'),
    '.SW':  ('Switzerland',   'Europe'),
    '.IR':  ('Ireland',       'Europe'),
    '.OL':  ('Norway',        'Europe'),
    # Americas
    '.TO':  ('Canada',        'North America'),
    '.SA':  ('Brazil',        'South America'),
    '.MX':  ('Mexico',        'North America'),
    # Oceania
    '.AX':  ('Australia',     'Oceania'),
    '.NZ':  ('New Zealand',   'Oceania'),
    # Africa
    '.JO':  ('South Africa',  'Africa'),
    '.EG':  ('Egypt',         'Africa'),
}

# Ticker-level overrides for indices and other non-suffix tickers
DIRECT_MAP = {
    # ── US broad market ──────────────────────────────────────────────────────
    '^GSPC': ('United States', 'North America'),
    '^IXIC': ('United States', 'North America'),
    '^NDX':  ('United States', 'North America'),
    '^DJI':  ('United States', 'North America'),
    '^RUT':  ('United States', 'North America'),
    '^RUA':  ('United States', 'North America'),
    '^RUI':  ('United States', 'North America'),
    '^MID':  ('United States', 'North America'),
    '^OEX':  ('United States', 'North America'),
    '^NYA':  ('United States', 'North America'),
    '^XAX':  ('United States', 'North America'),
    '^RUJ':  ('United States', 'North America'),
    '^RUO':  ('United States', 'North America'),
    '^RLG':  ('United States', 'North America'),
    '^RLV':  ('United States', 'North America'),
    # ── Germany ──────────────────────────────────────────────────────────────
    '^GDAXI':  ('Germany', 'Europe'),
    '^MDAXI':  ('Germany', 'Europe'),
    '^SDAXI':  ('Germany', 'Europe'),
    '^TECDAX': ('Germany', 'Europe'),
    '^GDAXIP': ('Germany', 'Europe'),
    # ── Western & Northern Europe ────────────────────────────────────────────
    '^STOXX50E': ('Europe',         'Europe'),
    '^FTSE':     ('United Kingdom', 'Europe'),
    '^FTMC':     ('United Kingdom', 'Europe'),
    '^FCHI':     ('France',         'Europe'),
    '^AEX':      ('Netherlands',    'Europe'),
    '^IBEX':     ('Spain',          'Europe'),
    'FTSEMIB.MI':('Italy',          'Europe'),
    '^SSMI':     ('Switzerland',    'Europe'),
    '^ATX':      ('Austria',        'Europe'),
    '^BFX':      ('Belgium',        'Europe'),
    'BCEX.BR':   ('Belgium',        'Europe'),
    '^OMXS30':   ('Sweden',         'Europe'),
    '^OMXH25':   ('Finland',        'Europe'),
    '^OMXC20':   ('Denmark',        'Europe'),
    '^OSEAX':    ('Norway',         'Europe'),
    '^PSI20':    ('Portugal',       'Europe'),
    '^WIG20':    ('Poland',         'Europe'),
    '^ISEQ':     ('Ireland',        'Europe'),
    # ── Americas (ex-US) ─────────────────────────────────────────────────────
    '^GSPTSE': ('Canada',    'North America'),
    '^BVSP':   ('Brazil',    'South America'),
    '^MXX':    ('Mexico',    'North America'),
    '^MERV':   ('Argentina', 'South America'),
    '^IPSA':   ('Chile',     'South America'),
    '^COLCAP': ('Colombia',  'South America'),
    '^GSPCVP': ('United States', 'North America'),
    # ── Asia-Pacific ─────────────────────────────────────────────────────────
    '^N225':      ('Japan',       'Asia'),
    '^TPX':       ('Japan',       'Asia'),
    '000001.SS':  ('China',       'Asia'),
    '399001.SZ':  ('China',       'Asia'),
    '000300.SS':  ('China',       'Asia'),
    '^HSI':       ('Hong Kong',   'Asia'),
    '^HSCE':      ('Hong Kong',   'Asia'),
    '^TWII':      ('Taiwan',      'Asia'),
    '^TWOTCI':    ('Taiwan',      'Asia'),
    '^KS11':      ('South Korea', 'Asia'),
    '^KS200':     ('South Korea', 'Asia'),
    '^BSESN':     ('India',       'Asia'),
    '^NSEI':      ('India',       'Asia'),
    '^AXJO':      ('Australia',   'Oceania'),
    '^AORD':      ('Australia',   'Oceania'),
    '^NZ50':      ('New Zealand', 'Oceania'),
    '^STI':       ('Singapore',   'Asia'),
    '^JKSE':      ('Indonesia',   'Asia'),
    '^KLSE':      ('Malaysia',    'Asia'),
    '^SET.BK':    ('Thailand',    'Asia'),
    '^PSEI.MN':   ('Philippines', 'Asia'),
    '^VNINDEX.VN':('Vietnam',     'Asia'),
    '^CSE':       ('Sri Lanka',   'Asia'),
    '^KSE':       ('Pakistan',    'Asia'),
    # ── Middle East & Africa ─────────────────────────────────────────────────
    '^J203.JO':  ('South Africa',        'Africa'),
    '^TASI.SR':  ('Saudi Arabia',        'Asia'),
    '^TA125.TA': ('Israel',              'Asia'),
    '^QE':       ('Qatar',               'Asia'),
    '^DFMGI':    ('United Arab Emirates','Asia'),
    '^ADI':      ('United Arab Emirates','Asia'),
    '^EGX30':    ('Egypt',               'Africa'),
    '^KWSE':     ('Kuwait',              'Asia'),
    # ── US volatility / sector indices ───────────────────────────────────────
    '^VIX':  ('United States', 'North America'),
    '^VVIX': ('United States', 'North America'),
    '^VXN':  ('United States', 'North America'),
    '^RVX':  ('United States', 'North America'),
    '^VXEEM':('United States', 'North America'),
    '^VXD':  ('United States', 'North America'),
    '^VIX9D':('United States', 'North America'),
    '^VIX3M':('United States', 'North America'),
    '^VIX6M':('United States', 'North America'),
    '^VIX1Y':('United States', 'North America'),
    '^VNDX': ('United States', 'North America'),
    '^MOVE': ('United States', 'North America'),
    '^EVZ':  ('United States', 'North America'),
    '^OVX':  ('United States', 'North America'),
    '^GVZ':  ('United States', 'North America'),
    '^VIFX': ('United States', 'North America'),
    '^VXGDX':('United States', 'North America'),
    '^VXSLV':('United States', 'North America'),
    '^SKEW': ('United States', 'North America'),
    '^V2TX': ('Europe',        'Europe'),
    '^SOX':  ('United States', 'North America'),
    '^XAU':  ('United States', 'North America'),
    '^HGX':  ('United States', 'North America'),
    '^OSX':  ('United States', 'North America'),
    '^UTY':  ('United States', 'North America'),
    '^TRAN': ('United States', 'North America'),
    '^XBD':  ('United States', 'North America'),
    '^IXBK': ('United States', 'North America'),
    '^IXFN': ('United States', 'North America'),
    '^IXTR': ('United States', 'North America'),
    '^IXHC': ('United States', 'North America'),
    '^IXCO': ('United States', 'North America'),
    '^DRG':  ('United States', 'North America'),
    # ── US Treasury yield indices ─────────────────────────────────────────────
    '^IRX':  ('United States', 'North America'),
    '^FVX':  ('United States', 'North America'),
    '^TNX':  ('United States', 'North America'),
    '^TYX':  ('United States', 'North America'),
    # ── Crypto ───────────────────────────────────────────────────────────────
    'BTC-USD': ('Global', 'Global'),
    'ETH-USD': ('Global', 'Global'),
    'USDT-USD':('Global', 'Global'),
}

def get_country_continent(ticker, default_country, default_continent):
    """Return (country, continent) for a ticker using direct map then suffix map."""
    if ticker in DIRECT_MAP:
        return DIRECT_MAP[ticker]
    dot = ticker.rfind('.')
    if dot > 0:
        suffix = ticker[dot:]
        if suffix in SUFFIX_MAP:
            return SUFFIX_MAP[suffix]
    return default_country, default_continent


def add_geo_columns(df, default_country, default_continent):
    """Add Country and Continent columns derived from the Ticker index level."""
    tickers = df.index.get_level_values('Ticker')
    df = df.copy()
    df['Country'] = [
        get_country_continent(t, default_country, default_continent)[0] for t in tickers
    ]
    df['Continent'] = [
        get_country_continent(t, default_country, default_continent)[1] for t in tickers
    ]
    return df


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
        # Single ticker -yfinance returns flat columns
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
        print('  [WARN] FRED_API_KEY not set -skipping macro.')
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


# ─── Market-cap backfill ──────────────────────────────────────────────────────

def fetch_current_mcaps(tickers: list) -> dict:
    """
    Fetch current market cap via Yahoo Finance's batch quote endpoint.
    Sends 100 tickers per request — no per-ticker sessions, no crumb issues.
    """
    CHUNK   = 100
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    URL     = 'https://query2.finance.yahoo.com/v8/finance/quote'

    chunks  = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    mcap_map = {}

    print(f'  Fetching market caps for {len(tickers)} tickers ({len(chunks)} batches) ...')
    for i, chunk in enumerate(chunks, 1):
        try:
            resp = requests.get(
                URL,
                params={'symbols': ','.join(chunk), 'fields': 'marketCap,symbol'},
                headers=HEADERS,
                timeout=30,
            )
            if resp.ok:
                for item in resp.json().get('quoteResponse', {}).get('result', []):
                    mcap = item.get('marketCap')
                    sym  = item.get('symbol')
                    if mcap and mcap > 0 and sym:
                        mcap_map[sym] = float(mcap)
            else:
                print(f'    [WARN] batch {i}: HTTP {resp.status_code}')
        except Exception as e:
            print(f'    [WARN] batch {i}: {e}')

        if i % 5 == 0 or i == len(chunks):
            print(f'    {min(i * CHUNK, len(tickers))}/{len(tickers)} done ...')

    print(f'  Received {len(mcap_map)}/{len(tickers)} market caps.')
    return mcap_map


def fetch_sector_industry(tickers: list) -> dict:
    """
    Fetch sector and industry for each ticker via Yahoo Finance batch endpoint.
    Returns {ticker: {'Sector': ..., 'Industry': ...}}.
    """
    CHUNK   = 100
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    URL     = 'https://query2.finance.yahoo.com/v8/finance/quote'

    chunks   = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    meta_map = {}

    print(f'  Fetching sector/industry for {len(tickers)} tickers ({len(chunks)} batches) ...')
    for i, chunk in enumerate(chunks, 1):
        try:
            resp = requests.get(
                URL,
                params={'symbols': ','.join(chunk), 'fields': 'sector,industry,symbol'},
                headers=HEADERS,
                timeout=30,
            )
            if resp.ok:
                for item in resp.json().get('quoteResponse', {}).get('result', []):
                    sym = item.get('symbol')
                    if sym:
                        meta_map[sym] = {
                            'Sector':   item.get('sector')   or None,
                            'Industry': item.get('industry') or None,
                        }
            else:
                print(f'    [WARN] batch {i}: HTTP {resp.status_code}')
        except Exception as e:
            print(f'    [WARN] batch {i}: {e}')

        if i % 5 == 0 or i == len(chunks):
            print(f'    {min(i * CHUNK, len(tickers))}/{len(tickers)} done ...')

    covered = sum(1 for v in meta_map.values() if v['Sector'] or v['Industry'])
    print(f'  Received sector/industry for {covered}/{len(tickers)} tickers.')
    return meta_map


def backfill_sector_industry(path: str, meta_map: dict) -> None:
    """
    Add / refresh Sector and Industry columns in an equity parquet file.
    Applied uniformly to all historical rows — sector rarely changes.
    """
    if not os.path.exists(path):
        return

    df = pd.read_parquet(path)
    if df.empty:
        return

    tickers        = df.index.get_level_values('Ticker')
    df             = df.copy()
    df['Sector']   = [meta_map.get(t, {}).get('Sector')   for t in tickers]
    df['Industry'] = [meta_map.get(t, {}).get('Industry') for t in tickers]
    df.to_parquet(path)

    unique  = tickers.unique()
    covered = sum(1 for t in unique if meta_map.get(t, {}).get('Sector'))
    print(f'  Sector/Industry: {covered}/{len(unique)} tickers -> {os.path.basename(path)}')


def backfill_market_cap(path: str, mcap_map: dict) -> None:
    """
    Add / refresh the Market_Cap column in an equity parquet file.

    Formula: historical_mcap(t) = close(t) / close_today × current_mcap
    Assumes shares outstanding is constant — valid as a first-order approximation.
    """
    if not os.path.exists(path):
        return

    df = pd.read_parquet(path)
    if df.empty or 'Close' not in df.columns:
        return

    close = df['Close'].unstack(level='Ticker')
    tickers = close.columns.tolist()

    current_mcaps = pd.Series({t: mcap_map.get(t) for t in tickers}, dtype=float)
    last_close = close.apply(
        lambda col: col.dropna().iloc[-1] if col.dropna().size else float('nan')
    )

    # scale[ticker] = current_mcap / last_close  →  multiply × historical close
    scale     = current_mcaps / last_close
    mcap_wide = close.multiply(scale, axis='columns')

    try:
        mcap_long = mcap_wide.stack(future_stack=True).rename('Market_Cap')
    except TypeError:
        mcap_long = mcap_wide.stack().rename('Market_Cap')
    mcap_long.index.names = ['Date', 'Ticker']

    df['Market_Cap'] = mcap_long.reindex(df.index)
    df.to_parquet(path)

    covered = int(current_mcaps.notna().sum())
    print(f'  Market_Cap: {covered}/{len(tickers)} tickers -> {os.path.basename(path)}')


# ─── Group updaters ───────────────────────────────────────────────────────────

def update_yfinance_group(
    tickers, filename,
    default_country='United States', default_continent='North America',
):
    tickers = to_list(tickers)
    path = os.path.join(DATABASE_DIR, filename)

    last = get_last_date(path)
    if last is not None:
        start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  Last date: {last.date()} ->fetching from {start}')
    else:
        start = START_DATE
        print(f'  No existing data ->fetching from {start}')

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Downloading {len(tickers)} tickers ...')
    new_df = fetch_yfinance(tickers, start, END_DATE)
    if new_df.empty:
        print('  No new data returned.')
        return

    new_df = add_geo_columns(new_df, default_country, default_continent)

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        # Backfill geo columns if the file predates this feature
        if 'Country' not in existing.columns:
            existing = add_geo_columns(existing, default_country, default_continent)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_df.sort_index()

    combined.to_parquet(path)
    print(f'  +{len(new_df)} rows ->{filename}')


def update_fred_group(tickers, filename):
    tickers = to_list(tickers)
    path = os.path.join(DATABASE_DIR, filename)

    last = get_last_date(path)
    if last is not None:
        start = (last + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  Last date: {last.date()} ->fetching from {start}')
    else:
        start = START_DATE
        print(f'  No existing data ->fetching from {start}')

    if start > END_DATE:
        print('  Already up to date.')
        return

    print(f'  Fetching {len(tickers)} FRED series ...')
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
    print(f'  +{len(new_df)} rows ->{filename}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'=== Database Update -{END_DATE} ===\n')

    # (tickers, filename, default_country, default_continent)
    yf_groups = [
        (ticker_us,         'US_equities.parquet',   'United States',  'North America'),
        (ticker_asia,       'ASIA_equities.parquet',  'Asia',           'Asia'),
        (ticker_europe,     'EU_equities.parquet',    'Europe',         'Europe'),
        (ticker_de,         'DE_equities.parquet',    'Germany',        'Europe'),
        (ticker_rotw,       'ROTW_equities.parquet',  'Global',         'Global'),
        (ticker_bonds,      'bonds.parquet',           'United States',  'North America'),
        (ticker_commodities,'commodities.parquet',     'United States',  'North America'),
        (ticker_crypto,     'crypto.parquet',          'Global',         'Global'),
        (ticker_forex,      'forex.parquet',           'Global',         'Global'),
        (ticker_indices,    'indices.parquet',         'United States',  'North America'),
        (ticker_sectors,    'sectors.parquet',         'Global',         'Global'),
        (ticker_sentiment,  'sentiment.parquet',       'United States',  'North America'),
    ]

    labels = [
        'US equities', 'Asia equities', 'EU equities', 'DE equities',
        'ROTW equities', 'Bonds', 'Commodities', 'Crypto', 'Forex',
        'Indices', 'Sectors', 'Sentiment',
    ]

    for label, (tickers, filename, def_country, def_continent) in zip(labels, yf_groups):
        print(f'[{label}]')
        try:
            update_yfinance_group(tickers, filename, def_country, def_continent)
        except Exception as e:
            print(f'  [ERROR] {e}')

    print('\n[Macro (FRED)]')
    try:
        update_fred_group(ticker_macro, 'macro.parquet')
    except Exception as e:
        print(f'  [ERROR] {e}')

    equity_files = [
        (ticker_us,      'US_equities.parquet'),
        (ticker_asia,    'ASIA_equities.parquet'),
        (ticker_europe,  'EU_equities.parquet'),
        (ticker_de,      'DE_equities.parquet'),
        (ticker_rotw,    'ROTW_equities.parquet'),
    ]
    all_equity_tickers = list({t for tickers, _ in equity_files for t in to_list(tickers)})

    print('\n[Market Cap Backfill]')
    try:
        mcap_map = fetch_current_mcaps(all_equity_tickers)
        for tickers, filename in equity_files:
            path = os.path.join(DATABASE_DIR, filename)
            try:
                backfill_market_cap(path, mcap_map)
            except Exception as e:
                print(f'  [ERROR] {filename}: {e}')
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('\n[Sector / Industry Backfill]')
    try:
        meta_map = fetch_sector_industry(all_equity_tickers)
        for tickers, filename in equity_files:
            path = os.path.join(DATABASE_DIR, filename)
            try:
                backfill_sector_industry(path, meta_map)
            except Exception as e:
                print(f'  [ERROR] {filename}: {e}')
    except Exception as e:
        print(f'  [ERROR] {e}')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
