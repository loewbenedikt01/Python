import os
import sys
import time
import concurrent.futures as cf

import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from update_database import DATABASE_DIR, get_currency, get_country_continent
from ticker_files.equities import ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw

MAPPING_FILENAME = 'equities_mapping.parquet'
MAX_WORKERS = 10
RETRIES = 3


def _fetch_sector_industry(ticker):
    """One-off, slow: yfinance has no bulk endpoint for this, so it's a
    single web request per ticker. Retried with backoff on failure."""
    for attempt in range(RETRIES):
        try:
            info = yf.Ticker(ticker).info
            return ticker, info.get('sector'), info.get('industry')
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return ticker, None, None


def build_equities_mapping(name_map, existing=None):
    """Build the Ticker -> metadata table. Sector/Industry come from
    yfinance and rarely change, so tickers already present in `existing`
    with a resolved Sector/Industry are reused instead of refetched — only
    genuinely new tickers, and tickers that previously came back empty
    (e.g. rate-limited or unsupported by yfinance for that exchange), hit
    the network."""
    known = set(existing.index) if existing is not None else set()

    sector_industry = {}
    if existing is not None:
        for t in known & set(name_map):
            sector_industry[t] = (existing.at[t, 'Sector'], existing.at[t, 'Industry'])

    to_fetch = [
        t for t in name_map
        if t not in known or pd.isna(sector_industry[t][0])
    ]

    if to_fetch:
        print(f'  Fetching sector/industry for {len(to_fetch)} new/missing tickers ...')
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, (ticker, sector, industry) in enumerate(ex.map(_fetch_sector_industry, to_fetch), 1):
                sector_industry[ticker] = (sector, industry)
                if i % 100 == 0:
                    print(f'    ...{i}/{len(to_fetch)}')

    rows = []
    for ticker, name in name_map.items():
        country, continent = get_country_continent(ticker)
        currency = get_currency(ticker)
        sector, industry = sector_industry.get(ticker, (None, None))
        rows.append({
            'Ticker': ticker,
            'Name': name,
            'Country': country,
            'Continent': continent,
            'Currency': currency,
            'Sector': sector,
            'Industry': industry,
        })

    return pd.DataFrame(rows).set_index('Ticker').sort_index()


def update_equities_mapping(filename=MAPPING_FILENAME):
    combined_map = {}
    for group in (ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw):
        combined_map.update(group)

    path = os.path.join(DATABASE_DIR, filename)
    existing = pd.read_parquet(path) if os.path.exists(path) else None

    df = build_equities_mapping(combined_map, existing=existing)
    df.to_parquet(path)

    missing = int(df['Sector'].isna().sum())
    print(f'  {filename}: {len(df)} tickers saved ({missing} missing sector/industry).')


if __name__ == '__main__':
    update_equities_mapping()
