import os
import sys
import time
import concurrent.futures as cf

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Database._updating.update_database import DATABASE_DIR
from Database._updating.indexing.generate_id import assign_corporate_ids, get_corporate_id_map
from Database._tickers.equities import ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw


# ----
# PARAMETERS
# ----

CORPORATE_FILENAME = 'equities_mapped.parquet'
CORPORATE_COLS     = ['corporate_id', 'Name', 'Sector', 'Industry', 'Country', 'Continent', 'Currency']
MAX_WORKERS        = 10
RETRIES            = 3


# ----
# COUNTRY / CURRENCY MAPPING
# ----

# ticker suffix -> (country, continent, currency); no suffix = US listing
MAPPING = {
    '.T':  ('Japan',         'Asia',          'JPY'),
    '.HK': ('Hong Kong',     'Asia',          'HKD'),
    '.SS': ('China',         'Asia',          'CNY'),
    '.SZ': ('China',         'Asia',          'CNY'),
    '.NS': ('India',         'Asia',          'INR'),
    '.TW': ('Taiwan',        'Asia',          'TWD'),
    '.KS': ('South Korea',   'Asia',          'KRW'),
    '.SI': ('Singapore',     'Asia',          'SGD'),
    '.JK': ('Indonesia',     'Asia',          'IDR'),
    '.KL': ('Malaysia',      'Asia',          'MYR'),
    '.BK': ('Thailand',      'Asia',          'THB'),
    '.DE': ('Germany',       'Europe',        'EUR'),
    '.PA': ('France',        'Europe',        'EUR'),
    '.AS': ('Netherlands',   'Europe',        'EUR'),
    '.MI': ('Italy',         'Europe',        'EUR'),
    '.MC': ('Spain',         'Europe',        'EUR'),
    '.ST': ('Sweden',        'Europe',        'SEK'),
    '.CO': ('Denmark',       'Europe',        'DKK'),
    '.HE': ('Finland',       'Europe',        'EUR'),
    '.BR': ('Belgium',       'Europe',        'EUR'),
    '.VI': ('Austria',       'Europe',        'EUR'),
    '.LS': ('Portugal',      'Europe',        'EUR'),
    '.WA': ('Poland',        'Europe',        'PLN'),
    '.L':  ('United Kingdom','Europe',        'GBP'),
    '.SW': ('Switzerland',   'Europe',        'CHF'),
    '.TO': ('Canada',        'North America', 'CAD'),
    '.SA': ('Brazil',        'South America', 'BRL'),
    '.MX': ('Mexico',        'North America', 'MXN'),
    '.AX': ('Australia',     'Oceania',       'AUD'),
}


def country_mapping(ticker):
    dot = ticker.rfind('.')
    if dot > 0:
        country, continent, _ = MAPPING.get(ticker[dot:], ('Unknown', 'Unknown', None))
        return country, continent
    return 'United States', 'North America'


def currency_mapping(ticker):
    dot = ticker.rfind('.')
    if dot > 0:
        _, _, currency = MAPPING.get(ticker[dot:], (None, None, None))
        return currency or 'USD'
    return 'USD'


# ----
# SECTOR / INDUSTRY MAPPING
# ----

def _fetch_sector_industry(ticker):
    """
    One web request per ticker (yfinance has no bulk endpoint), retried with backoff.
    """
    for attempt in range(RETRIES):
        try:
            info = yf.Ticker(ticker).info
            return ticker, info.get('sector'), info.get('industry')
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return ticker, None, None


def sector_mapping(tickers, existing=None):
    """
    {ticker: (sector, industry)}. Sector/Industry rarely change, so tickers
    already resolved in `existing` are reused; only new tickers and ones that
    previously came back empty are fetched.
    """
    result = {}
    if existing is not None:
        for t in set(tickers) & set(existing.index):
            sector, industry = existing.at[t, 'Sector'], existing.at[t, 'Industry']
            if pd.notna(sector):
                result[t] = (sector, industry)

    to_fetch = [t for t in tickers if t not in result]
    if to_fetch:
        print(f'  Fetching sector/industry for {len(to_fetch)} tickers ...')
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for i, (ticker, sector, industry) in enumerate(ex.map(_fetch_sector_industry, to_fetch), 1):
                result[ticker] = (sector, industry)
                if i % 100 == 0:
                    print(f'    ...{i}/{len(to_fetch)}')
    return result


# ----
# UPDATE
# ----

def update_corporate_info(name_map, filename=CORPORATE_FILENAME):
    """
    Build the per-ticker info table (index Ticker):
    corporate_id, Name, Sector, Industry, Country, Continent, Currency.
    """
    path     = os.path.join(DATABASE_DIR, filename)
    existing = pd.read_parquet(path) if os.path.exists(path) else None

    assign_corporate_ids(name_map)
    id_map  = get_corporate_id_map()
    sectors = sector_mapping(list(name_map), existing)

    rows = []
    for ticker, name in name_map.items():
        country, continent = country_mapping(ticker)
        sector, industry   = sectors.get(ticker, (None, None))
        rows.append({
            'Ticker':       ticker,
            'corporate_id': id_map.get(ticker),
            'Name':         name,
            'Sector':       sector,
            'Industry':     industry,
            'Country':      country,
            'Continent':    continent,
            'Currency':     currency_mapping(ticker),
        })

    df = pd.DataFrame(rows).set_index('Ticker').sort_index()[CORPORATE_COLS]
    df['corporate_id'] = df['corporate_id'].astype('Int32')
    df.to_parquet(path)

    missing = int(df['Sector'].isna().sum())
    print(f'  {filename}: {len(df)} tickers saved ({missing} missing sector/industry).')
    return df


if __name__ == '__main__':
    groups = (ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw)
    update_corporate_info({t: n for group in groups for t, n in group.items()})
