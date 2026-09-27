import os
import sys
import random

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Database._updating.update_database import DATABASE_DIR
from Database._tickers.equities import ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw


# ----
# PARAMETERS
# ----

REGISTRY_PATH = os.path.join(DATABASE_DIR, 'corporate_ids.csv')
ID_MIN, ID_MAX = 10000, 99999
REGISTRY_COLS  = ['corporate_id', 'Name', 'Ticker']


# ----
# REGISTRY
# ----

def load_registry(path=REGISTRY_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=REGISTRY_COLS).astype({'corporate_id': int})
    return pd.read_csv(path, dtype={'corporate_id': int, 'Name': str, 'Ticker': str})


def assign_corporate_ids(name_map, path=REGISTRY_PATH):
    """
    Give every ticker in the {ticker: name} dict a corporate_id. Existing ids
    are never changed; only new tickers get one. A new ticker reuses the id of
    a company with the same name, otherwise it gets a new random unused id.
    """
    registry = load_registry(path)
    known    = set(registry['Ticker'])
    by_name  = dict(zip(registry['Name'], registry['corporate_id']))
    used     = set(registry['corporate_id'])

    new_rows = []
    for ticker, name in name_map.items():
        if ticker in known:
            continue
        cid = by_name.get(name)
        if cid is None:
            if len(used) >= ID_MAX - ID_MIN + 1:
                raise RuntimeError('No free 5-digit corporate_ids left.')
            cid = random.randint(ID_MIN, ID_MAX)
            while cid in used:
                cid = random.randint(ID_MIN, ID_MAX)
            used.add(cid)
            by_name[name] = cid
        new_rows.append({'corporate_id': cid, 'Name': name, 'Ticker': ticker})

    if new_rows:
        registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
        registry = registry.sort_values(['Name', 'Ticker'])[REGISTRY_COLS]
        registry.to_csv(path, index=False)
    print(f'  corporate_ids.csv: {len(new_rows)} new tickers, '
          f'{registry["corporate_id"].nunique()} companies / {len(registry)} tickers total.')
    return registry


def get_corporate_id_map(path=REGISTRY_PATH):
    registry = load_registry(path)
    return dict(zip(registry['Ticker'], registry['corporate_id']))


def add_corporate_id(df, id_map=None):
    id_map  = id_map if id_map is not None else get_corporate_id_map()
    df      = df.copy()
    tickers = (df.index.get_level_values('Ticker') if 'Ticker' in df.index.names
               else df['Ticker'])
    df['corporate_id'] = pd.Series(tickers, index=df.index).map(id_map).astype('Int32')
    return df


# ----
# UPDATE
# ----

def update_corporate_ids(name_map, filenames=('equities_financial.parquet', 'equities_mapped.parquet')):
    """
    Assign ids to new tickers, then rewrite the corporate_id column in each file.
    """
    assign_corporate_ids(name_map)
    id_map = get_corporate_id_map()

    for filename in filenames:
        path = os.path.join(DATABASE_DIR, filename)
        if not os.path.exists(path):
            print(f'  {filename} not found — skipping corporate_id.')
            continue
        df = add_corporate_id(pd.read_parquet(path), id_map)
        df.to_parquet(path)
        missing = int(df['corporate_id'].isna().sum())
        print(f'  corporate_id added to {filename} ({missing} rows without id).')


if __name__ == '__main__':
    groups = (ticker_us, ticker_de, ticker_asia, ticker_europe, ticker_rotw)
    update_corporate_ids({t: n for group in groups for t, n in group.items()})
