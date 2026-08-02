import os
import pandas as pd
from config import DATABASE_DIR, PARQUET_FILES, START_DATA, END_DATE

def load_all() -> dict[str, pd.DataFrame]:
    """
    Returns a dict of clean DataFrames keyed by asset class.
    All DataFrames are clipped to [START_DATA, END_DATE].
    Continent/Country/... metadata is joined onto df_equities from equities_mapping.
    Files not yet produced (see Database_Update/update_database.py) are skipped.
    """
    raw = {}
    for key, fname in PARQUET_FILES.items():
        path = os.path.join(DATABASE_DIR, fname)
        if os.path.exists(path):
            raw[key] = pd.read_parquet(path)
        else:
            print(f'[data] {fname} not found — skipping {key!r}.')

    # Attach per-ticker metadata (Continent, Country, ...) to the equities frame
    mapping = raw.pop('equities_mapping', None)
    if mapping is not None and 'equities' in raw:
        raw['equities'] = raw['equities'].join(mapping[['Continent', 'Country']], on='Ticker')

    # Clip all to date range
    for key, df in raw.items():
        if isinstance(df.index, pd.MultiIndex):
            raw[key] = df.loc[pd.IndexSlice[START_DATA:END_DATE, :], :]
        else:
            raw[key] = df.loc[START_DATA:END_DATE]

    return raw