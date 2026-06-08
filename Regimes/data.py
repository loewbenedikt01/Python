import os
import pandas as pd
from config import DATABASE_DIR, PARQUET_FILES, START_DATA, END_DATE

def load_all() -> dict[str, pd.DataFrame]:
    """
    Returns a dict of clean DataFrames keyed by asset class.
    All DataFrames are clipped to [START_DATA, END_DATE].
    Equities are pre-concatenated into df_equities.
    """
    raw = {
        key: pd.read_parquet(os.path.join(DATABASE_DIR, fname))
        for key, fname in PARQUET_FILES.items()
    }

    # Combine equity frames
    raw['equities'] = (
        pd.concat([raw['us'], raw['de'], raw['eu'], raw['asia'], raw['rotw']])
        .sort_index()
    )

    # Clip all to date range
    for key, df in raw.items():
        if isinstance(df.index, pd.MultiIndex):
            raw[key] = df.loc[pd.IndexSlice[START_DATA:END_DATE, :], :]
        else:
            raw[key] = df.loc[START_DATA:END_DATE]

    return raw