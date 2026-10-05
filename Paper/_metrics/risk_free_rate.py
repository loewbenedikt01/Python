'''
Risk-Free Rate

Source: 3-month US Treasury bill
_database/risk_free_rate.parquet.

Conversion
  * discount rate d -> bond-equivalent (investment) yield
        y = 365 d / (360 - 91 d)
  * per trading day, as a log return:  ln(1 + y) / 252
  * the rate earned on day t is the last quote strictly before t: a bill
    bought at the close of t-1 earns that yield over day t.  No information
    from day t itself is used.

RISK_FREE (config.py):
    'tbill_3m'   the T-bill rate above (publication setting)
    'zero'       rf = 0 everywhere (thesis setting; robustness)
'''

from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    RISK_FREE, 
    RISK_FREE_PATH, 
    TRADING_DAYS_PER_YEAR, 
    MONTHS_PER_YEAR,
)

@lru_cache(maxsize=None)
def _yield() -> pd.Series:
    df = pd.read_parquet(RISK_FREE_PATH)
    d = pd.to_numeric(df['DTB3'], errors='coerce').dropna() / 100.0
    d.index = pd.to_datetime(d.index)
    y = 365.0 * d / (360.0 - 91.0 * d)
    y.index = y.index + pd.Timedelta(days=1)
    return y.sort_index()


def _annual_yield_at(idx: pd.DatetimeIndex) -> pd.Series:
    y = _yield().reindex(idx, method='ffill')
    if y.isna().any():
        bad = y.index[y.isna()]
        raise ValueError(f'[risk_free] no T-bill quote before {bad[0].date()} '
                         f'({len(bad)} date(s)); check {RISK_FREE_PATH}')
    return y


def rf_daily(index) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if RISK_FREE == 'zero':
        return pd.Series(0.0, index=idx, name='rf')
    if RISK_FREE != 'tbill_3m':
        raise ValueError(f'[risk_free] unknown RISK_FREE {RISK_FREE!r}; "tbill_3m" | "zero"')
    y = _annual_yield_at(idx)
    return pd.Series(np.log1p(y.to_numpy()) / TRADING_DAYS_PER_YEAR, index=idx, name='rf')


def rf_period(index, freq: str = 'D') -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if freq == 'D':
        return rf_daily(idx)
    if freq != 'M':
        raise ValueError(f'freq must be "D" or "M", got {freq!r}')
    if RISK_FREE == 'zero':
        return pd.Series(0.0, index=idx, name='rf')
    starts = idx.to_period('M').to_timestamp()
    y = _annual_yield_at(pd.DatetimeIndex(starts))
    return pd.Series(np.log1p(y.to_numpy()) / MONTHS_PER_YEAR, index=idx, name='rf')
