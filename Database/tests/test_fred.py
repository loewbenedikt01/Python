"""
FRED update rule, monthly OECD FX rates, the ARS peg and the no-USD rule. No network.
"""

from datetime import date

import duckdb
import pandas as pd
import pytest

from pipeline import fx, schema
from pipeline.fred import update_mode

T = pd.Timestamp


def test_update_mode():
    assert update_mode(None, T('2026-09-01'), 'M', has_obs=False) == 'full'           # new series
    assert update_mode(T('2026-09-01'), T('2026-09-01'), 'M', has_obs=True) == 'skip'  # unchanged
    assert update_mode(T('2026-09-01'), T('2026-09-20'), 'D', has_obs=True) == 'overlap'
    assert update_mode(T('2026-09-01'), T('2026-09-20'), 'M', has_obs=True) == 'full'  # revisions go back years
    assert update_mode(T('2026-09-01'), T('2026-09-20'), 'Q', has_obs=True) == 'full'
    assert update_mode(T('2026-09-01'), T('2026-09-20'), 'W', has_obs=True) == 'full'


def test_oecd_monthly_rate_on_every_business_day():
    obs = {'CCUSMA02PLM618N': pd.DataFrame({'date': [T('2000-01-01'), T('2000-02-01')], 'value': [4.0, 5.0]})}
    (src, s), = fx.fred_rates('PLN', obs)
    assert src == 'fred:CCUSMA02PLM618N monthly'
    assert s[T('2000-01-31')] == pytest.approx(0.25) and s[T('2000-02-15')] == pytest.approx(0.20)
    assert T('2000-01-15') not in s.index            # Saturday


def test_ars_peg_until_2002_then_market_rate():
    yahoo = pd.Series([0.99, 0.98], index=[T('2001-07-13'), T('2001-07-16')])
    df = fx.combine_rates('ARS', yahoo, 'yahoo:USDARS=X', fx.peg_rates('ARS'))
    df = df.set_index(pd.to_datetime(df['date']))
    assert df.loc[T('1996-10-08'), 'usd_per_unit'] == 1.0
    assert df.loc[T('1996-10-08'), 'source'].startswith('peg:ARS=USD')
    assert df.loc[T('2001-07-13'), 'usd_per_unit'] == 0.99        # market rate wins where it exists
    assert T('2002-01-08') not in df.index                          # peg ended 2002-01-06


def test_no_usd_for_yields_and_volatility():
    con = duckdb.connect()
    schema.create(con, 'prices')
    con.executemany('INSERT INTO instruments (ticker, name, currency) VALUES (?, ?, ?)', [
        ('^TNX', 'CBOE 10-Year Treasury Note Yield', 'USD'),
        ('^TNX_FRED', '10-Year Treasury Note Yield (FRED: DGS10)', 'USD'),
        ('^V2TX', 'Euro Stoxx 50 Volatility Index', 'EUR'),
        ('^GSPC', 'S&P 500 Index', 'USD'),
    ])
    con.executemany('INSERT INTO prices_daily (ticker, date, close, currency) VALUES (?, ?, ?, ?)',
                    [(t, date(2026, 9, 25), 4.2, c) for t, c in
                     [('^TNX', 'USD'), ('^TNX_FRED', 'USD'), ('^V2TX', 'EUR'), ('^GSPC', 'USD')]])
    con.execute("INSERT INTO fx_rates_daily VALUES ('EUR', DATE '2026-09-25', 1.1, 'test')")
    fx.apply_usd(con)
    got = dict(con.execute('SELECT ticker, close_usd FROM prices_daily').fetchall())
    assert got == {'^TNX': None, '^TNX_FRED': None, '^V2TX': None, '^GSPC': 4.2}
