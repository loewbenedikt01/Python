"""
FX direction, FRED back-fill (incl. DEM for EUR), USD conversion for JPY /
GBp / EUR, forward-fill limit and the adj_close change check. No network.
"""

from datetime import date

import duckdb
import pandas as pd
import pytest

import config
from pipeline import fx, schema
from pipeline.prices import adj_close_changed


# ----
# PAIR DIRECTION
# ----

def test_pair_direction():
    assert fx.pair_currencies('EURUSD=X') == ('EUR', 'USD')      # USD per EUR
    assert fx.pair_currencies('JPY=X') == ('USD', 'JPY')         # JPY per USD
    assert fx.pair_currencies('USDJPY=X') == ('USD', 'JPY')
    assert fx.pair_currencies('AAPL') is None


def test_usd_per_unit():
    assert fx.usd_per_unit('EURUSD=X', 'EUR', 1.10) == pytest.approx(1.10)
    assert fx.usd_per_unit('JPY=X', 'JPY', 150.0) == pytest.approx(1 / 150)
    assert fx.usd_per_unit('USDJPY=X', 'JPY', 150.0) == pytest.approx(1 / 150)
    assert fx.usd_per_unit('GBPUSD=X', 'GBP', 1.35) == pytest.approx(1.35)
    assert fx.usd_per_unit('EURUSD=X', 'JPY', 1.10) is None     # wrong pair for the currency


def test_choose_pairs_prefers_direct_then_adds_helper():
    pairs = fx.choose_pairs({'EUR', 'JPY', 'TWD', 'USD'}, ['EURUSD=X', 'JPY=X'])
    assert pairs == {'EUR': 'EURUSD=X', 'JPY': 'JPY=X', 'TWD': 'TWD=X'}


def test_minor_units():
    assert fx.major_and_divisor('GBp') == ('GBP', 100)
    assert fx.major_and_divisor('EUR') == ('EUR', 1.0)


# ----
# FRED BACK-FILL
# ----

def test_fred_fills_before_yahoo_and_dem_before_1999():
    ts = pd.Timestamp
    yahoo = pd.Series([1.25, 1.26], index=[ts('2003-12-01'), ts('2003-12-02')])
    obs = {
        'DEXUSEU': pd.DataFrame({'date': [ts('1999-01-04'), ts('2003-12-01')], 'value': [1.18, 9.99]}),
        'EXGEUS':  pd.DataFrame({'date': [ts('1998-11-01'), ts('1998-12-01')], 'value': [1.67, 1.65]}),
    }
    df = fx.combine_rates('EUR', yahoo, 'yahoo:EURUSD=X', fx.fred_rates('EUR', obs))
    df = df.set_index(pd.to_datetime(df['date']))
    assert df.loc[ts('2003-12-01'), 'usd_per_unit'] == 1.25            # Yahoo wins where it exists
    assert df.loc[ts('1999-01-04'), 'source'] == 'fred:DEXUSEU'         # FRED EUR from 1999
    dec = df.loc[ts('1998-12-15')]                                       # monthly DEM on every business day
    assert dec['usd_per_unit'] == pytest.approx(1.95583 / 1.65)
    assert dec['source'] == 'fred:EXGEUS/DEM monthly'
    assert df.loc[ts('1998-11-30'), 'usd_per_unit'] == pytest.approx(1.95583 / 1.67)
    assert df.index.is_unique


def test_fred_fills_long_gaps_inside_yahoo_but_not_short_ones():
    ts = pd.Timestamp
    yahoo = pd.Series([7.8, 7.8, 7.8], index=[ts('2003-05-01'), ts('2003-05-05'), ts('2003-12-01')])
    fred_days = pd.bdate_range('2003-05-02', '2003-11-28')
    obs = {'DEXHKUS': pd.DataFrame({'date': fred_days, 'value': 7.79})}
    df = fx.combine_rates('HKD', yahoo, 'yahoo:USDHKD=X', fx.fred_rates('HKD', obs))
    fred_dates = set(pd.to_datetime(df.loc[df['source'] == 'fred:DEXHKUS', 'date']))
    assert ts('2003-05-02') not in fred_dates        # 1 day after Yahoo -> forward fill, no FRED
    assert ts('2003-05-09') not in fred_dates        # 4 days after the last Yahoo value (05-05) -> forward fill
    assert ts('2003-05-12') in fred_dates            # 7 days -> real gap, FRED fills
    assert ts('2003-11-28') in fred_dates


def test_fred_per_usd_series_is_inverted():
    obs = {'DEXJPUS': pd.DataFrame({'date': [pd.Timestamp('1995-01-03')], 'value': [100.0]})}
    (src, s), = fx.fred_rates('JPY', obs)
    assert s.iloc[0] == pytest.approx(0.01)


# ----
# USD COLUMNS
# ----

@pytest.fixture
def pcon():
    con = duckdb.connect()
    schema.create(con, 'prices')
    rows = [
        # ticker, date, close, currency
        ('7203.T', date(2026, 9, 25), 2989.5, 'JPY'),
        ('SHEL.L', date(2026, 9, 25), 3611.0, 'GBp'),
        ('SAP.DE', date(2026, 9, 25), 200.0, 'EUR'),
        ('SAP.DE', date(2026, 9, 29), 201.0, 'EUR'),     # FX 4 days old -> still used
        ('SAP.DE', date(2026, 10, 5), 202.0, 'EUR'),     # FX 10 days old -> no USD value
        ('AAPL', date(2026, 9, 25), 250.0, 'USD'),
    ]
    con.executemany('INSERT INTO prices_daily (ticker, date, close, adj_close, currency) VALUES (?, ?, ?, ?, ?)',
                    [(t, d, c, c, cur) for t, d, c, cur in rows])
    con.executemany('INSERT INTO fx_rates_daily VALUES (?, ?, ?, ?)', [
        ('JPY', date(2026, 9, 25), 1 / 150, 'test'),
        ('GBP', date(2026, 9, 25), 1.35, 'test'),
        ('EUR', date(2026, 9, 25), 1.10, 'test'),
    ])
    fx.apply_usd(con)
    return con


def usd(con, ticker, d):
    return con.execute('SELECT close_usd, fx_to_usd FROM prices_daily WHERE ticker = ? AND date = ?',
                       [ticker, d]).fetchone()


def test_usd_jpy(pcon):
    assert usd(pcon, '7203.T', date(2026, 9, 25))[0] == pytest.approx(2989.5 / 150)


def test_usd_gbp_pence(pcon):
    close_usd, fx_to_usd = usd(pcon, 'SHEL.L', date(2026, 9, 25))
    assert close_usd == pytest.approx(36.11 * 1.35)        # 3611 pence = 36.11 GBP
    assert fx_to_usd == pytest.approx(1.35 / 100)


def test_usd_eur_and_forward_fill_limit(pcon):
    assert usd(pcon, 'SAP.DE', date(2026, 9, 25))[0] == pytest.approx(220.0)
    assert usd(pcon, 'SAP.DE', date(2026, 9, 29))[0] == pytest.approx(201.0 * 1.10)
    assert usd(pcon, 'SAP.DE', date(2026, 10, 5)) == (None, None)
    assert config.FX_FFILL_LIMIT == 5


def test_usd_quoted_in_usd(pcon):
    assert usd(pcon, 'AAPL', date(2026, 9, 25)) == (250.0, 1.0)


# ----
# ADJ_CLOSE CHANGE CHECK
# ----

def test_adj_close_tolerance_and_partial_last_row():
    d = [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
    stored = pd.DataFrame({'date': d, 'adj_close': [100.0, 101.0, 102.0]})
    tiny = pd.DataFrame({'date': d, 'adj_close': [100.000001, 101.0, 105.0]})       # last row was partial
    assert not adj_close_changed(stored, tiny, before=d[-1])
    dividend = pd.DataFrame({'date': d, 'adj_close': [99.5, 100.5, 102.0]})          # 0.5% lower
    assert adj_close_changed(stored, dividend, before=d[-1])
