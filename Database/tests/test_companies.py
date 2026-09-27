"""
ISIN checks, CUSIP derivation and primary-listing choice. No network.
"""

from pipeline.companies import choose_primary, cusip_from_isin, isin_plausible, major_currency, valid_isin


def test_valid_isin():
    assert valid_isin('US0378331005')          # Apple
    assert valid_isin('CNE1000062P8')          # BYD H-share
    assert not valid_isin('US0378331006')      # wrong check digit
    assert not valid_isin('NET00BYDSEM1')      # junk returned by yfinance for .SZ
    assert not valid_isin('-') and not valid_isin(None)


def test_isin_must_fit_listing_or_company_country():
    assert isin_plausible('AAPL', 'US0378331005', 'United States')
    assert not isin_plausible('GOOGL', 'CA02080M1005', 'United States')    # Canadian CDR
    assert isin_plausible('MDT', 'IE00BTN1Y115', 'Ireland')                  # incorporated abroad
    assert isin_plausible('1211.HK', 'CNE1000062P8', 'China')


def test_cusip_only_from_us_isin():
    assert cusip_from_isin('US0378331005') == '037833100'
    assert cusip_from_isin('CNE1000062P8') is None
    assert cusip_from_isin('bad') is None


def test_primary_listing():
    assert choose_primary(1, ['GOOGL', 'GOOG'], 'United States', is_sec=True) == 'GOOGL'
    assert choose_primary(2, ['1211.HK', '002594.SZ'], 'China', is_sec=False) == '002594.SZ'
    assert choose_primary(3, ['X.DE', 'X2.DE'], 'Germany', is_sec=False) == 'X.DE'
    assert choose_primary(4, ['A.L', 'A.DE'], 'France', is_sec=False) == 'A.L'     # no home listing -> first


def test_minor_currency_units():
    assert major_currency('GBp') == 'GBP' and major_currency('ZAc') == 'ZAR' and major_currency('usd') == 'USD'


def test_run_with_no_company_due_and_no_data_tickers_skipped(tmp_path, monkeypatch):
    """
    Every company refreshed within COMPANY_REFRESH_DAYS (crashed before: empty frame without columns),
    and a no_data instrument without currency must not be looked up (was one failure per day).
    """
    from datetime import datetime

    import duckdb
    import pandas as pd

    from pipeline import common, companies, schema

    dbs = {'companies': tmp_path / 'companies.duckdb', 'prices': tmp_path / 'prices.duckdb',
           'raw': tmp_path / 'raw.duckdb'}
    monkeypatch.setattr(common, 'DATABASES', dbs)
    c = duckdb.connect(str(dbs['companies']))
    schema.create(c, 'companies')
    c.execute("INSERT INTO company_info (corporate_id, name, updated_at) VALUES (12345, 'Fresh Inc.', ?)", [datetime.now()])
    c.close()
    p = duckdb.connect(str(dbs['prices']))
    schema.create(p, 'prices')
    p.execute("INSERT INTO instruments (ticker, asset_class, status) VALUES ('^TPX', 'indices', 'no_data')")
    p.close()

    registry = pd.DataFrame({'corporate_id': [12345], 'company_name': ['Fresh Inc.'], 'cik': [None],
                             'tickers': ['FRSH'], 'status': ['active']})
    monkeypatch.setattr(companies, 'load_registry', lambda: registry)
    monkeypatch.setattr(companies, 'review_name_groups', lambda: pd.DataFrame())

    def no_network(*a, **k):
        raise AssertionError('no download expected')
    for fn in ('fetch_yfinance', 'fetch_submissions', 'fetch_finnhub'):
        monkeypatch.setattr(companies, fn, no_network)

    result = companies.run(common.RunContext())
    assert result.failed == [] and result.updated == 0 and result.skipped == 1
