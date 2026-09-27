"""
Checks against the real companies.duckdb (skipped if it has not been built yet).
"""

from datetime import date

import duckdb
import pytest

import config


def _con():
    if not config.COMPANIES_DB.exists():
        pytest.skip('companies.duckdb not built')
    return duckdb.connect(str(config.COMPANIES_DB), read_only=True)


def _yearly(ticker: str) -> dict:
    con = _con()
    try:
        rows = con.execute("""
            SELECT y.fiscal_year, y.period_end, y.revenue FROM fundamentals_yearly y
            JOIN company_info c USING (corporate_id) WHERE c.primary_ticker = ?""", [ticker]).fetchall()
    finally:
        con.close()
    if not rows:
        pytest.skip(f'no fundamentals for {ticker}')
    return {fy: (end, rev) for fy, end, rev in rows}


def test_aapl_revenue_continuous_across_tag_change():
    """
    SalesRevenueNet (FY2017) -> Revenues (FY2018) -> RevenueFromContractWithCustomer... (FY2019).
    """
    y = _yearly('AAPL')
    rev = [y[fy][1] for fy in (2016, 2017, 2018, 2019, 2020)]
    assert all(r is not None and r > 200e9 for r in rev)
    assert all(abs(b / a - 1) < 0.25 for a, b in zip(rev, rev[1:]))


def test_wmt_fiscal_year_ends_in_january():
    y = _yearly('WMT')
    assert y[2025][0] == date(2025, 1, 31)
    assert y[2026][0] == date(2026, 1, 31)
