"""
Market cap across stock splits: synthetic data (no network) and the real
AAPL / 7203.T data (skipped if prices.duckdb is not built).
"""

from datetime import date

import duckdb
import pytest

import config
from pipeline import marketcap, schema


def make_db(tmp_path, prices, company, shares_rows):
    comp_path = tmp_path / 'companies.duckdb'
    c = duckdb.connect(str(comp_path))
    schema.create(c, 'companies')
    c.execute('INSERT INTO company_info (corporate_id, primary_ticker, sec_filer, yf_shares_outstanding) VALUES (?, ?, ?, ?)',
              company)
    for cid, fy, fq, period_end, as_of, shares in shares_rows:
        c.execute("""INSERT INTO fundamentals_quarterly (corporate_id, fiscal_year, fiscal_quarter, period_end,
                     shares_as_of, shares_outstanding) VALUES (?, ?, ?, ?, ?, ?)""",
                  [cid, fy, fq, period_end, as_of, shares])
    c.close()
    p = duckdb.connect()
    schema.create(p, 'prices')
    p.executemany("""INSERT INTO prices_daily (ticker, date, close, stock_splits, currency, fx_to_usd)
                     VALUES (?, ?, ?, ?, ?, ?)""", prices)
    p.execute(f"ATTACH '{comp_path}' AS c (READ_ONLY)")
    marketcap.build(p)
    return {d: (mc, sh, src) for d, mc, sh, src in p.execute(
        'SELECT date, market_cap, shares_outstanding, shares_source FROM market_cap_daily ORDER BY date').fetchall()}


def test_sec_shares_before_split_are_adjusted(tmp_path):
    """
    AAPL-like: 4:1 split on 2020-08-31, the last SEC count (4.275bn) is from before the split.
    """
    prices = [('AAPL', date(2020, 8, 27), 125.01, 0.0, 'USD', 1.0),
              ('AAPL', date(2020, 8, 28), 124.81, 0.0, 'USD', 1.0),
              ('AAPL', date(2020, 8, 31), 129.04, 4.0, 'USD', 1.0),
              ('AAPL', date(2020, 9, 1), 134.18, 0.0, 'USD', 1.0)]
    mc = make_db(tmp_path, prices, (1, 'AAPL', True, None),
                 [(1, 2020, 3, date(2020, 6, 27), date(2020, 7, 17), 4.275e9)])
    before, after = mc[date(2020, 8, 28)], mc[date(2020, 8, 31)]
    assert before[0] == pytest.approx(124.81 * 4 * 4.275e9)          # real price 499.24 x real shares
    assert after[0] == pytest.approx(129.04 * 4 * 4.275e9)
    assert after[0] / before[0] == pytest.approx(129.04 / 124.81)    # continuous: only the price move
    assert before[1] == pytest.approx(4.275e9) and after[1] == pytest.approx(17.1e9)
    assert before[2] == 'sec_cover_page'
    assert date(2020, 8, 27) in mc                                   # count as of 07-17 carried forward


def test_earliest_sec_count_backfilled_split_adjusted(tmp_path):
    """
    Before the first SEC count (2020-02-15, 1M shares) the count is extended backwards;
    across the 2:1 split in 2019 it stays continuous (real count 0.5M before the split).
    """
    prices = [('X', date(2019, 5, 1), 20.0, 0.0, 'USD', 1.0), ('X', date(2019, 6, 3), 10.2, 2.0, 'USD', 1.0),
              ('X', date(2020, 1, 2), 10.0, 0.0, 'USD', 1.0), ('X', date(2020, 3, 2), 11.0, 0.0, 'USD', 1.0)]
    mc = make_db(tmp_path, prices, (2, 'X', True, None), [(2, 2020, 1, date(2020, 1, 31), date(2020, 2, 15), 1e6)])
    assert mc[date(2020, 1, 2)][2] == 'sec_backfilled' and mc[date(2020, 3, 2)][2] == 'sec_cover_page'
    assert mc[date(2019, 5, 1)][1] == pytest.approx(0.5e6)          # real count before the split
    assert mc[date(2019, 5, 1)][0] == pytest.approx(20.0 * 1e6)     # close is split-adjusted: 40 real x 0.5M


def test_yfinance_current_shares_across_split(tmp_path):
    """
    Toyota-like: 5:1 split on 2021-09-29, yfinance shares are today's count.
    """
    prices = [('7203.T', date(2021, 9, 28), 2064.0, 0.0, 'JPY', 1 / 110),
              ('7203.T', date(2021, 9, 29), 2073.0, 5.0, 'JPY', 1 / 110)]
    mc = make_db(tmp_path, prices, (3, '7203.T', False, 11.84e9), [])
    before, after = mc[date(2021, 9, 28)], mc[date(2021, 9, 29)]
    assert after[0] / before[0] == pytest.approx(2073 / 2064)
    assert before[1] == pytest.approx(11.84e9 / 5) and after[1] == pytest.approx(11.84e9)
    assert after[2] == 'yfinance_current'


# ----
# REAL DATA
# ----

def _real(ticker, before, after):
    if not config.PRICES_DB.exists():
        pytest.skip('prices.duckdb not built')
    con = duckdb.connect(str(config.PRICES_DB), read_only=True)
    try:
        rows = dict(con.execute("""
            SELECT m.date, m.market_cap / p.close FROM market_cap_daily m
            JOIN prices_daily p ON p.ticker = m.primary_ticker AND p.date = m.date
            WHERE m.primary_ticker = ? AND m.date IN (?, ?)""", [ticker, before, after]).fetchall())
    finally:
        con.close()
    if len(rows) < 2:
        pytest.skip(f'no market cap for {ticker}')
    return rows


def test_aapl_continuous_across_2020_split():
    rows = _real('AAPL', date(2020, 8, 28), date(2020, 8, 31))
    # market cap / close = shares on today's basis; must not jump at the split
    assert rows[date(2020, 8, 31)] / rows[date(2020, 8, 28)] == pytest.approx(1.0, rel=0.02)


def test_toyota_continuous_across_2021_split():
    rows = _real('7203.T', date(2021, 9, 28), date(2021, 9, 29))
    assert rows[date(2021, 9, 29)] / rows[date(2021, 9, 28)] == pytest.approx(1.0, rel=0.02)


def test_glitch_share_counts_are_dropped(tmp_path):
    """
    Cover page tagged in thousands in one quarter (1,000x too small) -> ignored, previous count carried.
    """
    prices = [('G', date(2020, m, 15), 10.0, 0.0, 'USD', 1.0) for m in range(1, 13)]
    counts = [(4, 2020, q, date(2020, 3 * q - 2, 1), date(2020, 3 * q - 2, 1), 1e9 if q != 3 else 1e6)
              for q in (1, 2, 3, 4)]
    counts.append((4, 2021, 1, date(2020, 12, 1), date(2020, 12, 1), 1e9))
    mc = make_db(tmp_path, prices, (4, 'G', True, None), counts)
    assert mc[date(2020, 8, 15)][1] == pytest.approx(1e9)            # the 1e6 glitch (July) is ignored


def test_stale_sec_count_falls_back_to_yfinance(tmp_path):
    prices = [('S', date(2012, 1, 3), 10.0, 0.0, 'USD', 1.0), ('S', date(2026, 1, 5), 20.0, 0.0, 'USD', 1.0)]
    mc = make_db(tmp_path, prices, (5, 'S', True, 3e9), [(5, 2011, 4, date(2011, 12, 31), date(2011, 12, 31), 1e9)])
    assert mc[date(2012, 1, 3)][2] == 'sec_balance_sheet' and mc[date(2012, 1, 3)][1] == pytest.approx(1e9)
    assert mc[date(2026, 1, 5)][2] == 'yfinance_current' and mc[date(2026, 1, 5)][1] == pytest.approx(3e9)


# ----
# Spin-offs recorded by Yahoo as splits (corporate_actions). SEC counts as in the real data.
# ----

SEC = [  # corporate_id, as-of date, shares
    (10, '2025-12-31', 284741809), (10, '2026-03-31', 275540427),                       # BDX
    (20, '2026-01-30', 298800000), (20, '2026-04-24', 296000000),                       # SPGI (no count after)
    (30, '2025-09-30', 634887208), (30, '2026-01-23', 635675701),                       # HON
    (30, '2026-03-31', 633653119), (30, '2026-06-30', 316940010),
    (40, '2026-05-01', 409921306), (40, '2026-07-31', 135042975),                       # DD
]
EVENTS = [('BDX', '2026-02-10', 10, 1.272), ('SPGI', '2026-07-01', 20, 1.057), ('HON', '2025-10-30', 30, 1.061),
          ('HON', '2026-06-29', 30, 0.9535), ('DD', '2026-06-24', 40, 1 / 3)]


def classified(overrides=None):
    import pandas as pd
    ev = pd.DataFrame(EVENTS, columns=['ticker', 'date', 'corporate_id', 'yahoo_factor'])
    sec = pd.DataFrame(SEC, columns=['corporate_id', 'd', 'shares'])
    yf = {20: (294800000, date(2026, 9, 26))}
    out = marketcap.classify_events(ev, sec, yf, overrides=overrides or {}, max_days=120)
    return {(r.ticker, str(r.date)): r for r in out.itertuples()}


def test_clean_ratio():
    assert marketcap.clean_ratio(0.5002, 0.03) == 0.5 and marketcap.clean_ratio(1.001, 0.03) == 1.0
    assert marketcap.clean_ratio(0.3294, 0.03) == pytest.approx(1 / 3) and marketcap.clean_ratio(1.49, 0.03) == 1.5
    assert marketcap.clean_ratio(0.9677, 0.03) is None                    # BDX: 3.2 % off 1:1


def test_spin_off_only_hon_2025():
    r = classified()[('HON', '2025-10-30')]
    assert (r.split_factor, r.method) == (1.0, 'sec_counts') and r.price_only_factor == pytest.approx(1.061)


def test_spin_off_only_spgi_via_yfinance_count():
    r = classified()[('SPGI', '2026-07-01')]
    assert (r.split_factor, r.method) == (1.0, 'yfinance_after') and not r.review


def test_spin_off_bdx_goes_to_review_as_price_only():
    r = classified()[('BDX', '2026-02-10')]
    assert (r.split_factor, r.method, r.review) == (1.0, 'price_only_review', True)
    assert r.price_only_factor == pytest.approx(1.272)
    ok = classified({('BDX', '2026-02-10'): 1.0})[('BDX', '2026-02-10')]            # after review
    assert (ok.method, ok.review) == ('override', False)


def test_reverse_split_plus_spin_off_hon_2026():
    r = classified()[('HON', '2026-06-29')]
    assert r.split_factor == 0.5 and r.price_only_factor == pytest.approx(1.907)    # 1:2 reverse split + spin-off


def test_real_reverse_split_dd_2026():
    r = classified()[('DD', '2026-06-24')]
    assert r.split_factor == pytest.approx(1 / 3) and r.price_only_factor == pytest.approx(1.0)


def test_counts_too_far_from_event_go_to_review():
    import pandas as pd
    ev = pd.DataFrame([('X', '2026-06-01', 1, 2.0)], columns=['ticker', 'date', 'corporate_id', 'yahoo_factor'])
    sec = pd.DataFrame([(1, '2025-12-31', 1e6), (1, '2026-09-30', 1.0e6)], columns=['corporate_id', 'd', 'shares'])
    r = marketcap.classify_events(ev, sec, {}, overrides={}, max_days=120).iloc[0]
    assert (r.split_factor, r.review) == (1.0, True)


def test_market_cap_across_combined_reverse_split_and_spin_off(tmp_path):
    """
    HON 2026 end to end: Yahoo 0.9535 = 1:2 reverse split x 1.907 spin-off. Shares halve (SEC confirms),
    market cap drops by the spun-off value only.
    """
    p = tmp_path
    prices = [('HON', date(2026, 6, 26), 243.534348, 0.0, 'USD', 1.0),
              ('HON', date(2026, 6, 29), 227.800003, 0.9535, 'USD', 1.0)]
    comp_path = p / 'companies.duckdb'
    c = duckdb.connect(str(comp_path))
    schema.create(c, 'companies')
    c.execute("INSERT INTO company_info (corporate_id, primary_ticker, sec_filer) VALUES (30, 'HON', TRUE)")
    for q, as_of, sh in [(1, "2026-03-31", 633653119), (2, "2026-06-30", 316940010)]:
        c.execute("""INSERT INTO fundamentals_quarterly (corporate_id, fiscal_year, fiscal_quarter, period_end,
                     shares_as_of, shares_outstanding) VALUES (30, 2026, ?, ?, ?, ?)""", [q, as_of, as_of, sh])
    c.close()
    px = duckdb.connect()
    schema.create(px, 'prices')
    px.execute("INSERT INTO instruments (ticker, corporate_id) VALUES ('HON', 30)")
    px.executemany('INSERT INTO prices_daily (ticker, date, close, stock_splits, currency, fx_to_usd) VALUES (?, ?, ?, ?, ?, ?)',
                   prices)
    px.execute(f"ATTACH '{comp_path}' AS c (READ_ONLY)")
    marketcap.build(px)
    mc = dict(px.execute('SELECT date, market_cap FROM market_cap_daily').fetchall())
    assert mc[date(2026, 6, 26)] == pytest.approx(243.534348 * 0.9535 * 633653119)   # real price x real shares
    assert mc[date(2026, 6, 29)] == pytest.approx(227.800003 * 316940010, rel=1e-3)


def test_far_counts_confirming_yahoos_clean_split_are_a_split():
    import pandas as pd
    ev = pd.DataFrame([('CSX', '2011-06-16', 1, 3.0), ('X', '2020-06-01', 2, 2.0)],
                      columns=['ticker', 'date', 'corporate_id', 'yahoo_factor'])
    sec = pd.DataFrame([(1, '2010-09-24', 1.2e9), (1, '2011-07-01', 3.5125e9),       # ratio 2.927 -> 3:1
                        (2, '2019-01-01', 1e6), (2, '2021-01-01', 1.7e6)],             # 1.7: doesn't confirm 2:1
                       columns=['corporate_id', 'd', 'shares'])
    out = marketcap.classify_events(ev, sec, {}, overrides={}, max_days=120).set_index('ticker')
    assert out.loc['CSX', 'split_factor'] == 3.0 and not out.loc['CSX', 'review']
    assert out.loc['X', 'split_factor'] == 1.0 and out.loc['X', 'review']


def test_restated_balance_sheet_count_after_split_googl_2014():
    """
    GOOGL: Class C dividend (2:1) on 2014-04-03. The 2014-03-31 balance-sheet count (674.5M) comes from the
    10-Q filed 2014-04-24, already restated post-split; 2013-12-31 (335.8M) was filed before the event.
    The SEC counts classify the event as 2:1 by themselves.
    """
    import pandas as pd
    ev = pd.DataFrame([('GOOGL', '2014-04-03', 1, 1.998)], columns=['ticker', 'date', 'corporate_id', 'yahoo_factor'])
    sec = pd.DataFrame([(1, '2013-12-31', 335.832e6, '2014-02-12'), (1, '2014-03-31', 674.462e6, '2014-04-24')],
                       columns=['corporate_id', 'd', 'shares', 'basis'])
    r = marketcap.classify_events(ev, sec, {}, overrides={}, max_days=120).iloc[0]
    assert (r.split_factor, r.method) == (2.0, 'sec_counts')


def test_daily_jump_check_skips_corporate_actions(tmp_path, monkeypatch):
    """
    A 2:1 split day (Yahoo split, SEC counts confirm it) is not flagged; a price glitch day (+50 %) is.
    """
    comp_path = tmp_path / 'companies.duckdb'
    c = duckdb.connect(str(comp_path))
    schema.create(c, 'companies')
    c.execute("INSERT INTO company_info (corporate_id, primary_ticker, sec_filer, yf_shares_outstanding) VALUES (7, 'X', TRUE, 2e6)")
    for q, as_of, sh in [(1, '2020-03-31', 1e6), (2, '2020-06-30', 2e6)]:
        c.execute("""INSERT INTO fundamentals_quarterly (corporate_id, fiscal_year, fiscal_quarter, period_end,
                     shares_as_of, shares_outstanding) VALUES (7, 2020, ?, ?, ?, ?)""", [q, as_of, as_of, sh])
    c.close()
    p = duckdb.connect()
    schema.create(p, 'prices')
    p.execute("INSERT INTO instruments (ticker, corporate_id) VALUES ('X', 7)")
    prices = [('X', date(2020, 5, 1), 50.0, 0.0), ('X', date(2020, 5, 4), 50.5, 2.0),     # split: close already adjusted
              ('X', date(2020, 5, 5), 51.0, 0.0), ('X', date(2020, 5, 6), 76.5, 0.0),      # glitch +50 %
              ('X', date(2020, 5, 7), 51.0, 0.0)]
    p.executemany("INSERT INTO prices_daily (ticker, date, close, stock_splits, currency, fx_to_usd) VALUES (?, ?, ?, ?, 'USD', 1.0)",
                  prices)
    p.execute(f"ATTACH '{comp_path}' AS c (READ_ONLY)")
    marketcap.build(p)
    monkeypatch.setattr(marketcap, 'CHECKS_CSV', tmp_path / 'checks.csv')            # not the real review file
    out = marketcap.run_checks(p)
    jumps = out[out['check'] == 'daily_jump']
    assert sorted(jumps['date'].astype(str)) == ['2020-05-06', '2020-05-07']          # up and back down
    assert set(jumps['driver']) == {'price'}


def test_one_off_share_count_blip_is_dropped():
    import pandas as pd
    ev = pd.DataFrame({'corporate_id': [1] * 4 + [2] * 3,
                       'd': pd.to_datetime(['2023-02-10', '2023-04-21', '2023-07-21', '2023-10-20',
                                            '2020-01-01', '2020-04-01', '2020-07-01']),
                       'shares_today': [3.79e9, 3.75e9, 1.82e9, 3.63e9,          # WFC: blip, back next quarter
                                        1.0e9, 1.4e9, 1.41e9]})                  # merger: stays up
    out = marketcap.clean_sec_shares(ev)
    assert list(out.loc[out['corporate_id'] == 1, 'shares_today']) == [3.79e9, 3.75e9, 3.63e9]
    assert len(out[out['corporate_id'] == 2]) == 3
