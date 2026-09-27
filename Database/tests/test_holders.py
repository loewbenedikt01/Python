"""
13F rules with made-up filings: amendments, value units, report periods, CUSIP check digit.
"""

from datetime import date

import pandas as pd
import pytest

from pipeline.holders import cusip_valid, holdings


def write(folder, submissions, cover, info):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(submissions, columns=['accession', 'filing_date', 'submission_type', 'filer_cik', 'period']) \
        .to_parquet(folder / 'submission.parquet')
    pd.DataFrame(cover, columns=['accession', 'is_amendment', 'amendment_type', 'manager']) \
        .to_parquet(folder / 'coverpage.parquet')
    pd.DataFrame(info, columns=['accession', 'cusip', 'issuer', 'title', 'value', 'shares']) \
        .to_parquet(folder / 'infotable.parquet')


Q = date(2026, 6, 30)


def test_restatement_replaces_and_new_holdings_add(tmp_path):
    a = tmp_path / 'a'
    write(a,
          [('orig', date(2026, 8, 10), '13F-HR', '1', Q),
           ('restate', date(2026, 8, 20), '13F-HR/A', '1', Q),
           ('add', date(2026, 8, 25), '13F-HR/A', '1', Q)],
          [('orig', 'N', None, 'Fund A'), ('restate', 'Y', 'RESTATEMENT', 'Fund A'),
           ('add', 'Y', 'NEW HOLDINGS', 'Fund A')],
          [('orig', '037833100', 'APPLE INC', 'COM', 1000.0, 10.0),
           ('restate', '037833100', 'APPLE INC', 'COM', 1200.0, 12.0),
           ('add', '594918104', 'MICROSOFT CORP', 'COM', 500.0, 5.0)])
    h = holdings([a], [Q]).set_index('cusip')
    assert h.loc['037833100', 'shares'] == 12          # restatement replaced the original (not 10 + 12)
    assert h.loc['594918104', 'shares'] == 5           # new holdings added


def test_late_filing_in_later_window_and_thousands_before_2023(tmp_path):
    old_q = date(2022, 9, 30)
    a, b = tmp_path / 'a', tmp_path / 'b'
    write(a, [('x', date(2022, 11, 10), '13F-HR', '1', old_q)], [('x', 'N', None, 'Fund A')],
          [('x', '037833100', 'APPLE INC', 'COM', 150.0, 1.0)])                    # thousands
    write(b, [('y', date(2023, 2, 1), '13F-HR', '2', old_q)], [('y', 'N', None, 'Fund B')],   # late, next window
          [('y', '037833100', 'APPLE INC', 'COM', 150000.0, 1.0)])                 # dollars (filed after 2023-01-03)
    h = holdings([a, b], [old_q])
    assert len(h) == 2 and set(h['value_usd']) == {150000.0}


def test_other_periods_ignored(tmp_path):
    a = tmp_path / 'a'
    write(a, [('x', date(2026, 8, 1), '13F-HR', '1', date(2001, 9, 30))], [('x', 'N', None, 'Old')],
          [('x', '037833100', 'APPLE INC', 'COM', 1.0, 1.0)])
    assert holdings([a], [Q]).empty


def test_cusip_check_digit():
    assert cusip_valid('037833100') and cusip_valid('02079K305') and cusip_valid('G1151C101')
    assert not cusip_valid('00206R10R') and not cusip_valid('037833101X') and not cusip_valid('037833109')
