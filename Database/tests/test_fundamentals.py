"""
XBRL facts -> wide quarterly / yearly rows, with made-up numbers. No network.
"""

import math

import pandas as pd
import pytest

from pipeline.fundamentals import add_derived, build_company

T = pd.Timestamp


def fact(tag, val, end, start=None, accn='a', fy=2019, fp='FY', form='10-K', unit='USD', filed='2020-02-01'):
    return {'tax': 'us-gaap', 'tag': tag, 'unit': unit, 'start': T(start) if start else pd.NaT, 'end': T(end),
            'val': float(val), 'accn': accn, 'fy': fy, 'fp': fp, 'form': form, 'filed': T(filed)}


def filing(accn, form, filed, report, cik=1):
    return {'accession_no': accn, 'form': form, 'filed_date': T(filed), 'report_date': T(report),
            'items': '', 'primary_doc': '', 'cik': cik}


def company():
    """
    Calendar-year company. FY2018 revenue tagged SalesRevenueNet, FY2019 the
    new ASC 606 tag. Cash flow only year-to-date in the 10-Qs. Each 10-Q also
    carries the prior-year comparative (must be ignored).
    """
    F, filings = [], []
    # FY2018 10-K (old revenue tag) + an amendment with a different number
    F += [fact('SalesRevenueNet', 400, '2018-12-31', '2018-01-01', 'k18', 2018, 'FY'),
          fact('NetCashProvidedByUsedInOperatingActivities', 60, '2018-12-31', '2018-01-01', 'k18', 2018, 'FY')]
    F += [fact('SalesRevenueNet', 999, '2018-12-31', '2018-01-01', 'k18a', 2018, 'FY', '10-K/A')]
    filings += [filing('k18', '10-K', '2019-02-01', '2018-12-31'), filing('k18a', '10-K/A', '2019-06-01', '2018-12-31')]

    rev_q = {1: 100, 2: 110, 3: 120}
    ocf_ytd = {1: 10, 2: 25, 3: 45}
    ends = {1: '2019-03-31', 2: '2019-06-30', 3: '2019-09-30'}
    starts_q = {1: '2019-01-01', 2: '2019-04-01', 3: '2019-07-01'}
    for q in (1, 2, 3):
        a, fp = f'q{q}', f'Q{q}'
        F += [fact('RevenueFromContractWithCustomerExcludingAssessedTax', rev_q[q], ends[q], starts_q[q], a, 2019, fp, '10-Q'),
              # prior-year comparative in the same filing, same fy/fp
              fact('RevenueFromContractWithCustomerExcludingAssessedTax', 1, ends[q].replace('2019', '2018'),
                   starts_q[q].replace('2019', '2018'), a, 2019, fp, '10-Q'),
              fact('NetCashProvidedByUsedInOperatingActivities', ocf_ytd[q], ends[q], '2019-01-01', a, 2019, fp, '10-Q'),
              fact('PaymentsToAcquirePropertyPlantAndEquipment', 2 * q, ends[q], '2019-01-01', a, 2019, fp, '10-Q'),
              fact('EarningsPerShareDiluted', 0.5, ends[q], starts_q[q], a, 2019, fp, '10-Q', 'USD/shares'),
              fact('CashAndCashEquivalentsAtCarryingValue', 50 + q, ends[q], None, a, 2019, fp, '10-Q')]
        filings.append(filing(a, '10-Q', f'2019-{3 * q + 1:02d}-15', ends[q]))
    # FY2019 10-K: 53-week year (371 days)
    F += [fact('RevenueFromContractWithCustomerExcludingAssessedTax', 460, '2020-01-05', '2018-12-31', 'k19', 2019, 'FY'),
          fact('NetCashProvidedByUsedInOperatingActivities', 70, '2020-01-05', '2018-12-31', 'k19', 2019, 'FY'),
          fact('PaymentsToAcquirePropertyPlantAndEquipment', 8, '2020-01-05', '2018-12-31', 'k19', 2019, 'FY'),
          fact('EarningsPerShareDiluted', 2.1, '2020-01-05', '2018-12-31', 'k19', 2019, 'FY', unit='USD/shares'),
          fact('CashAndCashEquivalentsAtCarryingValue', 60, '2020-01-05', None, 'k19', 2019, 'FY'),
          fact('Assets', 1000, '2020-01-05', None, 'k19', 2019, 'FY'),
          fact('EntityCommonStockSharesOutstanding', 300, '2020-01-25', None, 'k19', 2019, 'FY', unit='shares')]
    filings.append(filing('k19', '10-K', '2020-02-01', '2020-01-05'))
    for f in F:
        f['tax'] = 'dei' if f['tag'] == 'EntityCommonStockSharesOutstanding' else f['tax']
    return pd.DataFrame(F), pd.DataFrame(filings)


@pytest.fixture
def built():
    facts, filings = company()
    q, y = build_company(facts, filings)
    return q.set_index(['fiscal_year', 'fiscal_quarter']), y.set_index('fiscal_year')


def test_tag_chosen_per_period(built):
    _, y = built
    assert y.loc[2018, 'revenue'] == 400          # SalesRevenueNet
    assert y.loc[2019, 'revenue'] == 460          # RevenueFromContractWithCustomer...


def test_original_filing_wins_over_amendment(built):
    _, y = built
    assert y.loc[2018, 'accession_no'] == 'k18' and y.loc[2018, 'revenue'] == 400


def test_prior_year_comparatives_are_ignored(built):
    q, _ = built
    assert [q.loc[(2019, n), 'revenue'] for n in (1, 2, 3)] == [100, 110, 120]


def test_ytd_cash_flow_converted_to_quarters_and_q4(built):
    q, _ = built
    assert [q.loc[(2019, n), 'operating_cash_flow'] for n in (1, 2, 3, 4)] == [10, 15, 20, 25]
    assert q.loc[(2019, 4), 'revenue'] == 460 - 330
    assert bool(q.loc[(2019, 4), 'q4_derived']) and not bool(q.loc[(2019, 1), 'q4_derived'])


def test_per_share_empty_for_derived_q4(built):
    q, y = built
    assert q.loc[(2019, 3), 'eps_diluted'] == 0.5
    assert math.isnan(q.loc[(2019, 4), 'eps_diluted'])
    assert y.loc[2019, 'eps_diluted'] == 2.1


def test_53_week_year_and_balance_sheet_q4(built):
    q, y = built
    assert y.loc[2019, 'period_end'].isoformat() == '2020-01-05'
    assert q.loc[(2019, 4), 'cash'] == 60 and q.loc[(2019, 4), 'total_assets'] == 1000


def test_shares_outstanding_from_cover_page(built):
    _, y = built
    assert y.loc[2019, 'shares_outstanding'] == 300
    assert y.loc[2019, 'shares_as_of'].isoformat() == '2020-01-25'


def test_free_cash_flow_sign_and_missing_inputs(built):
    q, y = built
    assert y.loc[2019, 'free_cash_flow'] == 70 - 8              # capex stored positive
    assert math.isnan(y.loc[2018, 'free_cash_flow'])            # no capex in FY2018 -> empty, not 60
    assert math.isnan(y.loc[2019, 'gross_margin'])              # no gross profit -> empty, not 0


def test_derived_never_zero_when_input_missing():
    df = add_derived(pd.DataFrame([{'revenue': 0.0, 'gross_profit': 5.0, 'operating_income': 3.0}]))
    assert math.isnan(df.loc[0, 'gross_margin'])                # revenue 0 -> empty
    assert math.isnan(df.loc[0, 'ebitda'])                      # no D&A -> empty


def test_14_week_quarter_accepted():
    facts, filings = company()
    facts.loc[(facts['accn'] == 'q1') & facts['start'].notna() & (facts['tag'].str.startswith('Revenue')), 'start'] = \
        T('2018-12-24')                                         # 98 days
    q, _ = build_company(facts, filings)
    assert q.set_index(['fiscal_year', 'fiscal_quarter']).loc[(2019, 1), 'revenue'] == 100


def test_two_ciks_each_filing_once():
    facts, filings = company()
    old = filings[filings['accession_no'].isin(['k18', 'k18a'])].assign(cik=2)
    both_facts = pd.concat([facts, facts[facts['accn'] == 'k18']])          # k18 seen under both CIKs
    q, y = build_company(both_facts, pd.concat([filings, old]))
    assert list(y['fiscal_year']) == [2018, 2019] and y.set_index('fiscal_year').loc[2018, 'revenue'] == 400


def test_fiscal_year_from_filing_january_year_end():
    """
    WMT-like: fiscal year 2020 ends 2020-01-31 and is labelled fy 2020 by the filing.
    """
    F = [fact('Revenues', 520, '2020-01-31', '2019-02-01', 'w', 2020, 'FY')]
    q, y = build_company(pd.DataFrame(F), pd.DataFrame([filing('w', '10-K', '2020-03-20', '2020-01-31')]))
    assert list(y['fiscal_year']) == [2020] and y.iloc[0]['period_end'].isoformat() == '2020-01-31'


def test_largest_revenue_tag_in_group_wins():
    """
    TKR 2013: 'Revenues' = 80.1M (a partial line) next to SalesRevenueGoodsNet = 4.34bn.
    """
    F = [fact('Revenues', 80.1e6, '2013-12-31', '2013-01-01', 't', 2013, 'FY'),
         fact('SalesRevenueGoodsNet', 4341.2e6, '2013-12-31', '2013-01-01', 't', 2013, 'FY'),
         fact('RevenueFromContractWithCustomerIncludingAssessedTax', 9e9, '2013-12-31', '2013-01-01', 't', 2013, 'FY')]
    _, y = build_company(pd.DataFrame(F), pd.DataFrame([filing('t', '10-K', '2014-02-20', '2013-12-31')]))
    assert y.iloc[0]['revenue'] == 4341.2e6      # incl. sales tax tag is only a fallback


# ----
# excise, equity, gross profit, incomplete quarters
# ----

def _one_year(facts):
    F = [fact(tag, val, '2025-12-31', None if instant else '2025-01-01', 'x', 2025, 'FY')
         for tag, val, instant in facts]
    _, y = build_company(pd.DataFrame(F), pd.DataFrame([filing('x', '10-K', '2026-02-20', '2025-12-31')]))
    return y.iloc[0]


def test_excise_swapped_labels_tap():
    y = _one_year([('RevenueFromContractWithCustomerExcludingAssessedTax', 13.040, False),   # actually gross
                   ('RevenueFromContractWithCustomerIncludingAssessedTax', 11.141, False),   # actually net
                   ('CostOfGoodsAndServicesSold', 6.866, False), ('GrossProfit', 4.275, False)])
    assert y['revenue'] == pytest.approx(11.141) and y['revenue_incl_excise'] == pytest.approx(13.040)


def test_excise_gross_revenue_mo():
    y = _one_year([('RevenueFromContractWithCustomerExcludingAssessedTax', 23.279, False),
                   ('CostOfGoodsAndServicesSold', 5.597, False), ('OtherCostOfOperatingRevenue', 3.140, False),
                   ('GrossProfit', 14.542, False)])
    assert y['revenue'] == pytest.approx(20.139) and y['revenue_incl_excise'] == pytest.approx(23.279)


def test_excise_net_revenue_pm():
    y = _one_year([('RevenueFromContractWithCustomerExcludingAssessedTax', 40.648, False),
                   ('CostOfGoodsAndServicesSold', 13.366, False), ('ExciseAndSalesTaxes', 53.211, False),
                   ('GrossProfit', 27.282, False)])
    assert y['revenue'] == pytest.approx(40.648) and y['revenue_incl_excise'] == pytest.approx(93.859)


def test_no_excise_leaves_revenue_alone():
    y = _one_year([('Revenues', 100, False), ('CostOfRevenue', 60, False), ('GrossProfit', 40, False)])
    assert y['revenue'] == 100 and math.isnan(y['revenue_incl_excise'])


def test_parent_equity_and_nci():
    y = _one_year([('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 110, True),
                   ('StockholdersEquity', 100, True)])
    assert y['total_equity'] == 100 and y['noncontrolling_interest'] == 10
    y = _one_year([('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 110, True),
                   ('MinorityInterest', 10, True)])
    assert y['total_equity'] == 100


def test_gross_profit_reported_wins_and_derived_is_marked():
    y = _one_year([('Revenues', 100, False), ('CostOfRevenue', 60, False), ('GrossProfit', 38, False)])
    assert y['gross_profit'] == 38 and not y['gross_profit_derived']
    y = _one_year([('Revenues', 100, False), ('CostOfRevenue', 60, False)])
    assert y['gross_profit'] == 40 and y['gross_profit_derived']
    y = _one_year([('Revenues', 100, False)])
    assert math.isnan(y['gross_profit']) and not y['gross_profit_derived']


def test_q4_empty_when_quarters_incomplete():
    """
    First XBRL year: no Q1, Q2 without a 6-month YTD, Q3 without a 9-month YTD -> Q4 flows empty.
    """
    F = [fact('Revenues', 110, '2019-06-30', '2019-04-01', 'q2', 2019, 'Q2', '10-Q'),
         fact('Revenues', 120, '2019-09-30', '2019-07-01', 'q3', 2019, 'Q3', '10-Q'),
         fact('Revenues', 460, '2019-12-31', '2019-01-01', 'k', 2019, 'FY'),
         fact('Assets', 900, '2019-12-31', None, 'k', 2019, 'FY')]
    filings = pd.DataFrame([filing('q2', '10-Q', '2019-08-01', '2019-06-30'),
                            filing('q3', '10-Q', '2019-11-01', '2019-09-30'),
                            filing('k', '10-K', '2020-02-01', '2019-12-31')])
    q, y = build_company(pd.DataFrame(F), filings)
    q4 = q.set_index('fiscal_quarter').loc[4]
    assert math.isnan(q4['revenue']) and q4['total_assets'] == 900 and y.iloc[0]['revenue'] == 460


def test_q4_from_reported_nine_month_ytd_even_without_q1_q2():
    """
    If the Q3 10-Q reports the 9-month YTD, Q4 = FY - 9M is exact even when Q1/Q2 rows are missing.
    """
    F = [fact('Revenues', 120, '2019-09-30', '2019-07-01', 'q3', 2019, 'Q3', '10-Q'),
         fact('Revenues', 330, '2019-09-30', '2019-01-01', 'q3', 2019, 'Q3', '10-Q'),
         fact('Revenues', 460, '2019-12-31', '2019-01-01', 'k', 2019, 'FY')]
    filings = pd.DataFrame([filing('q3', '10-Q', '2019-11-01', '2019-09-30'),
                            filing('k', '10-K', '2020-02-01', '2019-12-31')])
    q, _ = build_company(pd.DataFrame(F), filings)
    assert q.set_index('fiscal_quarter').loc[4, 'revenue'] == 130


def test_share_count_in_thousands_is_fixed():
    def year(shares, eps):
        F = [fact('NetIncomeLoss', 1.0e9, '2025-12-31', '2025-01-01', 'x', 2025, 'FY'),
             fact('EarningsPerShareDiluted', eps, '2025-12-31', '2025-01-01', 'x', 2025, 'FY', unit='USD/shares'),
             fact('WeightedAverageNumberOfDilutedSharesOutstanding', shares, '2025-12-31', '2025-01-01', 'x', 2025,
                  'FY', unit='shares')]
        _, y = build_company(pd.DataFrame(F), pd.DataFrame([filing('x', '10-K', '2026-02-20', '2025-12-31')]))
        return y.iloc[0]
    y = year(500_000, 2.0)                                  # tagged in thousands, should be 500M
    assert y['shares_diluted_wavg'] == 5e8 and y['shares_scale_fixed']
    y = year(5e8, 2.1)                                      # 5 % off only: left as is (flagged by the check)
    assert y['shares_diluted_wavg'] == 5e8 and not y['shares_scale_fixed']


def test_time_of_day_from_edgar_acceptance():
    from pipeline.calendar import next_quarter, time_of_day
    assert time_of_day(pd.Timestamp('2026-07-30 20:30:28')) == (pd.Timestamp('2026-07-30').date(), 'amc')   # 16:30 EDT
    assert time_of_day(pd.Timestamp('2026-08-20 10:59:01')) == (pd.Timestamp('2026-08-20').date(), 'bmo')   # 06:59 EDT
    assert time_of_day(pd.Timestamp('2026-01-29 21:30:33'))[1] == 'amc'                                   # 16:30 EST
    assert time_of_day(pd.Timestamp('2026-03-10 15:00:00'))[1] == 'during'                                # 11:00 EDT
    assert time_of_day(pd.Timestamp('2026-02-03 23:10:00'))[0] == pd.Timestamp('2026-02-03').date()       # NY date
    assert next_quarter(2025, 4) == (2026, 1) and next_quarter(2026, 2) == (2026, 3)


def test_mislabelled_fiscal_year_and_fake_fy_filing():
    """
    STX: 10-K for June 2025 tagged fy 2027 -> 2025. SNOW-like: an 'FY' filing ending in April
    for a January year-end company is dropped.
    """
    F = [fact('Revenues', 10, '2024-06-28', '2023-07-01', 'k24', 2024, 'FY'),
         fact('Revenues', 11, '2025-06-27', '2024-06-29', 'k25', 2027, 'FY')]
    fl = pd.DataFrame([filing('k24', '10-K', '2024-08-01', '2024-06-28'), filing('k25', '10-K', '2025-08-01', '2025-06-27')])
    _, y = build_company(pd.DataFrame(F), fl)
    assert sorted(y['fiscal_year']) == [2024, 2025]
    F = [fact('Revenues', 100, '2022-01-31', '2021-02-01', 'a', 2022, 'FY'),
         fact('Revenues', 120, '2023-01-31', '2022-02-01', 'b', 2023, 'FY'),
         fact('Revenues', 30, '2022-04-30', '2021-05-01', 'c', 2023, 'FY')]
    fl = pd.DataFrame([filing('a', '10-K', '2022-03-20', '2022-01-31'), filing('b', '10-K', '2023-03-20', '2023-01-31'),
                       filing('c', '10-K', '2022-06-01', '2022-04-30')])
    _, y = build_company(pd.DataFrame(F), fl)
    assert y.set_index('fiscal_year')['period_end'].map(str).to_dict() == {2022: '2022-01-31', 2023: '2023-01-31'}
