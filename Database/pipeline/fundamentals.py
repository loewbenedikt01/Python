"""
Step 'fundamentals': quarterly and yearly financials from SEC XBRL company facts
(companies.duckdb: fundamentals_quarterly, fundamentals_yearly, sec_filings).

Rules (see DECISIONS.md):
- Point-in-time: every value comes from the filing that first reported the
  period (10-K / 10-Q; an amendment only if the original is missing).
- Only facts whose period end is the filing's report period (+-7 days) are
  used; the prior-year comparatives in the same filing are ignored.
- The XBRL tag is chosen per period: for each item the first tag (or group
  of tags) of its list that has a value in that filing. Within a group (a
  tuple) the largest value wins: companies sometimes put a partial number
  under a generic tag (TKR 'Revenues' = $80M next to $4.3bn goods revenue).
- fiscal_year / fiscal quarter are the company's own (the filing's fy / fp).
- Durations: quarter 80-100 days (13/14 weeks), half year 170-195, nine
  months 260-290, year 350-380 days (52/53 weeks).
- Flows (income statement, cash flow): quarter value as reported, else
  YTD - previous YTD. Q4 = full year - nine months YTD (q4_derived).
  Per-share and weighted-share items only from reported quarters; empty for Q4.
- Derived values stay empty (never 0) when an input is missing.
- Incremental: a company is rebuilt only if SEC has a 10-K / 10-Q / 20-F /
  40-F / 8-K newer than the one seen at its last build, if BUILD_VERSION or
  its CIKs (SEC_PREDECESSOR_CIKS) changed, or while a 10-Q / 10-K from the
  last 400 days is missing from its rows (company facts lag behind EDGAR).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from pipeline import common, schema


log = common.get_logger('pipeline.fundamentals')

FACTS_URL       = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
SUBMISSIONS_URL = 'https://data.sec.gov/submissions/{name}'
SUBMISSIONS_DIR = config.CACHE_DIR / 'sec_submissions'
CHECKS_CSV      = config.REVIEW_DIR / 'fundamentals_checks.csv'

ANNUAL_FORMS    = {'10-K', '20-F', '40-F'}
QUARTER_FORMS   = {'10-Q'}
TRIGGER_FORMS   = {'10-K', '10-Q', '20-F', '40-F', '8-K'}            # + their amendments
KEEP_FORMS      = TRIGGER_FORMS | {'6-K', 'DEF 14A'}                  # stored in sec_filings
PERIOD_TOLERANCE_DAYS = 7
# bump when the extraction logic changes: every company is rebuilt on the next run
BUILD_VERSION = 8

# duration buckets in days
BUCKETS = {'Q': (80, 100), 'H': (170, 195), 'N': (260, 290), 'Y': (350, 380)}


# ----
# ITEMS -> XBRL TAGS (first tag with a value wins, per period)
# ----
# kind: flow = additive duration, per = per-share / weighted shares (not additive), stock = instant
# unit: money / per_share / shares

ITEMS = {
    # income statement
    'revenue': ('flow', 'money', [
        ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet',
         'SalesRevenueGoodsNet', 'RevenuesNetOfInterestExpense', 'Revenue', 'RevenueFromContractsWithCustomers',
         'OperatingLeaseLeaseIncome'),                       # REIT rental income (CPT)
        'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueServicesNet']),
    'cost_of_revenue': ('flow', 'money', [
        'CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold', 'CostOfServices',
        'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization', 'CostOfSales']),
    'gross_profit': ('flow', 'money', ['GrossProfit']),
    'rnd_expense': ('flow', 'money', [
        'ResearchAndDevelopmentExpense', 'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost']),
    'sga_expense': ('flow', 'money', ['SellingGeneralAndAdministrativeExpense']),
    'operating_expenses': ('flow', 'money', ['OperatingExpenses', 'CostsAndExpenses', 'NoninterestExpense']),
    'operating_income': ('flow', 'money', ['OperatingIncomeLoss', 'ProfitLossFromOperatingActivities']),
    'interest_expense': ('flow', 'money', [
        'InterestExpense', 'InterestExpenseNonoperating', 'InterestExpenseDebt', 'InterestAndDebtExpense',
        'FinanceCosts']),
    'pretax_income': ('flow', 'money', [
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'ProfitLossBeforeTax']),
    'income_tax': ('flow', 'money', ['IncomeTaxExpenseBenefit', 'IncomeTaxExpenseContinuingOperations']),
    'net_income': ('flow', 'money', ['NetIncomeLoss', 'ProfitLossAttributableToOwnersOfParent', 'ProfitLoss']),
    'net_income_to_common': ('flow', 'money', ['NetIncomeLossAvailableToCommonStockholdersDiluted',
                                               'NetIncomeLossAvailableToCommonStockholdersBasic']),
    'eps_basic': ('per', 'per_share', ['EarningsPerShareBasic', 'EarningsPerShareBasicAndDiluted',
                                        'BasicEarningsLossPerShare']),
    'eps_diluted': ('per', 'per_share', ['EarningsPerShareDiluted', 'EarningsPerShareBasicAndDiluted',
                                          'DilutedEarningsLossPerShare']),
    'shares_basic_wavg': ('per', 'shares', ['WeightedAverageNumberOfSharesOutstandingBasic']),
    'shares_diluted_wavg': ('per', 'shares', ['WeightedAverageNumberOfDilutedSharesOutstanding']),
    'depreciation_amortization': ('flow', 'money', [
        'DepreciationDepletionAndAmortization', 'DepreciationAmortizationAndAccretionNet',
        'DepreciationAndAmortization', 'Depreciation']),
    # balance sheet
    'cash': ('stock', 'money', ['CashAndCashEquivalentsAtCarryingValue', 'CashAndDueFromBanks', 'Cash',
                                 'CashAndCashEquivalents']),
    'short_term_investments': ('stock', 'money', [
        'ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesDebtSecuritiesCurrent']),
    'receivables': ('stock', 'money', ['AccountsReceivableNetCurrent', 'ReceivablesNetCurrent',
                                        'TradeAndOtherCurrentReceivables']),
    'inventory': ('stock', 'money', ['InventoryNet', 'Inventories']),
    'current_assets': ('stock', 'money', ['AssetsCurrent', 'CurrentAssets']),
    'ppe_net': ('stock', 'money', [
        'PropertyPlantAndEquipmentNet',
        'PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization',
        'PropertyPlantAndEquipment']),
    'goodwill': ('stock', 'money', ['Goodwill']),
    'intangibles': ('stock', 'money', ['IntangibleAssetsNetExcludingGoodwill', 'FiniteLivedIntangibleAssetsNet',
                                        'IntangibleAssetsOtherThanGoodwill']),
    'total_assets': ('stock', 'money', ['Assets']),
    'accounts_payable': ('stock', 'money', ['AccountsPayableCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent',
                                             'TradeAndOtherCurrentPayables']),
    'current_liabilities': ('stock', 'money', ['LiabilitiesCurrent', 'CurrentLiabilities']),
    'short_term_debt': ('stock', 'money', [
        'DebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent', 'LongTermDebtCurrent',
        'ShortTermBorrowings', 'CurrentBorrowings']),
    'long_term_debt': ('stock', 'money', [
        'LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations', 'NoncurrentBorrowings']),
    'total_liabilities': ('stock', 'money', ['Liabilities']),
    'total_equity': ('stock', 'money', ['StockholdersEquity', 'EquityAttributableToOwnersOfParent']),   # parent only
    'noncontrolling_interest': ('stock', 'money', ['MinorityInterest', 'EquityAttributableToNoncontrollingInterest']),
    'redeemable_equity': ('stock', 'money', [                       # mezzanine / temporary equity
        'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
        ('RedeemableNoncontrollingInterestEquityCarryingAmount', 'TemporaryEquityCarryingAmountAttributableToParent'),
        'TemporaryEquityCarryingAmount', 'TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest',
        ('RedeemableNoncontrollingInterestEquityFairValue', 'RedeemableNoncontrollingInterestEquityCommonCarryingAmount',
         'RedeemableNoncontrollingInterestEquityPreferredFairValue', 'RedeemableNoncontrollingInterestEquityOtherFairValue',
         'RedeemableNoncontrollingInterestEquityCommonFairValue'),
        'TemporaryEquityRedemptionValue', 'TemporaryEquityValueExcludingAdditionalPaidInCapital',
        'DebtInstrumentConvertibleCarryingAmountOfTheEquityComponent']),
    'retained_earnings': ('stock', 'money', ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings']),
    # cash flow (outflows as reported by the "Payments..." tags: positive numbers)
    'operating_cash_flow': ('flow', 'money', [
        'NetCashProvidedByUsedInOperatingActivities', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
        'CashFlowsFromUsedInOperatingActivities']),
    'capex': ('flow', 'money', ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets',
                                 'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities']),
    'investing_cash_flow': ('flow', 'money', [
        'NetCashProvidedByUsedInInvestingActivities', 'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations',
        'CashFlowsFromUsedInInvestingActivities']),
    'financing_cash_flow': ('flow', 'money', [
        'NetCashProvidedByUsedInFinancingActivities', 'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations',
        'CashFlowsFromUsedInFinancingActivities']),
    'dividends_paid': ('flow', 'money', ['PaymentsOfDividends', 'PaymentsOfDividendsCommonStock',
                                          'DividendsPaidClassifiedAsFinancingActivities']),
    'share_buybacks': ('flow', 'money', ['PaymentsForRepurchaseOfCommonStock', 'PaymentsForRepurchaseOfEquity']),
    'stock_based_comp': ('flow', 'money', ['ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense']),
    # internal helpers (not stored): excise / gross revenue and equity incl. NCI
    '_rev_excl_tax': ('flow', 'money', ['RevenueFromContractWithCustomerExcludingAssessedTax']),
    '_rev_incl_tax': ('flow', 'money', ['RevenueFromContractWithCustomerIncludingAssessedTax']),
    '_excise': ('flow', 'money', ['ExciseAndSalesTaxes']),
    '_other_cost': ('flow', 'money', ['OtherCostOfOperatingRevenue']),
    '_equity_incl_nci': ('stock', 'money', [
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'Equity']),
    # other
    'dividends_per_share': ('per', 'per_share', ['CommonStockDividendsPerShareDeclared',
                                                  'CommonStockDividendsPerShareCashPaid']),
}

COVER_SHARES_TAG   = 'EntityCommonStockSharesOutstanding'       # dei, cover page
BALANCE_SHARES_TAG = 'CommonStockSharesOutstanding'             # us-gaap, balance sheet date
EMPLOYEES_TAG      = 'EntityNumberOfEmployees'                  # dei, rarely tagged

INTERNAL = [k for k in ITEMS if k.startswith('_')]
DERIVED = ['ebitda', 'total_debt', 'net_debt', 'free_cash_flow', 'gross_margin', 'operating_margin', 'net_margin']
FIN_COLUMNS = schema.FINANCIAL_COLUMNS


def tag_map() -> pd.DataFrame:
    """
    One row per (tag, item) with its rank; tags in a tuple share a rank (largest value wins).
    """
    rows = []
    for item, (kind, unit, tags) in ITEMS.items():
        seen = set()
        for rank, entry in enumerate(tags):
            for tag in (entry if isinstance(entry, tuple) else (entry,)):
                if tag not in seen:
                    seen.add(tag)
                    rows.append({'tag': tag, 'item': item, 'kind': kind, 'unit_type': unit, 'rank': rank})
    return pd.DataFrame(rows)


# ----
# DOWNLOADS
# ----

def fetch_facts(cik: int) -> pd.DataFrame:
    """
    Company facts as rows: tax, tag, unit, start, end, val, accn, fy, fp, form, filed.
    Only the tags this module uses are kept.
    """
    data = common.http_get(FACTS_URL.format(cik=cik), 'sec').json()
    return flatten_facts(data)


def facts_fingerprint(facts: pd.DataFrame) -> str:
    """
    '<number of facts>|<newest filed date>' of the facts used; changes when SEC adds or amends facts.
    """
    newest = facts['filed'].max() if not facts.empty else None
    return f"{len(facts)}|{newest.date() if pd.notna(newest) else ''}"


def flatten_facts(data: dict) -> pd.DataFrame:
    wanted = set(tag_map()['tag']) | {COVER_SHARES_TAG, BALANCE_SHARES_TAG, EMPLOYEES_TAG}
    rows = []
    for tax, tags in (data.get('facts') or {}).items():
        for tag, d in tags.items():
            if tag not in wanted:
                continue
            for unit, values in d.get('units', {}).items():
                for v in values:
                    rows.append((tax, tag, unit, v.get('start'), v.get('end'), v.get('val'), v.get('accn'),
                                 v.get('fy'), v.get('fp'), v.get('form'), v.get('filed')))
    df = pd.DataFrame(rows, columns=['tax', 'tag', 'unit', 'start', 'end', 'val', 'accn', 'fy', 'fp', 'form', 'filed'])
    for c in ('start', 'end', 'filed'):
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df['val'] = pd.to_numeric(df['val'], errors='coerce')
    return df


def fetch_submissions(cik: int, all_pages: bool) -> pd.DataFrame:
    """
    Filings of a CIK (accession_no, form, filed_date, report_date, items, primary_doc).
    Older filings are on extra pages, fetched when `all_pages`.
    """
    data = common.http_get(SUBMISSIONS_URL.format(name=f'CIK{cik:010d}.json'), 'sec').json()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (SUBMISSIONS_DIR / f'CIK{cik:010d}.json').write_text(json.dumps(data), encoding='utf-8')
    pages = [data['filings']['recent']]
    if all_pages:
        for f in data['filings'].get('files', []):
            pages.append(common.http_get(SUBMISSIONS_URL.format(name=f['name']), 'sec').json())
    frames = []
    for p in pages:
        frames.append(pd.DataFrame({
            'accession_no': p.get('accessionNumber', []),
            'form':         p.get('form', []),
            'filed_date':   p.get('filingDate', []),
            'report_date':  p.get('reportDate', []),
            'items':        p.get('items', []),
            'primary_doc':  p.get('primaryDocument', []),
            'acceptance_time': p.get('acceptanceDateTime', []),
        }))
    df = pd.concat(frames, ignore_index=True)
    df['cik'] = cik
    df['filed_date'] = pd.to_datetime(df['filed_date'], errors='coerce')
    df['report_date'] = pd.to_datetime(df['report_date'].replace('', None), errors='coerce')
    # EDGAR acceptance time is UTC ('...Z'); stored without time zone
    df['acceptance_time'] = pd.to_datetime(df['acceptance_time'], errors='coerce', utc=True).dt.tz_localize(None)
    return df


# ----
# FILINGS -> VALUES
# ----

def base_form(form: str) -> str:
    return form[:-2] if isinstance(form, str) and form.endswith('/A') else form


def choose_filings(facts: pd.DataFrame, filings: pd.DataFrame, annual_forms=ANNUAL_FORMS) -> pd.DataFrame:
    """
    One filing per (fiscal_year, fiscal_period): the original, earliest filed;
    an amendment only if there is no original. fy / fp come from the facts
    (the filing's own DocumentFiscalYearFocus / DocumentFiscalPeriodFocus).
    """
    periodic = filings[filings['form'].map(base_form).isin(annual_forms | QUARTER_FORMS)].copy()
    meta = (facts.dropna(subset=['fy', 'fp']).groupby('accn')
                 .agg(fy=('fy', lambda s: int(s.mode().iloc[0])), fp=('fp', lambda s: s.mode().iloc[0]),
                      max_end=('end', 'max')))
    periodic = periodic.merge(meta, left_on='accession_no', right_index=True, how='inner')
    # report period: EDGAR's reportDate; fallback = latest period end in the filing's own facts
    periodic['period_end'] = periodic['report_date'].fillna(periodic['max_end'])
    periodic['is_amendment'] = periodic['form'].str.endswith('/A')
    periodic = periodic[periodic['fp'].isin(['Q1', 'Q2', 'Q3', 'FY'])]
    periodic = fix_fiscal_labels(periodic)
    periodic = periodic.sort_values(['fy', 'fp', 'is_amendment', 'filed_date'])
    return periodic.drop_duplicates(['fy', 'fp'], keep='first').reset_index(drop=True)


def fix_fiscal_labels(periodic: pd.DataFrame) -> pd.DataFrame:
    """
    Guard against mislabelled filings (rare, but they break the keys):
    - an 'FY' filing whose period end is more than a month away from the
      company's usual fiscal-year-end month is a quarter (SNOW, QRVO, DELL) -> dropped;
    - a fiscal year far from the period end (STX 10-K for June 2025 tagged 2027)
      -> period-end year + the company's usual offset (-1 / 0: some companies
      name the fiscal year after the year it starts).
    """
    if periodic.empty:
        return periodic
    periodic = periodic.copy()
    fy_rows = periodic[periodic['fp'] == 'FY']
    if fy_rows.empty:
        return periodic
    month = int(fy_rows['period_end'].dt.month.mode().iloc[0])
    gap = (periodic['period_end'].dt.month - month + 12) % 12
    wrong_fy = (periodic['fp'] == 'FY') & gap.isin(range(2, 11))        # > 1 month off the usual month
    periodic = periodic[~wrong_fy]
    offsets = (fy_rows['fy'] - fy_rows['period_end'].dt.year)
    offset = int(offsets[offsets.isin([-1, 0])].mode().iloc[0]) if offsets.isin([-1, 0]).any() else 0
    # fiscal year the period belongs to: the year of the next fiscal-year end, plus the offset
    fye_year = periodic['period_end'].dt.year + (periodic['period_end'].dt.month > month).astype(int)
    expected = fye_year + offset
    off = (periodic['fy'] - expected).abs() >= 1
    implausible = off & ((periodic['fy'] - periodic['period_end'].dt.year).abs() >= 2)
    periodic.loc[implausible, 'fy'] = expected[implausible]
    return periodic


def filing_values(facts: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """
    Long table accn, item, bucket ('I' for balance sheet, Q/H/N/Y for durations),
    val, start, unit. Per (filing, item, bucket) the highest-priority tag wins.
    """
    tm = tag_map()
    f = facts.merge(tm, on='tag').merge(
        chosen[['accession_no', 'period_end']], left_on='accn', right_on='accession_no')
    f = f[(f['end'] - f['period_end']).abs() <= pd.Timedelta(days=PERIOD_TOLERANCE_DAYS)]

    money = f['unit'].str.fullmatch(r'[A-Z]{3}')
    per_share = f['unit'].str.endswith('/shares')
    shares = f['unit'] == 'shares'
    f = f[((f['unit_type'] == 'money') & money) | ((f['unit_type'] == 'per_share') & per_share)
          | ((f['unit_type'] == 'shares') & shares)].copy()

    days = (f['end'] - f['start']).dt.days
    f['bucket'] = None
    f.loc[f['start'].isna() & (f['kind'] == 'stock'), 'bucket'] = 'I'
    for b, (lo, hi) in BUCKETS.items():
        f.loc[f['start'].notna() & (f['kind'] != 'stock') & days.between(lo, hi), 'bucket'] = b
    f = f.dropna(subset=['bucket'])
    f['dist'] = (f['end'] - f['period_end']).abs()
    f = f.sort_values(['accn', 'item', 'bucket', 'rank', 'dist', 'val'], ascending=[True, True, True, True, True, False])
    return f.drop_duplicates(['accn', 'item', 'bucket'], keep='first')[
        ['accn', 'item', 'bucket', 'val', 'start', 'end', 'unit', 'tag']]


def cover_shares(facts: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """
    Shares outstanding per filing: the cover page (dei) value, summed over share
    classes reported without dimensions, at its date. Fallback: balance-sheet
    CommonStockSharesOutstanding at the period end (multi-class companies tag the
    cover page per class, which company facts leave out).
    """
    out = []
    cover = facts[(facts['tag'] == COVER_SHARES_TAG) & (facts['unit'] == 'shares')]
    bal = facts[(facts['tag'] == BALANCE_SHARES_TAG) & (facts['unit'] == 'shares')]
    for accn, period_end in zip(chosen['accession_no'], chosen['period_end']):
        c = cover[cover['accn'] == accn]
        if not c.empty:
            last = c['end'].max()
            vals = c.loc[c['end'] == last, 'val'].drop_duplicates()
            out.append({'accn': accn, 'shares_outstanding': float(vals.sum()), 'shares_as_of': last})
            continue
        b = bal[(bal['accn'] == accn) & ((bal['end'] - period_end).abs() <= pd.Timedelta(days=PERIOD_TOLERANCE_DAYS))]
        if not b.empty:
            out.append({'accn': accn, 'shares_outstanding': float(b['val'].iloc[0]), 'shares_as_of': b['end'].iloc[0]})
    return pd.DataFrame(out, columns=['accn', 'shares_outstanding', 'shares_as_of'])


# ----
# ASSEMBLY
# ----

def _sub(a, b):
    return a - b if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)) else None


def _add(a, b):
    return a + b if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)) else None


def assemble(chosen: pd.DataFrame, values: pd.DataFrame, shares: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wide quarterly and yearly rows (without corporate_id) from the chosen filings.
    """
    v = {}
    for r in values.itertuples(index=False):
        v.setdefault(r.accn, {}).setdefault(r.item, {})[r.bucket] = r.val
    starts = {}
    for accn, g in values[values['bucket'].isin(['Q', 'Y'])].groupby('accn'):
        for b, gb in g.groupby('bucket'):
            starts[(accn, b)] = gb['start'].mode().iloc[0]
    currency = values[values['unit'].str.fullmatch(r'[A-Z]{3}')].groupby('accn')['unit'].agg(
        lambda s: 'USD' if 'USD' in set(s) else s.mode().iloc[0]).to_dict()
    sh = shares.set_index('accn').to_dict('index')

    def get(accn, item, bucket):
        return v.get(accn, {}).get(item, {}).get(bucket)

    def meta(f, fiscal_quarter=None, period_start=None):
        return {
            'fiscal_year': int(f.fy), **({'fiscal_quarter': fiscal_quarter} if fiscal_quarter else {}),
            'period_start': period_start, 'period_end': f.period_end.date(),
            'filed_date': f.filed_date.date(), 'form_type': f.form, 'accession_no': f.accession_no,
            'currency': currency.get(f.accession_no),
            'shares_outstanding': sh.get(f.accession_no, {}).get('shares_outstanding'),
            'shares_as_of': (sh[f.accession_no]['shares_as_of'].date() if f.accession_no in sh else None),
        }

    quarterly, yearly = [], []
    by_fy = {fy: {r.fp: r for r in g.itertuples(index=False)} for fy, g in chosen.groupby('fy')}
    for fy, fps in sorted(by_fy.items()):
        ytd = {}                                   # item -> YTD value as of the previous quarter
        prev_end = None
        for n, fp in enumerate(['Q1', 'Q2', 'Q3'], start=1):
            f = fps.get(fp)
            if f is None:
                ytd, prev_end = None, None         # a missing quarter breaks the YTD chain
                continue
            row = meta(f, n, _start(starts, f.accession_no, 'Q', prev_end))
            new_ytd = {}
            for item, (kind, _, _) in ITEMS.items():
                if kind == 'stock':
                    row[item] = get(f.accession_no, item, 'I')
                elif kind == 'per':
                    row[item] = get(f.accession_no, item, 'Q')
                else:
                    ytd_bucket = {1: 'Q', 2: 'H', 3: 'N'}[n]
                    q = get(f.accession_no, item, 'Q')
                    cum = get(f.accession_no, item, ytd_bucket)
                    prev = ytd.get(item) if ytd is not None else None
                    if q is None and n > 1:
                        q = _sub(cum, prev)
                    row[item] = q
                    new_ytd[item] = cum if cum is not None else (q if n == 1 else _add(prev, q))
            row['q4_derived'] = False
            quarterly.append(row)
            ytd, prev_end = new_ytd, f.period_end

        fy_filing = fps.get('FY')
        if fy_filing is None:
            continue
        y = meta(fy_filing, period_start=_start(starts, fy_filing.accession_no, 'Y', None))
        for item, (kind, _, _) in ITEMS.items():
            y[item] = get(fy_filing.accession_no, item, 'I' if kind == 'stock' else 'Y')
        yearly.append(y)

        # Q4 = full year - nine months YTD; balance sheet from the 10-K
        q4 = meta(fy_filing, 4, (prev_end + timedelta(days=1)).date() if prev_end is not None else None)
        for item, (kind, _, _) in ITEMS.items():
            if kind == 'stock':
                q4[item] = y[item]
            elif kind == 'per':
                q4[item] = None
            else:
                q4[item] = _sub(y[item], ytd.get(item)) if ytd is not None else None
        q4['q4_derived'] = True
        quarterly.append(q4)

    return pd.DataFrame(quarterly), pd.DataFrame(yearly)


def _start(starts, accn, bucket, prev_end):
    s = starts.get((accn, bucket))
    if s is not None and not pd.isna(s):
        return s.date()
    return (prev_end + timedelta(days=1)).date() if prev_end is not None else None


def _close(a, b, scale, tol=0.01):
    return pd.notna(a) and pd.notna(b) and pd.notna(scale) and scale != 0 and abs(a - b) <= tol * abs(scale)


def resolve_revenue_and_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per row:
    - revenue = net of excise taxes, revenue_incl_excise = gross, when it can be
      identified: both 'Excluding'- and 'IncludingAssessedTax' revenue tags ->
      the smaller is net (companies swap the labels, e.g. TAP); or a single
      revenue where revenue - cost_of_revenue - gross_profit equals the excise
      amount (revenue was gross, e.g. MO) or equals 0 (revenue was net, gross =
      revenue + ExciseAndSalesTaxes, e.g. PM). Otherwise revenue is unchanged.
    - total_equity (parent) and noncontrolling_interest completed from equity
      incl. NCI when one of them is missing.
    - gross_profit = revenue - cost_of_revenue only when not reported
      (gross_profit_derived = TRUE).
    """
    if df.empty:
        return df
    df = df.copy()
    for c in INTERNAL + ['revenue', 'cost_of_revenue', 'gross_profit', 'total_equity', 'noncontrolling_interest']:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    net, gross = [], []
    cols = zip(df['revenue'], df['_rev_excl_tax'], df['_rev_incl_tax'], df['cost_of_revenue'],
               df['gross_profit'], df['_excise'], df['_other_cost'])
    for rev, a, b, cogs, gp, excise_tag, other_cost in cols:
        n, g = rev, np.nan
        if pd.notna(a) and pd.notna(b) and a != b and pd.notna(rev) and rev in (a, b):
            n, g = min(a, b), max(a, b)
        elif pd.notna(rev) and pd.notna(cogs) and pd.notna(gp):
            resid = rev - cogs - gp
            excise = excise_tag if pd.notna(excise_tag) else other_cost
            if pd.notna(excise) and excise > 0 and _close(resid, excise, rev):
                n, g = rev - excise, rev                      # revenue was gross (MO)
            elif pd.notna(excise_tag) and excise_tag > 0 and _close(resid, 0, rev):
                n, g = rev, rev + excise_tag                  # revenue was net (PM)
        net.append(n)
        gross.append(g)
    df['revenue'], df['revenue_incl_excise'] = net, gross

    incl = df['_equity_incl_nci']
    df['total_equity'] = df['total_equity'].fillna(incl - df['noncontrolling_interest'])
    df['noncontrolling_interest'] = df['noncontrolling_interest'].fillna(incl - df['total_equity'])

    reported = df['gross_profit'].notna()
    df['gross_profit'] = df['gross_profit'].fillna(df['revenue'] - df['cost_of_revenue'])
    df['gross_profit_derived'] = ~reported & df['gross_profit'].notna()
    return df.drop(columns=INTERNAL)


def fix_share_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted share counts tagged in thousands or millions: if net income (to
    common) / shares is ~1,000x or ~1,000,000x the reported EPS (within 5 %),
    multiply the shares by that factor and set shares_scale_fixed = TRUE.
    Basic and diluted counts are checked against their own EPS.
    """
    if df.empty:
        return df
    df = df.copy()
    ni = pd.to_numeric(df.get('net_income_to_common'), errors='coerce').fillna(
        pd.to_numeric(df.get('net_income'), errors='coerce'))
    fixed = pd.Series(False, index=df.index)
    for shares_col, eps_col in (('shares_diluted_wavg', 'eps_diluted'), ('shares_basic_wavg', 'eps_basic')):
        shares = pd.to_numeric(df[shares_col], errors='coerce')
        eps = pd.to_numeric(df[eps_col], errors='coerce')
        ok = shares.gt(0) & eps.abs().ge(0.05) & ni.notna()
        ratio = (ni / shares / eps).where(ok)
        for factor in (1e3, 1e6):
            hit = ((ratio / factor) - 1).abs() < 0.05
            df.loc[hit, shares_col] = shares[hit] * factor
            fixed |= hit
    df['shares_scale_fixed'] = fixed
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived columns; NaN whenever an input is missing (never 0). Sign
    convention: capex, dividends_paid and share_buybacks are stored as the
    positive amounts reported by the 'Payments...' tags.
    """
    if df.empty:
        return df
    df = df.copy()
    for c in FIN_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    rev = df['revenue'].where(df['revenue'] != 0)
    df['ebitda']           = df['operating_income'] + df['depreciation_amortization']
    df['total_debt']       = df['short_term_debt'] + df['long_term_debt']
    df['net_debt']         = df['total_debt'] - df['cash']
    df['free_cash_flow']   = df['operating_cash_flow'] - df['capex']
    df['gross_margin']     = df['gross_profit'] / rev
    df['operating_margin'] = df['operating_income'] / rev
    df['net_margin']       = df['net_income'] / rev
    return df


def build_company(facts: pd.DataFrame, filings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quarterly + yearly rows for one company from its facts and filings (all CIKs).
    """
    facts = facts.drop_duplicates(['tag', 'unit', 'start', 'end', 'val', 'accn'])
    filings = filings.drop_duplicates('accession_no')
    chosen = choose_filings(facts, filings)
    chosen = chosen[chosen['fy'] >= config.FUNDAMENTALS_START_YEAR - 1]
    if chosen.empty:
        return pd.DataFrame(), pd.DataFrame()
    values = filing_values(facts, chosen)
    quarterly, yearly = assemble(chosen, values, cover_shares(facts, chosen))
    quarterly = add_derived(fix_share_scale(resolve_revenue_and_equity(quarterly)))
    yearly = add_derived(fix_share_scale(resolve_revenue_and_equity(yearly)))
    # employees are never in company facts; they are filled from company_info (see companies.sync_employees)
    keep = lambda d: d[d['fiscal_year'] >= config.FUNDAMENTALS_START_YEAR].drop(columns=['employees']) if not d.empty else d
    return keep(quarterly), keep(yearly)


# ----
# CHECKS
# ----

def run_checks(con) -> pd.DataFrame:
    """
    Plausibility checks over the stored tables -> data/review/fundamentals_checks.csv.
    """
    q = """
        SELECT '{t}' AS tbl, f.corporate_id, c.primary_ticker, f.fiscal_year, {fq} AS fiscal_quarter,
               f.total_assets, f.total_liabilities, f.total_equity, coalesce(f.redeemable_equity, 0) AS redeemable_equity,
               coalesce(f.noncontrolling_interest, 0) AS nci, f.gross_profit,
               coalesce(f.gross_profit_derived, false) AS gp_derived, f.revenue, f.cost_of_revenue,
               coalesce(f.net_income_to_common, f.net_income) AS net_income, f.shares_diluted_wavg, f.eps_diluted
        FROM {t} f LEFT JOIN company_info c USING (corporate_id)"""
    df = pd.concat([con.execute(q.format(t='fundamentals_yearly', fq='NULL')).df(),
                    con.execute(q.format(t='fundamentals_quarterly', fq='f.fiscal_quarter')).df()])
    issues = []
    keys = ['tbl', 'corporate_id', 'primary_ticker', 'fiscal_year', 'fiscal_quarter']

    bal = df.dropna(subset=['total_assets', 'total_liabilities', 'total_equity'])
    bal = bal[bal['total_assets'] != 0]
    expected = bal['total_liabilities'] + bal['redeemable_equity'] + bal['nci'] + bal['total_equity']
    diff = (bal['total_assets'] - expected) / bal['total_assets'].abs()
    for r, e, d in zip(bal[diff.abs() > 0.01].itertuples(index=False), expected[diff.abs() > 0.01], diff[diff.abs() > 0.01]):
        issues.append({**{k: getattr(r, k) for k in keys}, 'check': 'assets != liabilities + redeemable + NCI + parent equity',
                       'value': r.total_assets, 'expected': e, 'diff_pct': 100 * d})

    gp = df[~df['gp_derived'].astype(bool)].dropna(subset=['gross_profit', 'revenue', 'cost_of_revenue'])
    gp = gp[gp['revenue'] != 0]
    diff = (gp['gross_profit'] - (gp['revenue'] - gp['cost_of_revenue'])) / gp['revenue'].abs()
    for r, d in zip(gp[diff.abs() > 0.01].itertuples(index=False), diff[diff.abs() > 0.01]):
        issues.append({**{k: getattr(r, k) for k in keys}, 'check': 'gross_profit != revenue - cost_of_revenue',
                       'value': r.gross_profit, 'expected': r.revenue - r.cost_of_revenue, 'diff_pct': 100 * d})

    eps = df.dropna(subset=['net_income', 'shares_diluted_wavg', 'eps_diluted'])
    eps = eps[(eps['shares_diluted_wavg'] > 0) & (eps['eps_diluted'].abs() >= 0.05)]
    calc = eps['net_income'] / eps['shares_diluted_wavg']
    diff = (calc - eps['eps_diluted']) / eps['eps_diluted'].abs()
    bad = (diff.abs() > 0.05) & ((calc - eps['eps_diluted']).abs() > 0.01)     # EPS is rounded to cents
    for r, cv, d in zip(eps[bad].itertuples(index=False), calc[bad], diff[bad]):
        issues.append({**{k: getattr(r, k) for k in keys}, 'check': 'net_income / diluted shares vs eps_diluted > 5%',
                       'value': cv, 'expected': r.eps_diluted, 'diff_pct': 100 * d})

    yr = df[df['tbl'] == 'fundamentals_yearly'].dropna(subset=['revenue']).sort_values(['corporate_id', 'fiscal_year'])
    yr['prev'] = yr.groupby('corporate_id')['revenue'].shift()
    yr['prev_year'] = yr.groupby('corporate_id')['fiscal_year'].shift()
    jump = yr[(yr['prev_year'] == yr['fiscal_year'] - 1) & (yr['prev'] != 0)]
    ratio = jump['revenue'] / jump['prev'] - 1
    for r, d in zip(jump[ratio.abs() > 0.5].itertuples(index=False), ratio[ratio.abs() > 0.5]):
        issues.append({**{k: getattr(r, k) for k in keys}, 'check': 'revenue change vs previous year > 50%',
                       'value': r.revenue, 'expected': r.prev, 'diff_pct': 100 * d})

    out = pd.DataFrame(issues, columns=keys + ['check', 'value', 'expected', 'diff_pct'])
    out.to_csv(CHECKS_CSV, index=False)
    return out


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    con = common.connect('companies')
    try:
        companies = con.execute("""
            SELECT corporate_id, primary_ticker, tickers, cik FROM company_info
            WHERE sec_filer AND cik IS NOT NULL""").df()
        if not config.MATCH_NON_US_TO_SEC:
            us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
            companies = companies[companies['tickers'].map(lambda ts: bool(us & set(ts)))]
        if ctx.tickers:
            wanted = {t.upper() for t in ctx.tickers}
            companies = companies[companies['tickers'].map(lambda ts: bool(wanted & {t.upper() for t in ts}))]
        state = {cid: (d, k, fp) for cid, d, k, fp in con.execute(
            'SELECT corporate_id, last_filing_date, build_key, facts_fingerprint FROM fundamentals_state').fetchall()}
        unchanged = 0
        known = {r[0] for r in con.execute('SELECT DISTINCT cik FROM sec_filings').fetchall()}
        log.info(f'{len(companies)} SEC filers')

        for i, c in enumerate(companies.itertuples(index=False), 1):
            cid, ticker = int(c.corporate_id), c.primary_ticker
            ciks = [int(c.cik)] + config.SEC_PREDECESSOR_CIKS.get(int(c.cik), [])
            build_key = f'v{BUILD_VERSION}|' + ','.join(map(str, ciks))
            try:
                filings = pd.concat([fetch_submissions(k, all_pages=k not in known) for k in ciks],
                                    ignore_index=True).drop_duplicates('accession_no')
                keep = filings[filings['form'].map(base_form).isin(KEEP_FORMS)]
                common.upsert(con, 'sec_filings', keep.assign(corporate_id=cid).assign(
                    filed_date=keep['filed_date'].dt.date, report_date=keep['report_date'].dt.date)[
                    ['accession_no', 'corporate_id', 'cik', 'form', 'filed_date', 'report_date', 'items',
                     'primary_doc', 'acceptance_time']].rename(columns={'form': 'form_type'}))

                # build from the full stored history (older pages are only downloaded once per CIK)
                filings = con.execute("""
                    SELECT accession_no, form_type AS form, filed_date, report_date, items, primary_doc, cik
                    FROM sec_filings WHERE cik IN (SELECT unnest(?))""", [ciks]).df()
                filings['filed_date'] = pd.to_datetime(filings['filed_date'])
                filings['report_date'] = pd.to_datetime(filings['report_date'])
                trig = filings[filings['form'].map(base_form).isin(TRIGGER_FORMS)]
                newest = trig['filed_date'].max().date() if not trig.empty else None
                # SEC's company facts lag behind EDGAR: a recent 10-Q / 10-K that is not in the
                # fundamentals yet keeps the company on the list until its facts arrive
                pending = con.execute("""
                    SELECT count(*) FROM sec_filings s
                    WHERE s.cik IN (SELECT unnest(?)) AND s.form_type IN ('10-Q', '10-K')
                      AND s.filed_date >= current_date - INTERVAL 400 DAY
                      AND s.accession_no NOT IN (
                          SELECT accession_no FROM fundamentals_quarterly WHERE corporate_id = ? AND accession_no IS NOT NULL
                          UNION SELECT accession_no FROM fundamentals_yearly WHERE corporate_id = ? AND accession_no IS NOT NULL)""",
                    [ciks, cid, cid]).fetchone()[0]
                last, key, old_fp = state.get(cid, (None, None, None))
                up_to_date = key == build_key and last is not None and newest is not None and newest <= last
                if up_to_date and not pending:
                    result.skipped += 1
                    continue

                facts = pd.concat([fetch_facts(k) for k in ciks], ignore_index=True)
                fp = facts_fingerprint(facts)
                # only kept on the list by a filing whose facts haven't arrived: rebuild only if the facts changed
                if up_to_date and fp == old_fp:
                    result.skipped += 1
                    unchanged += 1
                    continue
                quarterly, yearly = build_company(facts, filings)
                if yearly.empty and quarterly.empty:
                    result.fail(ticker, 'no 10-K / 10-Q facts', log)
                    continue
                now = datetime.now()
                result.rows += common.upsert(con, 'fundamentals_yearly', yearly.assign(corporate_id=cid, updated_at=now))
                result.rows += common.upsert(con, 'fundamentals_quarterly', quarterly.assign(corporate_id=cid, updated_at=now))
                # rows from an earlier build whose key no longer exists (mislabelled filings); rows without a
                # period end (quarters created by the calendar / holders steps before their 10-Q) stay
                con.register('_yk', yearly[['fiscal_year']])
                con.register('_qk', quarterly[['fiscal_year', 'fiscal_quarter']])
                con.execute("""DELETE FROM fundamentals_yearly WHERE corporate_id = ? AND period_end IS NOT NULL
                               AND fiscal_year NOT IN (SELECT fiscal_year FROM _yk)""", [cid])
                con.execute("""DELETE FROM fundamentals_quarterly WHERE corporate_id = ? AND period_end IS NOT NULL
                               AND (fiscal_year, fiscal_quarter) NOT IN (SELECT (fiscal_year, fiscal_quarter) FROM _qk)""",
                            [cid])
                con.unregister('_yk')
                con.unregister('_qk')
                common.upsert(con, 'fundamentals_state', pd.DataFrame([{
                    'corporate_id': cid, 'last_filing_date': newest, 'processed_at': now, 'build_key': build_key,
                    'facts_fingerprint': fp}]))
                result.updated += 1
            except Exception as e:
                result.fail(ticker, common.format_error(e), log)
            if i % 25 == 0:
                log.info(f'  ...{i}/{len(companies)}')

        if unchanged:
            log.info(f'{unchanged} companies with a filing not yet in their company facts: facts unchanged, not rebuilt')
        checks = run_checks(con)
        if not checks.empty:
            per = checks.groupby('check')['corporate_id'].nunique()
            log.info('checks -> ' + CHECKS_CSV.name + ': ' + ', '.join(f'{k}: {v} companies' for k, v in per.items()))
    finally:
        con.close()
    return result
