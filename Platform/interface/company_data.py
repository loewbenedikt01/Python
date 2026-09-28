"""
Queries for the company page. Plain functions (no Streamlit): every call opens the
pipeline's DuckDB files read-only, queries and closes them at once, so an open app
never blocks the nightly pipeline run. `folder` is Database/data.

Raises DatabaseLocked while the pipeline writes to a database.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd


class DatabaseLocked(Exception):
    """The pipeline is writing to a database file."""


@contextmanager
def connect(folder: str | Path, db: str = 'prices', attach: tuple[str, ...] = ()):
    """
    Read-only connection to <db>.duckdb with other databases ATTACHed read-only under their names
    (e.g. attach=('companies',) -> companies.company_info). Closed on exit.
    """
    root = Path(folder).expanduser()
    try:
        con = duckdb.connect(str(root / f'{db}.duckdb'), read_only=True)
        for other in attach:
            con.execute(f"ATTACH '{(root / f'{other}.duckdb').as_posix()}' AS {other} (READ_ONLY)")
    except duckdb.IOException as exc:
        raise DatabaseLocked(str(exc)) from exc
    try:
        yield con
    finally:
        con.close()


# ----
# COMPANY LIST AND HEADER
# ----

def companies(folder) -> pd.DataFrame:
    """
    Active companies: corporate_id, primary_ticker, tickers, name, sector, industry, country, sec_filer.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT corporate_id, primary_ticker, tickers, name, sector, industry, country,
                   coalesce(sec_filer, FALSE) AS sec_filer
            FROM company_info
            WHERE coalesce(status, 'active') = 'active' AND primary_ticker IS NOT NULL
            ORDER BY name""").df()


def hq_points(folder) -> pd.DataFrame:
    """
    Active companies with headquarters coordinates: ticker, name, sector, industry, country, continent,
    hq_city, hq_country, hq_lat, hq_lon, hq_geo_source, sec_filer, market_cap_usd (latest).
    """
    with connect(folder, 'companies', attach=('prices',)) as con:
        return con.execute("""
            WITH mc AS (SELECT corporate_id, arg_max(market_cap_usd, date) AS market_cap_usd
                        FROM prices.market_cap_daily GROUP BY 1)
            SELECT c.corporate_id, c.primary_ticker, c.tickers, c.name, coalesce(c.sector, 'Unknown') AS sector,
                   coalesce(c.industry, 'Unknown') AS industry, c.country, coalesce(c.continent, 'Unknown') AS continent,
                   c.hq_city, coalesce(c.hq_country, c.country) AS hq_country, c.hq_lat, c.hq_lon, c.hq_geo_source,
                   coalesce(c.sec_filer, FALSE) AS sec_filer, mc.market_cap_usd
            FROM company_info c LEFT JOIN mc USING (corporate_id)
            WHERE coalesce(c.status, 'active') = 'active' AND c.hq_lat IS NOT NULL AND c.primary_ticker IS NOT NULL
            ORDER BY c.name""").df()


def _plain(text) -> str:
    import unicodedata
    return unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode().lower()


def search_companies(comps: pd.DataFrame, query: str, limit: int = 8) -> pd.DataFrame:
    """
    Companies matching `query` (case and accents ignored), best first:
    0 ticker equals it (any listing: GOOG -> Alphabet), 1 ticker starts with it, 2 name starts with it,
    3 a word of the name starts with it, 4 name contains it. No fuzzy matching.
    """
    q = _plain(query).strip()
    if not q:
        return comps.iloc[0:0]
    scores = []
    for p, ts, name in zip(comps['primary_ticker'], comps['tickers'], comps['name']):
        tks = [_plain(t) for t in ([p] + list(ts if ts is not None else []))]
        nm = _plain(name)
        words = nm.replace('-', ' ').replace(',', ' ').replace('.', ' ').split()
        if q in tks:
            sc = 0
        elif any(t.startswith(q) for t in tks):
            sc = 1
        elif nm.startswith(q):
            sc = 2
        elif any(w.startswith(q) for w in words):
            sc = 3
        elif q in nm:
            sc = 4
        else:
            sc = None
        scores.append(sc)
    out = comps.assign(_score=scores).dropna(subset=['_score'])
    out = out.assign(_len=out['name'].str.len()).sort_values(['_score', '_len', 'name'])
    return out.drop(columns=['_score', '_len']).head(limit)


def header(folder, corporate_id: int) -> dict:
    """
    Company info plus the latest price, market cap, shares, 52-week range and next earnings date.
    """
    with connect(folder, 'prices', attach=('companies',)) as con:
        info = con.execute('SELECT * FROM companies.company_info WHERE corporate_id = ?', [corporate_id]).df()
        if info.empty:
            return {}
        out = info.iloc[0].to_dict()
        ticker = out['primary_ticker']
        px = con.execute("""
            SELECT date, close, currency, close_usd FROM prices_daily
            WHERE ticker = ? AND close IS NOT NULL AND NOT coalesce(price_suspect, FALSE)
            ORDER BY date DESC LIMIT 2""", [ticker]).fetchall()
        if px:
            out.update(price_date=px[0][0], price=px[0][1], price_currency=px[0][2], price_usd=px[0][3],
                       prev_close=px[1][1] if len(px) > 1 else None)
        rng = con.execute("""
            SELECT min(low), max(high) FROM prices_daily
            WHERE ticker = ? AND date > (SELECT max(date) FROM prices_daily WHERE ticker = ?) - INTERVAL 365 DAY
              AND NOT coalesce(price_suspect, FALSE)""", [ticker, ticker]).fetchone()
        out.update(low_52w=rng[0], high_52w=rng[1])
        mc = con.execute("""
            SELECT date, market_cap, currency, market_cap_usd, shares_outstanding, shares_source
            FROM market_cap_daily WHERE corporate_id = ? ORDER BY date DESC LIMIT 1""", [corporate_id]).fetchone()
        if mc:
            out.update(mcap_date=mc[0], market_cap=mc[1], mcap_currency=mc[2], market_cap_usd=mc[3],
                       shares=mc[4], shares_source=mc[5])
        nxt = con.execute("""
            SELECT earnings_date, earnings_time, earnings_date_status FROM companies.fundamentals_quarterly
            WHERE corporate_id = ? AND earnings_date >= current_date ORDER BY earnings_date LIMIT 1""",
            [corporate_id]).fetchone()
        if nxt:
            out.update(next_earnings=nxt[0], next_earnings_time=nxt[1], next_earnings_status=nxt[2])
        # share count as of: latest SEC count date
        asof = con.execute("""
            SELECT max(shares_as_of) FROM companies.fundamentals_quarterly WHERE corporate_id = ?""",
            [corporate_id]).fetchone()[0]
        out['shares_as_of'] = asof
        return out


# ----
# PRICES
# ----

def daily_prices(folder, ticker: str, corporate_id: int | None = None) -> pd.DataFrame:
    """
    date, open, high, low, close, volume (+ market_cap, market_cap_usd if corporate_id); suspect days left out.
    """
    with connect(folder, 'prices') as con:
        df = con.execute("""
            SELECT date, open, high, low, close, volume FROM prices_daily
            WHERE ticker = ? AND close IS NOT NULL AND NOT coalesce(price_suspect, FALSE) ORDER BY date""",
            [ticker]).df()
        if corporate_id is not None and not df.empty:
            mc = con.execute("""SELECT date, market_cap, market_cap_usd FROM market_cap_daily
                                WHERE corporate_id = ? ORDER BY date""", [corporate_id]).df()
            df = df.merge(mc, on='date', how='left')
    df['date'] = pd.to_datetime(df['date'])
    return df


INTRADAY_INTERVALS = ['1m', '5m', '10m', '1h']


def has_intraday(folder, ticker: str) -> bool:
    return any((Path(folder) / 'intraday' / '1m' / f'ticker={ticker}').glob('*.parquet'))


def intraday_prices(folder, ticker: str, interval: str, days: int) -> pd.DataFrame:
    """
    Bars of the last `days` calendar days (of the ticker's data), ts in New York time.
    """
    files = sorted((Path(folder) / 'intraday' / interval / f'ticker={ticker}').glob('*.parquet'))
    if not files:
        return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'vwap'])
    paths = [f.as_posix() for f in files[-2:]]           # yearly files: the last two cover any window <= 1 year
    con = duckdb.connect()
    try:
        df = con.execute("""
            WITH b AS (SELECT * FROM read_parquet(?))
            SELECT ts, open, high, low, close, volume, vwap FROM b
            WHERE ts >= (SELECT max(ts) FROM b) - to_days(?) ORDER BY ts""", [paths, days]).df()
    finally:
        con.close()
    df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    return df


# ----
# OVERVIEW
# ----

KEY_FIGURES = ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda', 'free_cash_flow',
               'net_debt', 'total_equity', 'eps_diluted', 'gross_margin', 'operating_margin', 'net_margin']


def key_figures(folder, corporate_id: int) -> pd.DataFrame:
    """
    Last fiscal year and trailing twelve months (sum of the last 4 quarters for flows, latest quarter for
    balance-sheet items and margins recomputed) of KEY_FIGURES. Columns: item, last_fy, ttm; plus attrs.
    """
    with connect(folder, 'companies') as con:
        fy = con.execute("""
            SELECT * FROM fundamentals_yearly WHERE corporate_id = ? AND period_end IS NOT NULL
            ORDER BY fiscal_year DESC LIMIT 1""", [corporate_id]).df()
        q = con.execute("""
            SELECT * FROM fundamentals_quarterly WHERE corporate_id = ? AND period_end IS NOT NULL
            ORDER BY period_end DESC LIMIT 4""", [corporate_id]).df()
    if fy.empty and q.empty:
        return pd.DataFrame()
    flows = ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda', 'free_cash_flow', 'eps_diluted']
    ttm = {}
    if len(q) == 4:
        for c in flows:
            ttm[c] = q[c].sum(min_count=4)
        for c in ('net_debt', 'total_equity'):
            ttm[c] = q[c].iloc[0]
        rev = ttm.get('revenue')
        for m, num in (('gross_margin', 'gross_profit'), ('operating_margin', 'operating_income'),
                       ('net_margin', 'net_income')):
            ttm[m] = ttm[num] / rev if rev and pd.notna(ttm[num]) else None
    out = pd.DataFrame({'item': KEY_FIGURES,
                        'last_fy': [fy.iloc[0][c] if not fy.empty else None for c in KEY_FIGURES],
                        'ttm': [ttm.get(c) for c in KEY_FIGURES]})
    out.attrs['fiscal_year'] = int(fy.iloc[0]['fiscal_year']) if not fy.empty else None
    out.attrs['ttm_end'] = q.iloc[0]['period_end'] if len(q) == 4 else None
    out.attrs['currency'] = (fy.iloc[0]['currency'] if not fy.empty else q.iloc[0]['currency'])
    return out


def filings(folder, corporate_id: int, n: int = 8) -> pd.DataFrame:
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT filed_date, form_type, report_date, items, accession_no, cik, primary_doc FROM sec_filings
            WHERE corporate_id = ? AND form_type IN ('10-K', '10-Q', '8-K', '20-F', '10-K/A', '10-Q/A', 'DEF 14A')
            ORDER BY filed_date DESC LIMIT ?""", [corporate_id, n]).df()


def filing_url(row) -> str:
    acc = str(row['accession_no']).replace('-', '')
    return f"https://www.sec.gov/Archives/edgar/data/{int(row['cik'])}/{acc}/{row['primary_doc']}"


# ----
# FINANCIALS
# ----

STATEMENTS = {
    'Income statement': ['revenue', 'cost_of_revenue', 'gross_profit', 'rnd_expense', 'sga_expense',
                         'operating_expenses', 'operating_income', 'interest_expense', 'pretax_income',
                         'income_tax', 'net_income', 'net_income_to_common', 'ebitda',
                         'depreciation_amortization', 'eps_basic', 'eps_diluted', 'shares_basic_wavg',
                         'shares_diluted_wavg', 'gross_margin', 'operating_margin', 'net_margin'],
    'Balance sheet': ['cash', 'short_term_investments', 'receivables', 'inventory', 'current_assets', 'ppe_net',
                      'goodwill', 'intangibles', 'total_assets', 'accounts_payable', 'current_liabilities',
                      'short_term_debt', 'long_term_debt', 'total_debt', 'net_debt', 'total_liabilities',
                      'redeemable_equity', 'noncontrolling_interest', 'total_equity', 'retained_earnings',
                      'shares_outstanding'],
    'Cash flow': ['operating_cash_flow', 'capex', 'free_cash_flow', 'investing_cash_flow', 'financing_cash_flow',
                  'dividends_paid', 'share_buybacks', 'stock_based_comp', 'dividends_per_share'],
}
PER_SHARE = {'eps_basic', 'eps_diluted', 'dividends_per_share'}
RATIOS = {'gross_margin', 'operating_margin', 'net_margin'}
COUNTS = {'shares_basic_wavg', 'shares_diluted_wavg', 'shares_outstanding'}


def financials(folder, corporate_id: int, period: str = 'Y', n: int = 10) -> pd.DataFrame:
    """
    Rows of fundamentals_yearly (period 'Y', last n years) or fundamentals_quarterly ('Q', last n quarters),
    newest first. Columns: label, period_end, currency, q4_derived, gross_profit_derived + all line items.
    """
    items = sorted({c for cols in STATEMENTS.values() for c in cols})
    with connect(folder, 'companies') as con:
        if period == 'Y':
            df = con.execute(f"""
                SELECT 'FY' || fiscal_year AS label, period_end, filed_date, currency, FALSE AS q4_derived,
                       gross_profit_derived, {', '.join(items)}
                FROM fundamentals_yearly WHERE corporate_id = ? AND period_end IS NOT NULL
                ORDER BY fiscal_year DESC LIMIT ?""", [corporate_id, n]).df()
        else:
            df = con.execute(f"""
                SELECT 'Q' || fiscal_quarter || ' ' || fiscal_year AS label, period_end, filed_date, currency,
                       coalesce(q4_derived, FALSE) AS q4_derived, gross_profit_derived, {', '.join(items)}
                FROM fundamentals_quarterly WHERE corporate_id = ? AND period_end IS NOT NULL
                ORDER BY period_end DESC LIMIT ?""", [corporate_id, n]).df()
    return df


def statement_table(fin: pd.DataFrame, statement: str) -> pd.DataFrame:
    """
    Line items as rows, periods as columns (newest left). Lines without any value are dropped.
    """
    cols = [c for c in STATEMENTS[statement] if c in fin.columns]
    t = fin.set_index('label')[cols].T
    t = t.dropna(how='all')
    t.index.name = 'item'
    return t


# ----
# HOLDERS / EXECUTIVES / SUBSIDIARIES
# ----

def holders(folder, corporate_id: int) -> pd.DataFrame:
    """
    Quarters with 13F data: label, period_end, holders_report_date, inst_ownership_pct,
    n_institutional_holders, top_holders (list of dicts); newest first.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT 'Q' || fiscal_quarter || ' ' || fiscal_year AS label, period_end, holders_report_date,
                   inst_ownership_pct, n_institutional_holders, top_holders
            FROM fundamentals_quarterly
            WHERE corporate_id = ? AND (top_holders IS NOT NULL OR inst_ownership_pct IS NOT NULL)
            ORDER BY fiscal_year DESC, fiscal_quarter DESC""", [corporate_id]).df()


def executives(folder, corporate_id: int) -> pd.DataFrame:
    """
    fiscal_year, executives (list of dicts); newest first.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT fiscal_year, executives FROM fundamentals_yearly
            WHERE corporate_id = ? AND executives IS NOT NULL AND len(executives) > 0
            ORDER BY fiscal_year DESC""", [corporate_id]).df()


def insider_transactions(folder, corporate_id: int, n: int = 50) -> pd.DataFrame:
    with connect(folder, 'raw') as con:
        return con.execute("""
            SELECT trans_date, filing_date, owner_readable_name AS name, role, title, trans_code, acquired_disposed,
                   shares, price, shares * price AS value, shares_owned_after, coalesce(date_suspect, FALSE) AS date_suspect
            FROM insider_transactions_raw WHERE corporate_id = ?
            ORDER BY filing_date DESC, trans_date DESC LIMIT ?""", [corporate_id, n]).df()


def subsidiaries(folder, corporate_id: int) -> pd.DataFrame:
    """
    fiscal_year, n_subsidiaries, subsidiaries (list of dicts), subsidiaries_scope, carried_forward; newest first.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT fiscal_year, n_subsidiaries, subsidiaries, subsidiaries_scope,
                   coalesce(subsidiaries_carried_forward, FALSE) AS carried_forward
            FROM fundamentals_yearly WHERE corporate_id = ? AND subsidiaries IS NOT NULL
            ORDER BY fiscal_year DESC""", [corporate_id]).df()


def year_changes(rows: pd.DataFrame, list_col: str, key: str, year: int) -> tuple[list[str], list[str]]:
    """
    (added, removed) names between `year` and the year before (by `key` of the list entries).
    """
    by_year = {int(r['fiscal_year']): {x[key] for x in (r[list_col] if r[list_col] is not None else []) if x[key]}
               for _, r in rows.iterrows()}
    cur, prev = by_year.get(year, set()), by_year.get(year - 1)
    if prev is None:
        return [], []
    return sorted(cur - prev), sorted(prev - cur)


# ----
# CALENDAR
# ----

def earnings_history(folder, corporate_id: int, ticker: str, n: int = 12) -> pd.DataFrame:
    """
    Earnings dates (past and upcoming) with the price reaction: close before -> close after the release
    (bmo: previous close -> same-day close; amc / unknown: same-day close -> next close).
    """
    with connect(folder, 'prices', attach=('companies',)) as con:
        return con.execute("""
            WITH e AS (
                SELECT 'Q' || fiscal_quarter || ' ' || fiscal_year AS quarter, period_end, earnings_date,
                       earnings_time, earnings_date_status
                FROM companies.fundamentals_quarterly
                WHERE corporate_id = ? AND earnings_date IS NOT NULL
                ORDER BY earnings_date DESC LIMIT ?),
            p AS (SELECT date, close, lag(close) OVER (ORDER BY date) AS prev_close,
                         lead(close) OVER (ORDER BY date) AS next_close
                  FROM prices_daily WHERE ticker = ? AND NOT coalesce(price_suspect, FALSE))
            SELECT e.*, CASE WHEN e.earnings_time = 'bmo' THEN p.close / p.prev_close - 1
                             ELSE p.next_close / p.close - 1 END AS reaction
            FROM e ASOF LEFT JOIN p ON e.earnings_date >= p.date
            ORDER BY e.earnings_date DESC""", [corporate_id, n, ticker]).df()


def annual_reports(folder, corporate_id: int, n: int = 6) -> pd.DataFrame:
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT fiscal_year, period_end, annual_report_date FROM fundamentals_yearly
            WHERE corporate_id = ? AND annual_report_date IS NOT NULL
            ORDER BY fiscal_year DESC LIMIT ?""", [corporate_id, n]).df()


def dividends_splits(folder, ticker: str, n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    with connect(folder, 'prices') as con:
        div = con.execute("""
            SELECT date, dividends, currency FROM prices_daily WHERE ticker = ? AND dividends > 0
            ORDER BY date DESC LIMIT ?""", [ticker, n]).df()
        ca = con.execute("""
            SELECT date, yahoo_factor, split_factor, price_only_factor, method FROM corporate_actions
            WHERE ticker = ? ORDER BY date DESC""", [ticker]).df()
    return div, ca


# ----
# CLINICAL TRIALS
# ----

def _missing_ok(fn):
    """Trial tables / the version view don't exist before the first trials run: empty result instead."""
    def wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except duckdb.CatalogException:
            return pd.DataFrame()
    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


@_missing_ok
def trials(folder, corporate_id: int) -> pd.DataFrame:
    """
    Trials where the company is lead sponsor or collaborator (role column), newest start first.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT t.nct_id, t.title, s.role, t.phases, t.phase_groups, t.overall_status, t.study_type,
                   t.start_date, t.start_year, t.primary_completion_date, t.completion_date, t.enrollment,
                   t.conditions, t.interventions, t.has_results, t.lead_sponsor, t.version
            FROM clinical_trials t JOIN clinical_trial_sponsors s USING (nct_id)
            WHERE s.corporate_id = ?
            QUALIFY row_number() OVER (PARTITION BY t.nct_id ORDER BY s.role = 'lead' DESC) = 1
            ORDER BY t.start_date DESC NULLS LAST""", [corporate_id]).df()


@_missing_ok
def trial_phase_counts(folder, corporate_id: int) -> pd.DataFrame:
    """
    start_year, phase (1-4), trials. A Phase 1/2 trial counts in 1 and 2; each trial once per company.
    """
    with connect(folder, 'companies') as con:
        return con.execute("""
            SELECT start_year, phase, count(DISTINCT nct_id) AS trials
            FROM (SELECT nct_id, start_year, unnest(phase_groups) AS phase FROM clinical_trials
                  WHERE nct_id IN (SELECT nct_id FROM clinical_trial_sponsors WHERE corporate_id = ?)
                    AND start_year IS NOT NULL)
            GROUP BY ALL ORDER BY 1, 2""", [corporate_id]).df()


@_missing_ok
def trial_versions(folder, nct_id: str) -> pd.DataFrame:
    """
    Version history of one trial: version, downloaded_at, last_change_date, changed_sections.
    """
    with connect(folder, 'raw') as con:
        return con.execute("""
            SELECT version, downloaded_at, last_change_date, change_flag, changed_sections
            FROM clinical_trials_raw WHERE nct_id = ? ORDER BY version DESC""", [nct_id]).df()
