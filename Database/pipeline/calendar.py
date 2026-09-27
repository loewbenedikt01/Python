"""
Step 'calendar': earnings dates in fundamentals_quarterly / fundamentals_yearly.

- Past (status 'reported'): the first 8-K with item 2.02 accepted after the
  quarter's period end and within EARNINGS_WINDOW_DAYS. The date and time of
  day come from the filing's EDGAR acceptance time converted to New York time
  (before 09:30 -> bmo, after 16:00 -> amc, else during). The acceptance date
  is used rather than the filing date, because EDGAR dates filings accepted
  after 17:30 on the next business day.
- A 2.02 8-K after the latest quarter with fundamentals (10-Q not filed yet)
  creates the row of the next fiscal quarter with status 'reported'.
- Upcoming (US companies only; Finnhub's free plan has no others): the
  earliest Finnhub date from today on, assigned to the fiscal quarter after
  the latest reported one (Finnhub's own quarter labels are not used). Time
  from Finnhub's hour field. Status 'confirmed' only if Finnhub says so; its
  calendar has no such field, so upcoming dates are 'estimated'.
- Yearly rows: earnings_date / earnings_time / status of fiscal Q4, and
  annual_report_date = the 10-K filing date.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

import config
from pipeline import common
from pipeline.fundamentals import fetch_submissions


log = common.get_logger('pipeline.calendar')

EARNINGS_WINDOW_DAYS = 90          # 2.02 8-K must follow the period end within this many days
UPCOMING_DAYS        = 100         # Finnhub look-ahead
FINNHUB_CHUNK_DAYS   = 7           # Finnhub returns at most 1,500 rows per call
NEW_QUARTER_GAP_DAYS = 45          # a 2.02 this long after the last reported one is the next quarter
FINNHUB_HOUR = {'bmo': 'bmo', 'amc': 'amc', 'dmh': 'during'}


def new_york_time(acceptance_utc):
    """
    EDGAR acceptance time (UTC) as New York local time without time zone.
    """
    if acceptance_utc is None or pd.isna(acceptance_utc):
        return None
    return pd.Timestamp(acceptance_utc).tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)


def time_of_day(acceptance_utc) -> tuple[date | None, str | None]:
    """
    (New York date, 'bmo' / 'during' / 'amc') of an EDGAR acceptance time in UTC.
    """
    ny = new_york_time(acceptance_utc)
    if ny is None:
        return None, None
    minutes = ny.hour * 60 + ny.minute
    kind = 'bmo' if minutes < 9 * 60 + 30 else ('amc' if minutes >= 16 * 60 else 'during')
    return ny.date(), kind


def next_quarter(fy: int, q: int) -> tuple[int, int]:
    return (fy + 1, 1) if q == 4 else (fy, q + 1)


# ----
# INPUTS
# ----

def backfill_acceptance(con, companies: pd.DataFrame, result: common.StepResult) -> None:
    """
    Acceptance times were not stored for filings loaded before this step
    existed; fetch the full submissions once for those CIKs.
    """
    ciks = [r[0] for r in con.execute("""
        SELECT DISTINCT cik FROM sec_filings
        WHERE form_type = '8-K' AND acceptance_time IS NULL AND cik IN (SELECT unnest(?))""",
        [companies['all_ciks'].explode().dropna().astype(int).tolist()]).fetchall()]
    if not ciks:
        return
    log.info(f'fetching acceptance times for {len(ciks)} CIKs (once)')
    for cik in ciks:
        try:
            f = fetch_submissions(int(cik), all_pages=True)
            f = f[f['acceptance_time'].notna()][['accession_no', 'acceptance_time']]
            # only fill rows that exist (the filings kept in sec_filings), never add others
            con.register('_acc', f)
            con.execute("""UPDATE sec_filings AS s SET acceptance_time = a.acceptance_time FROM _acc a
                           WHERE s.accession_no = a.accession_no AND s.acceptance_time IS NULL""")
            con.unregister('_acc')
        except Exception as e:
            result.fail(f'CIK {cik}', common.format_error(e), log)


def finnhub_calendar(start: date, end: date) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        to = min(d + timedelta(days=FINNHUB_CHUNK_DAYS - 1), end)
        r = common.http_get('https://finnhub.io/api/v1/calendar/earnings', 'finnhub', params={
            'from': str(d), 'to': str(to), 'token': common.get_key('FINNHUB')})
        cal = r.json().get('earningsCalendar') or []
        if len(cal) >= 1500:
            log.warning(f'Finnhub returned the maximum of 1500 rows for {d}..{to}; some dates may be missing')
        rows += cal
        d = to + timedelta(days=1)
    df = pd.DataFrame(rows, columns=['symbol', 'date', 'hour'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def yahoo_to_finnhub(ticker: str) -> str:
    return ticker.replace('-', '.')           # BRK-B -> BRK.B


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    today = date.today()
    con = common.connect('companies')
    try:
        us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
        companies = con.execute("""
            SELECT corporate_id, primary_ticker, tickers, cik FROM company_info
            WHERE sec_filer AND cik IS NOT NULL""").df()
        companies = companies[companies['tickers'].map(lambda ts: bool(us & set(ts)))]
        if ctx.tickers:
            wanted = {t.upper() for t in ctx.tickers}
            companies = companies[companies['tickers'].map(lambda ts: bool(wanted & {t.upper() for t in ts}))]
        companies['all_ciks'] = [[int(c)] + config.SEC_PREDECESSOR_CIKS.get(int(c), []) for c in companies['cik']]
        ids = companies['corporate_id'].astype(int).tolist()
        backfill_acceptance(con, companies, result)

        # 2.02 releases per company (original 8-Ks)
        rel = con.execute("""
            SELECT corporate_id, accession_no, filed_date, acceptance_time FROM sec_filings
            WHERE form_type = '8-K' AND items LIKE '%2.02%' AND corporate_id IN (SELECT unnest(?))""", [ids]).df()
        rel[['earnings_date', 'earnings_time']] = [time_of_day(a) for a in rel['acceptance_time']]
        rel['earnings_accepted_at'] = [new_york_time(a) for a in rel['acceptance_time']]
        rel['earnings_date'] = rel['earnings_date'].fillna(pd.to_datetime(rel['filed_date']).dt.date)
        rel = rel.sort_values(['corporate_id', 'earnings_date'])

        quarters = con.execute("""
            SELECT corporate_id, fiscal_year, fiscal_quarter, period_end FROM fundamentals_quarterly
            WHERE corporate_id IN (SELECT unnest(?)) AND period_end IS NOT NULL""", [ids]).df()
        quarters['period_end'] = pd.to_datetime(quarters['period_end']).dt.date

        # past: first 2.02 after the period end, within the window
        past = []
        by_cid = {cid: g for cid, g in rel.groupby('corporate_id')}
        for r in quarters.itertuples(index=False):
            g = by_cid.get(r.corporate_id)
            if g is None:
                continue
            hit = g[(g['earnings_date'] > r.period_end)
                    & (g['earnings_date'] <= r.period_end + timedelta(days=EARNINGS_WINDOW_DAYS))]
            if not hit.empty:
                h = hit.iloc[0]
                past.append({'corporate_id': r.corporate_id, 'fiscal_year': r.fiscal_year,
                             'fiscal_quarter': r.fiscal_quarter, 'earnings_date': h['earnings_date'],
                             'earnings_time': h['earnings_time'], 'earnings_accepted_at': h['earnings_accepted_at'],
                             'earnings_date_status': 'reported'})
        past = pd.DataFrame(past)

        # latest reported quarter per company; a newer 2.02 without a 10-Q yet = the next quarter
        latest = {}
        for cid, g in quarters.sort_values(['fiscal_year', 'fiscal_quarter']).groupby('corporate_id'):
            last = g.iloc[-1]
            latest[cid] = (int(last['fiscal_year']), int(last['fiscal_quarter']))
        last_release = past.groupby('corporate_id')['earnings_date'].max().to_dict() if not past.empty else {}
        extra = []
        for cid, (fy, q) in latest.items():
            g = by_cid.get(cid)
            if g is None or cid not in last_release:
                continue
            newer = g[g['earnings_date'] > last_release[cid] + timedelta(days=NEW_QUARTER_GAP_DAYS)]
            if not newer.empty:
                h = newer.iloc[0]
                fy, q = next_quarter(fy, q)
                extra.append({'corporate_id': cid, 'fiscal_year': fy, 'fiscal_quarter': q,
                              'earnings_date': h['earnings_date'], 'earnings_time': h['earnings_time'],
                              'earnings_accepted_at': h['earnings_accepted_at'], 'earnings_date_status': 'reported'})
                latest[cid] = (fy, q)
        reported = pd.concat([past, pd.DataFrame(extra)], ignore_index=True)

        # upcoming (Finnhub, US only) -> quarter after the latest reported one
        cal = finnhub_calendar(today, today + timedelta(days=UPCOMING_DAYS))
        symbol_to_cid = {yahoo_to_finnhub(t): int(cid) for cid, ts in zip(companies['corporate_id'], companies['tickers'])
                         for t in ts if t in us}
        cal['corporate_id'] = cal['symbol'].map(symbol_to_cid)
        cal = cal.dropna(subset=['corporate_id']).sort_values('date').drop_duplicates('corporate_id')
        upcoming = []
        for r in cal.itertuples(index=False):
            cid = int(r.corporate_id)
            if cid not in latest:
                continue
            fy, q = next_quarter(*latest[cid])
            upcoming.append({'corporate_id': cid, 'fiscal_year': fy, 'fiscal_quarter': q, 'earnings_date': r.date,
                             'earnings_time': FINNHUB_HOUR.get(r.hour), 'earnings_date_status': 'estimated'})
        upcoming = pd.DataFrame(upcoming)

        # write: old estimates are cleared first (dates move, or the company drops out of Finnhub)
        con.execute("""
            UPDATE fundamentals_quarterly SET earnings_date = NULL, earnings_time = NULL, earnings_accepted_at = NULL,
                   earnings_date_status = NULL
            WHERE earnings_date_status IN ('estimated', 'confirmed') AND corporate_id IN (SELECT unnest(?))""", [ids])
        result.rows += common.upsert(con, 'fundamentals_quarterly', reported)
        result.rows += common.upsert(con, 'fundamentals_quarterly', upcoming)

        # yearly: Q4 release + 10-K filing date
        con.execute("""
            UPDATE fundamentals_yearly AS y SET
                earnings_date = q.earnings_date, earnings_time = q.earnings_time,
                earnings_accepted_at = q.earnings_accepted_at,
                earnings_date_status = q.earnings_date_status, annual_report_date = y.filed_date
            FROM fundamentals_quarterly q
            WHERE q.corporate_id = y.corporate_id AND q.fiscal_year = y.fiscal_year AND q.fiscal_quarter = 4
              AND y.corporate_id IN (SELECT unnest(?))""", [ids])
        con.execute("""
            UPDATE fundamentals_yearly SET annual_report_date = filed_date
            WHERE annual_report_date IS NULL AND corporate_id IN (SELECT unnest(?))""", [ids])

        result.updated = len(set(reported['corporate_id']) | set(upcoming.get('corporate_id', []))) if not reported.empty else 0
        result.message = f'{len(reported)} reported, {len(upcoming)} upcoming'
        log.info(result.message)
    finally:
        con.close()
    return result
