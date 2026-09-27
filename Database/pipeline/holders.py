"""
Step 'holders': institutional holders from SEC Form 13F data sets
(companies.duckdb: fundamentals_quarterly.top_holders / inst_ownership_pct /
n_institutional_holders, fundamentals_yearly.top_holders).

- SEC publishes 13F data sets per 3-month filing window; late filings and
  amendments of a quarter appear in later windows, so holdings are grouped by
  PERIODOFREPORT across all loaded windows. Each window is downloaded once to
  data/cache/13f/ and converted to Parquet (processed_datasets).
- The latest HOLDERS_FIRST_RUN_QUARTERS + 1 report quarters are loaded; the
  oldest one only provides the change vs the previous quarter. The change is
  empty when the filer's previous report is missing or incomplete (< 10 % of
  its current positions; Norges Bank files full reports only for Q2 / Q4);
  a complete previous report without the stock means a new position.
- Only SH positions (no PRN), no puts / calls.
- Amendments per filer and quarter: the latest original or RESTATEMENT is the
  base (a restatement replaces the original); NEW HOLDINGS amendments filed
  after it are added.
- VALUE is in thousands of dollars for filings before 2023-01-03, in dollars after.
- Companies are matched by CUSIP: the CUSIPs of their US ISINs, widened to
  every common-stock CUSIP of the same issuer (first 6 characters), so all
  share classes are aggregated (GOOGL + GOOG). Common stock = valid CUSIP
  with a numeric issue code below 90 (letters = debt, 90-99 = options) and
  no preferred / note / warrant / option title. Fallback: the CUSIP with the
  most holder rows (>= 20) whose most common 13F issuer name equals the
  company name (normalized), widened only to share classes with the same
  issuer code and name (>= 5 % of its holder rows); then an abbreviation
  match (every company word abbreviated by a 13F word, any order: 'COSTCO
  WHSL', 'DISNEY WALT', 'TEXAS INSTRS') and a fuzzy match (same first word,
  similarity >= 0.85). Only US companies.
- The 13F quarter is mapped to the company's fiscal quarter with the nearest
  period end (max 45 days apart). pct_of_shares_out / inst_ownership_pct use
  the shares outstanding (all classes) from market_cap_daily on that date.
"""

from __future__ import annotations

import difflib
import re
import zipfile
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd

import config
from pipeline import common
from pipeline.companies import _LEGAL, _norm_name


log = common.get_logger('pipeline.holders')

LIST_URL   = 'https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets'
CACHE      = config.CACHE_DIR / '13f'
CHECKS_CSV = config.REVIEW_DIR / 'holders_checks.csv'
MATCH_CSV  = config.REVIEW_DIR / 'holders_matching.csv'
DOLLARS_FROM = date(2023, 1, 3)                   # VALUE in dollars from this filing date on
NOT_COMMON = r'PFD|PREF|NOTE|WT|WARR|UNIT|RIGHT|DEB|BOND|CONV|SUB|DEP|ETN'
MAX_QUARTER_GAP_DAYS = 45
MIN_ROWS_NAME_MATCH  = 20                           # a name-matched CUSIP must have >= 20 holder rows
FUZZY_MIN_SIMILARITY = 0.85
# the one extra word a 13F issuer name may have beyond the company's words
HARMLESS_EXTRA = {'ireland', 'del', 'hldgs', 'hldg', 'cl', 'com', 'inc', 'corp', 'grp', 'intl', 'plc', 'ltd',
                  'usa', 'us', 'the', 'n', 'v', 'sa', 'new'}


def cusip_valid(c: str) -> bool:
    """
    9 characters with a correct check digit (modulus 10 'double add double').
    """
    if not isinstance(c, str) or len(c) != 9 or not c[8].isdigit():
        return False
    total = 0
    for i, ch in enumerate(c[:8]):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - ord('A') + 10
        else:
            v = {'*': 36, '@': 37, '#': 38}.get(ch)
            if v is None:
                return False
        if i % 2:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10 == int(c[8])


# ----
# DATA SETS
# ----

def list_datasets() -> pd.DataFrame:
    """
    Available zip files with their filing window (start, end).
    """
    html = common.http_get(LIST_URL, 'sec').text
    rows = []
    for href in sorted(set(re.findall(r'href="([^"]+form13f[^"]*\.zip)"', html, re.I))):
        name = href.rsplit('/', 1)[1]
        m = re.match(r'(\d{2}[a-z]{3}\d{4})-(\d{2}[a-z]{3}\d{4})_form13f\.zip', name, re.I)
        if m:
            start, end = (datetime.strptime(x, '%d%b%Y').date() for x in m.groups())
        else:
            q = re.match(r'(\d{4})q(\d)_form13f\.zip', name, re.I)
            if not q:
                continue
            y, n = int(q.group(1)), int(q.group(2))
            start = date(y, 3 * n - 2, 1)
            end = (pd.Timestamp(start) + pd.offsets.QuarterEnd(0)).date()
        url = href if href.startswith('http') else 'https://www.sec.gov' + href
        rows.append({'name': name, 'url': url, 'start': start, 'end': end})
    return pd.DataFrame(rows).sort_values('start').reset_index(drop=True)


def report_periods(datasets: pd.DataFrame, n: int) -> list[date]:
    """
    The latest n quarter ends whose regular filing deadline (+45 days) lies in an available window.
    """
    last_end = datasets['end'].max()
    q = (pd.Timestamp(last_end) - pd.offsets.QuarterEnd(1)).date()
    periods = []
    while len(periods) < n:
        due = q + timedelta(days=45)
        if ((datasets['start'] <= due) & (datasets['end'] >= due)).any():
            periods.append(q)
        q = (pd.Timestamp(q) - pd.offsets.QuarterEnd(1)).date()
    return sorted(periods)


def ensure_dataset(con, row) -> None:
    """
    Download a zip once and convert its three tables to Parquet (SH, no options).
    """
    folder = CACHE / row['name'].replace('.zip', '')
    if (folder / 'infotable.parquet').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / row['name']
    if not zpath.exists():
        log.info(f"downloading {row['name']}")
        r = common.http_get(row['url'], 'sec', stream=True)
        with open(zpath, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        for member in ('SUBMISSION.tsv', 'COVERPAGE.tsv', 'INFOTABLE.tsv'):
            # some zips have the files in a subfolder
            name = next(n for n in z.namelist() if n.rsplit('/', 1)[-1].upper() == member.upper())
            with z.open(name) as src, open(folder / member, 'wb') as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    q = lambda p: str(p).replace('\\', '/')
    tsv = lambda n: f"read_csv('{q(folder / n)}', delim='\\t', header=true, all_varchar=true, quote='', ignore_errors=true)"
    duck = duckdb.connect()
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, strptime(FILING_DATE, '%d-%b-%Y')::DATE AS filing_date,
                            SUBMISSIONTYPE AS submission_type, CIK AS filer_cik,
                            strptime(PERIODOFREPORT, '%d-%b-%Y')::DATE AS period
                     FROM {tsv('SUBMISSION.tsv')}) TO '{q(folder / 'submission.parquet')}' (FORMAT parquet)""")
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, ISAMENDMENT AS is_amendment,
                            upper(AMENDMENTTYPE) AS amendment_type, FILINGMANAGER_NAME AS manager
                     FROM {tsv('COVERPAGE.tsv')}) TO '{q(folder / 'coverpage.parquet')}' (FORMAT parquet)""")
    duck.execute(f"""COPY (SELECT ACCESSION_NUMBER AS accession, upper(CUSIP) AS cusip, NAMEOFISSUER AS issuer,
                            upper(TITLEOFCLASS) AS title, TRY_CAST(VALUE AS DOUBLE) AS value,
                            TRY_CAST(SSHPRNAMT AS DOUBLE) AS shares
                     FROM {tsv('INFOTABLE.tsv')}
                     WHERE SSHPRNAMTTYPE = 'SH' AND coalesce(PUTCALL, '') = '')
                  TO '{q(folder / 'infotable.parquet')}' (FORMAT parquet)""")
    duck.close()
    for member in ('SUBMISSION.tsv', 'COVERPAGE.tsv', 'INFOTABLE.tsv'):
        (folder / member).unlink()
    zpath.unlink()                                  # the Parquet files keep what is needed
    common.upsert(con, 'processed_datasets', pd.DataFrame([{
        'dataset': f"13f:{row['name']}", 'processed_at': datetime.now(), 'rows_used': None}]))


# ----
# HOLDINGS
# ----

def holdings(folders: list, periods: list[date]) -> pd.DataFrame:
    """
    One row per (period, filer, CUSIP): shares, value_usd, manager, issuer, title.
    """
    q = lambda p: str(p).replace('\\', '/')
    files = lambda n: '[' + ', '.join(f"'{q(f / n)}'" for f in folders) + ']'
    duck = duckdb.connect()
    duck.register('_periods', pd.DataFrame({'period': periods}))
    df = duck.execute(f"""
        WITH sub AS (SELECT DISTINCT * FROM read_parquet({files('submission.parquet')})),
        cov AS (SELECT DISTINCT * FROM read_parquet({files('coverpage.parquet')})),
        f AS (
            SELECT s.accession, s.filing_date, s.filer_cik, s.period, c.manager,
                   CASE WHEN s.submission_type = '13F-HR' THEN 'original'
                        WHEN c.amendment_type LIKE 'RESTATEMENT%' THEN 'restatement'
                        WHEN c.amendment_type LIKE 'NEW HOLDINGS%' THEN 'new_holdings' END AS kind
            FROM sub s JOIN cov c USING (accession)
            WHERE s.submission_type IN ('13F-HR', '13F-HR/A') AND s.period IN (SELECT period FROM _periods)),
        base AS (
            SELECT * FROM f WHERE kind IN ('original', 'restatement')
            QUALIFY row_number() OVER (PARTITION BY filer_cik, period ORDER BY filing_date DESC, accession DESC) = 1),
        used AS (
            SELECT accession, filing_date, filer_cik, period, manager FROM base
            UNION ALL
            SELECT f.accession, f.filing_date, f.filer_cik, f.period, b.manager FROM f
            JOIN base b USING (filer_cik, period)
            WHERE f.kind = 'new_holdings' AND f.filing_date >= b.filing_date),
        info AS (SELECT DISTINCT * FROM read_parquet({files('infotable.parquet')}))
        SELECT u.period, u.filer_cik, any_value(u.manager) AS manager, i.cusip,
               any_value(i.issuer) AS issuer, any_value(i.title) AS title,
               sum(i.shares) AS shares,
               sum(i.value * CASE WHEN u.filing_date < DATE '{DOLLARS_FROM}' THEN 1000 ELSE 1 END) AS value_usd
        FROM used u JOIN info i USING (accession)
        WHERE i.shares > 0
        GROUP BY u.period, u.filer_cik, i.cusip""").df()
    duck.close()
    return df


def match_companies(con, h: pd.DataFrame, force_name: set | None = None,
                    exclude: set | None = None) -> tuple[dict, pd.DataFrame]:
    """
    {cusip: corporate_id} and a table of how each US company was matched.
    force_name: companies whose CUSIP gave < 5 % ownership (outdated) -> name match only;
    exclude: those outdated CUSIPs, which no name match may pick again.
    """
    force_name = force_name or set()
    exclude = exclude or set()
    comp = con.execute("""SELECT corporate_id, primary_ticker, tickers, name, legal_name, cusip FROM company_info
                          WHERE sec_filer""").df()
    us = set(common.load_ticker_files().query('group_name == @config.US_GROUP')['ticker'])
    comp = comp[comp['tickers'].map(lambda ts: bool(us & set(ts)))]
    pcon = common.connect('prices', read_only=True)
    isins = dict(pcon.execute("SELECT ticker, isin FROM instruments WHERE isin LIKE 'US%'").fetchall())
    pcon.close()

    # per CUSIP: its most common issuer name / title, total value
    cusips = (h.groupby(['cusip', 'issuer', 'title'], dropna=False).agg(n=('shares', 'size'), value=('value_usd', 'sum'))
               .reset_index().sort_values('n', ascending=False))
    cusips = (cusips.groupby('cusip')
                    .agg(issuer=('issuer', 'first'), title=('title', 'first'), value=('value', 'sum'), n=('n', 'sum'))
                    .reset_index())
    latest = h[h['period'] == h['period'].max()].groupby('cusip').size()
    cusips['n_latest'] = cusips['cusip'].map(latest).fillna(0)
    # common stock only: valid 9-character CUSIP, numeric issue code below 90 (letters = debt,
    # 90-99 = options), no preferred / notes / warrants / options in the title
    issue = cusips['cusip'].str[6:8]
    common_stock = (cusips['cusip'].map(cusip_valid) & issue.str.isdigit()
                    & (pd.to_numeric(issue, errors='coerce') < 90)
                    & ~cusips['title'].fillna('').str.contains(NOT_COMMON + r'|CALL|PUT|OPT', regex=True))
    cusips = cusips[common_stock].copy()
    cusips['issuer6'] = cusips['cusip'].str[:6]
    by_issuer6 = cusips.groupby('issuer6')['cusip'].apply(list).to_dict()
    rows_of = cusips.set_index('cusip')['n'].to_dict()
    cusips['norm'] = cusips['issuer'].map(lambda s: ' '.join(_norm_name(s)))
    # name fallback: among CUSIPs with enough holders, the one with the most holder rows per issuer name;
    # widened only to CUSIPs of the same issuer code AND the same name (share classes, not an issuer's ETFs)
    named = cusips[(cusips['n'] >= MIN_ROWS_NAME_MATCH) & (cusips['norm'] != '') & ~cusips['cusip'].isin(exclude)]
    # the CUSIP with the most holders in the latest quarter (an old CUSIP has few or none there)
    by_name = (named.sort_values(['n_latest', 'n'], ascending=False).drop_duplicates('norm')
                    .set_index('norm')['cusip'].to_dict())
    same_name = cusips.groupby(['issuer6', 'norm'])['cusip'].apply(list).to_dict()

    def widen_by_name(cusip, key):
        # other share classes: same issuer code and name, and >= 5 % of the main CUSIP's holder rows
        return {c for c in same_name.get((cusip[:6], key), [cusip]) if rows_of.get(c, 0) >= 0.05 * rows_of.get(cusip, 0)}

    def is_abbrev(t, w):
        # 13F word t abbreviates company word w: equal, or >= 3 letters, same first letter and t's letters in order in w
        if t == w or (len(t) >= 2 and w.startswith(t)):
            return True
        if len(t) < 3 or t[0] != w[0]:
            return False
        it = iter(w)
        return all(ch in it for ch in t)

    def harmless(word):
        # allowed extra 13F word: listed, or a legal form cut off / abbreviated ('CORPORATI', 'HLDNGS', 'PL')
        return word in HARMLESS_EXTRA or any(l.startswith(word) or (len(word) >= 3 and is_abbrev(word, l))
                                            for l in _LEGAL if len(word) >= 2)

    def abbrev(key):
        # every company word must be abbreviated by a 13F word; extra 13F words (max 2) must be harmless
        words = key.split()
        best, best_n = None, 0
        for k, cu in by_name.items():
            tw = k.split()
            if len(tw) > len(words) + 2:
                continue
            used = set()
            for w in words:
                t = next((t for t in tw if t not in used and is_abbrev(t, w)), None)
                if t is None:
                    break
                used.add(t)
            else:
                extra = set(tw) - used
                if all(harmless(x) for x in extra) and rows_of.get(cu, 0) > best_n:
                    best, best_n = k, rows_of.get(cu, 0)
        return best

    def fuzzy(key):
        first = key.split()[0]
        best, score = None, 0.0
        for k in by_name:
            if k.split()[0] == first:
                r = difflib.SequenceMatcher(None, key, k).ratio()
                if r > score:
                    best, score = k, r
        return best if score >= FUZZY_MIN_SIMILARITY else None

    mapping, report, pending = {}, [], []
    for c in comp.itertuples(index=False):                    # pass 0: config overrides
        forced = [cu for t in c.tickers for cu in config.CUSIP_OVERRIDES.get(t, [])]
        if forced:
            for cu in forced:
                mapping[cu] = int(c.corporate_id)
            report.append({'corporate_id': c.corporate_id, 'primary_ticker': c.primary_ticker, 'name': c.name,
                           'method': 'override', 'cusips': ';'.join(sorted(forced))})
    done = {r['corporate_id'] for r in report}
    for c in comp.itertuples(index=False):                    # pass 1: CUSIPs from ISINs
        if c.corporate_id in done:
            continue
        own = set() if int(c.corporate_id) in force_name else (
            {isins[t][2:11] for t in c.tickers if t in isins} | ({c.cusip} if c.cusip else set()))
        found = set()
        for cu in own:
            found |= set(by_issuer6.get(cu[:6], []))
        if found:                                              # other classes need >= 5 % of the main one's rows
            top = max(rows_of.get(cu, 0) for cu in found)
            found = {cu for cu in found if rows_of.get(cu, 0) >= 0.05 * top}
            for cu in found:
                mapping.setdefault(cu, int(c.corporate_id))
            report.append({'corporate_id': c.corporate_id, 'primary_ticker': c.primary_ticker, 'name': c.name,
                           'method': 'cusip', 'cusips': ';'.join(sorted(found))})
        else:
            pending.append(c)
    claimed = set(mapping)
    by_name = {k: v for k, v in by_name.items() if v not in claimed}
    for c in pending:                                          # pass 2: names, unclaimed CUSIPs only
        found, method = set(), None
        if True:
            for nm in (c.name, c.legal_name):
                key = ' '.join(_norm_name(nm))
                if key and key in by_name:
                    found, method = widen_by_name(by_name[key], key), 'name'
                    break
        for tier, finder in (('name_abbrev', abbrev), ('name_fuzzy', fuzzy)):
            if found:
                break
            for nm in (c.name, c.legal_name):
                key = ' '.join(_norm_name(nm))
                match = finder(key) if key else None
                if match:
                    found, method = widen_by_name(by_name[match], match), tier
                    break
        found -= claimed
        for cu in found:
            mapping.setdefault(cu, int(c.corporate_id))
        report.append({'corporate_id': c.corporate_id, 'primary_ticker': c.primary_ticker, 'name': c.name,
                       'method': method or 'unmatched', 'cusips': ';'.join(sorted(found))})
    return mapping, pd.DataFrame(report)


# ----
# PER COMPANY
# ----

def holder_group(name) -> str | None:
    """
    Group of related 13F filers (config.HOLDER_GROUPS), e.g. the Vanguard entities.
    """
    for group, pattern in config.HOLDER_GROUPS:
        if isinstance(name, str) and re.search(pattern, name.upper()):
            return group
    return None


def fiscal_quarter_for(cid, period, fq: pd.DataFrame, fye) -> tuple[int, int] | None:
    """
    Fiscal (year, quarter) for a 13F date: the company's quarter with the nearest
    period end (max 45 days). If its 10-Q is not filed yet, quarters are counted
    on from the latest row, with quarter ends derived from company_info.fiscal_year_end.
    """
    cand = fq[fq['corporate_id'] == cid]
    if cand.empty:
        return None
    gap = (cand['period_end'] - period).abs().dt.days
    if gap.min() <= MAX_QUARTER_GAP_DAYS:
        best = cand.loc[gap.idxmin()]
        return int(best['fiscal_year']), int(best['fiscal_quarter'])
    if not fye:
        return None
    last = cand.sort_values('period_end').iloc[-1]
    fy, q, end = int(last['fiscal_year']), int(last['fiscal_quarter']), last['period_end']
    mm, dd = int(fye[:2]), int(fye[2:])
    month_end = (pd.Timestamp(2001, mm, 1) + pd.offsets.MonthEnd(0)).day == dd
    for _ in range(8):                                    # at most two years ahead
        fy, q = (fy + 1, 1) if q == 4 else (fy, q + 1)
        end = end + pd.DateOffset(months=3)
        if month_end:
            end = end + pd.offsets.MonthEnd(0)
        if abs((end - period).days) <= MAX_QUARTER_GAP_DAYS:
            return fy, q
        if end > period:
            return None
    return None


def compute(con, h: pd.DataFrame, mapping: dict, report: pd.DataFrame, periods: list[date],
            positions: pd.Series, ctx: common.RunContext) -> tuple[pd.DataFrame, int]:
    """
    Rows for fundamentals_quarterly: top_holders, inst_ownership_pct, n_institutional_holders.
    Returns (rows, number of company-quarters without a fiscal quarter).
    """
    if ctx.tickers:
        wanted = {t.upper() for t in ctx.tickers}
        keep = set(report.loc[report['primary_ticker'].str.upper().isin(wanted), 'corporate_id'])
        mapping = {k: v for k, v in mapping.items() if v in keep}
    h = h.assign(corporate_id=h['cusip'].map(mapping)).dropna(subset=['corporate_id'])
    h['corporate_id'] = h['corporate_id'].astype(int)

    # per company, filer and quarter (all share classes together)
    agg = (h.groupby(['corporate_id', 'period', 'filer_cik'])
            .agg(manager=('manager', 'first'), shares=('shares', 'sum'), value_usd=('value_usd', 'sum'))
            .reset_index())
    agg['period'] = pd.to_datetime(agg['period'])
    # change vs the previous quarter; unknown if the filer's previous report is missing or incomplete
    # (< 10 % of its current positions: Norges Bank files full reports only for Q2 / Q4).
    # A complete previous report without the stock = a new position (change = shares).
    agg['prev_period'] = agg['period'] - pd.offsets.QuarterEnd(1)
    prev = agg[['corporate_id', 'filer_cik', 'period', 'shares']].rename(
        columns={'period': 'prev_period', 'shares': 'prev_shares'})
    agg = agg.merge(prev, on=['corporate_id', 'filer_cik', 'prev_period'], how='left')
    n_now = pd.Series(list(zip(agg['filer_cik'], agg['period']))).map(positions)
    n_prev = pd.Series(list(zip(agg['filer_cik'], agg['prev_period']))).map(positions)
    comparable = (n_prev.notna() & (n_prev >= 0.1 * n_now)).values
    agg['change'] = (agg['shares'] - agg['prev_shares'].fillna(0)).where(comparable)

    # shares outstanding (all classes) on the report date
    pcon = common.connect('prices', read_only=True)
    pcon.register('_q', agg[['corporate_id', 'period']].drop_duplicates())
    shares_out = pcon.execute("""
        SELECT q.corporate_id, q.period, m.shares_outstanding FROM _q q
        ASOF LEFT JOIN market_cap_daily m ON m.corporate_id = q.corporate_id AND m.date <= q.period""").df()
    pcon.close()
    shares_out['period'] = pd.to_datetime(shares_out['period'])
    agg = agg.merge(shares_out, on=['corporate_id', 'period'], how='left')
    agg['pct'] = 100 * agg['shares'] / agg['shares_outstanding']

    fq = con.execute("""SELECT corporate_id, fiscal_year, fiscal_quarter, period_end FROM fundamentals_quarterly
                        WHERE period_end IS NOT NULL""").df()
    fq['period_end'] = pd.to_datetime(fq['period_end'])
    fye = dict(con.execute('SELECT corporate_id, fiscal_year_end FROM company_info').fetchall())

    rows, unmapped = [], 0
    shown = [pd.Timestamp(p) for p in periods[1:]]
    for (cid, period), g in agg[agg['period'].isin(shown)].groupby(['corporate_id', 'period']):
        fyq = fiscal_quarter_for(cid, period, fq, fye.get(cid))
        if fyq is None:
            unmapped += 1
            continue
        top = g.sort_values('shares', ascending=False).head(config.HOLDERS_TOP_N)
        so = g['shares_outstanding'].iloc[0]
        rows.append({
            'corporate_id': int(cid), 'fiscal_year': fyq[0], 'fiscal_quarter': fyq[1],
            'top_holders': [{'holder_name': r.manager, 'holder_cik': str(r.filer_cik).lstrip('0'),
                             'holder_group': holder_group(r.manager),
                             'shares': float(r.shares), 'value_usd': float(r.value_usd),
                             'pct_of_shares_out': None if pd.isna(r.pct) else float(r.pct),
                             'change_shares_vs_prev_quarter': None if pd.isna(r.change) else float(r.change)}
                            for r in top.itertuples(index=False)],
            'inst_ownership_pct': 100 * g['shares'].sum() / so if pd.notna(so) and so > 0 else None,
            'n_institutional_holders': int(g['filer_cik'].nunique()),
            'holders_report_date': period.date(),
        })
    return pd.DataFrame(rows), unmapped


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    con = common.connect('companies')
    try:
        datasets = list_datasets()
        periods = report_periods(datasets, config.HOLDERS_FIRST_RUN_QUARTERS + 1)
        needed = datasets[datasets['end'] > periods[0]]
        log.info(f'periods {periods[0]} .. {periods[-1]}; data sets: {", ".join(needed["name"])}')
        for row in needed.to_dict('records'):
            try:
                ensure_dataset(con, row)
            except Exception as e:
                result.fail(row['name'], common.format_error(e), log)
        folders = [CACHE / n.replace('.zip', '') for n in needed['name']
                   if (CACHE / n.replace('.zip', '') / 'infotable.parquet').exists()]

        h = holdings(folders, periods)
        # positions per filer and quarter: tells whether a previous-quarter report is complete
        positions = h.groupby(['filer_cik', 'period']).size()
        positions.index = positions.index.set_levels(pd.to_datetime(positions.index.levels[1]), level=1)

        mapping, report = match_companies(con, h)
        out, unmapped = compute(con, h, mapping, report, periods, positions, ctx)
        # a CUSIP giving < 5 % ownership is outdated (new CUSIP after a re-domiciling, STX): match again by
        # name, without the CUSIPs used so far; the name match takes the CUSIP with the most holders in the
        # latest quarter
        if not out.empty:
            latest = out.sort_values('holders_report_date').groupby('corporate_id').tail(1)
            low = set(latest.loc[latest['inst_ownership_pct'] < 5, 'corporate_id'])
            low -= set(report.loc[report['method'] == 'override', 'corporate_id'])
            if low:
                old = {cu for cu, cid in mapping.items() if cid in low}
                log.info(f'{len(low)} companies with an outdated CUSIP -> matched again by name')
                mapping, report = match_companies(con, h, force_name={int(x) for x in low}, exclude=old)
                out, unmapped = compute(con, h, mapping, report, periods, positions, ctx)
        report.to_csv(MATCH_CSV, index=False)

        if not out.empty:
            result.rows = common.upsert(con, 'fundamentals_quarterly', out)
            ids = out['corporate_id'].unique().tolist()
            con.execute("""
                UPDATE fundamentals_yearly AS y SET top_holders = q.top_holders
                FROM fundamentals_quarterly q
                WHERE q.corporate_id = y.corporate_id AND q.fiscal_year = y.fiscal_year AND q.fiscal_quarter = 4
                  AND q.top_holders IS NOT NULL AND y.corporate_id IN (SELECT unnest(?))""", [ids])
        result.updated = out['corporate_id'].nunique() if not out.empty else 0
        result.failed += report.loc[report['method'] == 'unmatched', 'primary_ticker'].tolist()
        if unmapped:
            log.info(f'{unmapped} company-quarters without a fiscal quarter')

        # plausibility: latest quarter's institutional ownership
        if not out.empty:
            latest = out.sort_values('holders_report_date').groupby('corporate_id').tail(1)
            latest = latest.merge(report[['corporate_id', 'primary_ticker', 'name', 'method', 'cusips']],
                                  on='corporate_id')
            bad = latest[(latest['inst_ownership_pct'] < 5) | (latest['inst_ownership_pct'] > 110)]
            bad[['corporate_id', 'primary_ticker', 'name', 'holders_report_date', 'inst_ownership_pct',
                 'n_institutional_holders', 'method', 'cusips']].to_csv(CHECKS_CSV, index=False)
            log.info(f'{len(bad)} companies with institutional ownership < 5% or > 110% -> {CHECKS_CSV.name}')
        result.message = f"match: {report['method'].value_counts().to_dict()}"
        log.info(result.message)
    finally:
        con.close()
    return result
