"""
Step 'marketcap': market_cap_daily (prices.duckdb), one row per company per
trading day of its primary listing.

  market_cap = shares outstanding (all share classes) x close of the primary listing

Split rule (DECISIONS.md): prices_daily.close is split-adjusted to today's
share basis, SEC share counts are the real counts at their date. Both are
put on today's basis before multiplying:
  shares_today_basis = shares x product of all splits with ex-date after the shares' date
  market_cap         = close (today basis) x shares_today_basis
which equals the real historical price x the real share count, also when a
split falls between the shares' date and the price date. shares_outstanding
is stored as the real count on that day.

Spin-offs (corporate_actions): Yahoo records a spin-off as a fractional
stock_split, which adjusts prices (correct) but must not scale share counts.
Each Yahoo factor is split into split_factor x price_only_factor:
- SEC companies: split_factor = SEC shares after / before the event, rounded
  to a clean ratio (2:1, 1:2, 3:2, ... or 1:1) within SPLIT_RATIO_TOLERANCE.
  No SEC count after the event yet: the current yfinance count. No clean ratio,
  or counts more than SPLIT_SEC_MAX_DAYS away: price-only, listed in
  data/review/corporate_actions_review.csv until CORPORATE_ACTION_OVERRIDES
  decides it. Events before the company's first SEC count: Yahoo's factor.
- Other companies: Yahoo's factor is the split (as before).
Share counts use split factors only; prices use Yahoo's full factor:
  market_cap = close x (Yahoo factors after the day / split factors after the day)
               x shares on today's split basis

Shares:
- SEC filers: cover-page count (or balance-sheet count for multi-class
  companies) from fundamentals, effective from its as-of date and carried
  forward until the next one, at most SHARES_STALE_DAYS; after that the
  current yfinance count. Glitches (0, > 2x off the neighbours) are dropped.
  Before the first SEC count the earliest count is extended backwards in
  split-adjusted terms (shares_source = 'sec_backfilled'). A jump of more than
  20 % between two counts with a completed acquisition (8-K item 2.01) in
  between applies from the 8-K date (shares_source = 'sec_after_deal').
  MARKETCAP_YFINANCE_ONLY: primary tickers whose SEC counts are for another
  share class (BRK-B) use yfinance only.
- Others (and SEC filers without any count): yfinance impliedSharesOutstanding
  (all classes; falls back to sharesOutstanding), current value only, applied
  to the whole price history (shares_source = 'yfinance_current').
The table is rebuilt completely on every run (a few seconds in DuckDB).
"""

from __future__ import annotations

import concurrent.futures as cf
from datetime import datetime

import pandas as pd
import yfinance as yf

import config
from pipeline import common


log = common.get_logger('pipeline.marketcap')


# ----
# YFINANCE SHARES
# ----

def fetch_yf_shares(ticker: str) -> float | None:
    common.limiter('yfinance').wait()
    info = yf.Ticker(ticker).info or {}
    return info.get('impliedSharesOutstanding') or info.get('sharesOutstanding')


def ensure_yf_shares(con, corporate_ids: list[int]) -> list[str]:
    """
    Fetch yfinance shares for companies that need them and have none stored.
    Returns tickers that failed.
    """
    todo = con.execute("""
        SELECT corporate_id, primary_ticker FROM company_info
        WHERE corporate_id IN (SELECT unnest(?)) AND yf_shares_outstanding IS NULL""",
        [corporate_ids]).fetchall()
    if not todo:
        return []
    log.info(f'fetching yfinance shares for {len(todo)} companies')
    rows, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=config.YF_WORKERS) as ex:
        futures = {ex.submit(fetch_yf_shares, t): (cid, t) for cid, t in todo}
        for f in cf.as_completed(futures):
            cid, t = futures[f]
            try:
                shares = f.result()
            except Exception:
                shares = None
            if shares:
                rows.append({'corporate_id': cid, 'yf_shares_outstanding': float(shares),
                             'yf_shares_date': datetime.now().date()})
            else:
                failed.append(t)
    common.upsert(con, 'company_info', pd.DataFrame(rows))
    return failed


# ----
# CORPORATE ACTIONS (splits vs spin-offs)
# ----

CLEAN_RATIOS = sorted({p / q for p in range(1, 11) for q in range(1, 11)}
                      | {float(n) for n in range(1, 51)} | {1 / n for n in range(1, 51)})
ACTIONS_REVIEW_CSV = config.REVIEW_DIR / 'corporate_actions_review.csv'


def clean_ratio(r: float, tol: float = None) -> float | None:
    """
    The clean split ratio (p:q with p, q <= 10, or n:1 / 1:n up to 50, incl. 1:1) within `tol`, else None.
    """
    tol = config.SPLIT_RATIO_TOLERANCE if tol is None else tol
    if not (r and r > 0):
        return None
    best = min(CLEAN_RATIOS, key=lambda c: abs(r / c - 1))
    return best if abs(r / best - 1) <= tol else None


SEC_SHARES_SQL = """
    -- one count per company and as-of date. basis = the date whose split basis the count is on:
    -- cover-page counts are real at their as-of date; balance-sheet counts are restated for splits
    -- up to the filing date (a 10-Q filed after a split shows the period-end count post-split, GOOGL 2014)
    SELECT corporate_id, shares_as_of AS d, max(shares_outstanding) AS shares,
           CASE WHEN bool_or(shares_as_of > period_end) THEN shares_as_of
                ELSE coalesce(arg_max(filed_date, shares_outstanding), shares_as_of) END AS basis,
           CASE WHEN bool_or(shares_as_of > period_end) THEN 'sec_cover_page' ELSE 'sec_balance_sheet' END AS src
    FROM (SELECT corporate_id, shares_as_of, shares_outstanding, period_end, filed_date FROM c.fundamentals_quarterly
          UNION ALL
          SELECT corporate_id, shares_as_of, shares_outstanding, period_end, filed_date FROM c.fundamentals_yearly)
    WHERE shares_as_of IS NOT NULL GROUP BY 1, 2"""


def classify_events(events: pd.DataFrame, sec: pd.DataFrame, yf_shares: dict,
                    overrides: dict | None = None, max_days: int | None = None) -> pd.DataFrame:
    """
    Pure function. events: ticker, date, corporate_id (NaN = no SEC data), yahoo_factor.
    sec: corporate_id, d, shares (real SEC counts). yf_shares: corporate_id -> (shares, date).
    Returns one row per event with split_factor, price_only_factor, method, the SEC counts used, note
    and review (True = listed in the review file).
    """
    overrides = config.CORPORATE_ACTION_OVERRIDES if overrides is None else overrides
    max_gap = pd.Timedelta(days=config.SPLIT_SEC_MAX_DAYS if max_days is None else max_days)
    sec = sec[sec['shares'] >= 1e4].copy()
    sec['d'] = pd.to_datetime(sec['d'])
    sec['basis'] = pd.to_datetime(sec['basis']) if 'basis' in sec else sec['d']
    by_cid = {cid: g.sort_values('d') for cid, g in sec.groupby('corporate_id')}
    rows = []
    for e in events.itertuples(index=False):
        date, y = pd.Timestamp(e.date), float(e.yahoo_factor)
        row = {'ticker': e.ticker, 'date': date.date(), 'corporate_id': e.corporate_id, 'yahoo_factor': y,
               'split_factor': y, 'method': 'no_sec_data', 'sec_before_date': None, 'sec_before_shares': None,
               'sec_after_date': None, 'sec_after_shares': None, 'note': None, 'review': False}
        g = by_cid.get(e.corporate_id) if pd.notna(e.corporate_id) else None
        key = (e.ticker, str(date.date()))
        if key in overrides:
            row.update(split_factor=float(overrides[key]), method='override', note='CORPORATE_ACTION_OVERRIDES')
        elif g is not None:
            # before: the last count still on the pre-event basis; after: the first one on the post-event
            # basis (may be dated before the event when restated in a filing after it, GOOGL 2014-03-31)
            before = g[(g['d'] < date) & (g['basis'] < date)].tail(1)
            after = g[g['basis'] >= date].head(1)
            if before.empty:
                if clean_ratio(y) is None:
                    row.update(review=True, note='before the first SEC count: Yahoo factor used as split')
                else:
                    row['note'] = 'before the first SEC count'
                rows.append(row)
                continue
            row.update(sec_before_date=before['d'].iloc[0].date(), sec_before_shares=float(before['shares'].iloc[0]))
            method = 'sec_counts'
            if after.empty:
                shares, d = yf_shares.get(e.corporate_id, (None, None))
                if shares and d is not None and pd.Timestamp(d) >= date:
                    after = pd.DataFrame({'d': [pd.Timestamp(d)], 'basis': [pd.Timestamp(d)], 'shares': [float(shares)]})
                    method = 'yfinance_after'
            if after.empty:
                row.update(split_factor=1.0, method='price_only_review', review=True,
                           note='no share count after the event yet')
            else:
                row.update(sec_after_date=after['d'].iloc[0].date(), sec_after_shares=float(after['shares'].iloc[0]))
                ratio = row['sec_after_shares'] / row['sec_before_shares']
                c = clean_ratio(ratio)
                # distance: the as-of or filing date, whichever is nearer to the event
                gap_before = date - max(before['d'].iloc[0], before['basis'].iloc[0])
                gap_after = min(after['d'].iloc[0], after['basis'].iloc[0]) - date
                far = gap_before > max_gap or gap_after > max_gap
                if far and c is not None and c != 1.0 and c == clean_ratio(y):
                    # far-away counts, but they confirm Yahoo's own clean split ratio (CSX 3:1, V 4:1, SMCI 10:1)
                    row.update(split_factor=c, method=method,
                               note=f'share counts more than {max_gap.days} days away, ratio confirms the split')
                elif far:
                    row.update(split_factor=1.0, method='price_only_review', review=True,
                               note=f'share counts more than {max_gap.days} days from the event (ratio {ratio:.4f})')
                elif c is None:
                    row.update(split_factor=1.0, method='price_only_review', review=True,
                               note=f'no clean ratio for shares after / before = {ratio:.4f}')
                else:
                    row.update(split_factor=c, method=method)
        rows.append(row)
    out = pd.DataFrame(rows, columns=['ticker', 'date', 'corporate_id', 'yahoo_factor', 'split_factor', 'method',
                                      'sec_before_date', 'sec_before_shares', 'sec_after_date', 'sec_after_shares',
                                      'note', 'review'])
    out['price_only_factor'] = out['yahoo_factor'] / out['split_factor']
    return out


def build_corporate_actions(pcon) -> pd.DataFrame:
    """
    Classify all of Yahoo's split events; rewrite corporate_actions and the review file.
    """
    events = pcon.execute("""
        SELECT p.ticker, p.date, i.corporate_id, p.stock_splits AS yahoo_factor
        FROM prices_daily p LEFT JOIN instruments i USING (ticker)
        WHERE p.stock_splits > 0 AND p.stock_splits <> 1""").df()
    events.loc[events['ticker'].isin(config.MARKETCAP_YFINANCE_ONLY), 'corporate_id'] = None
    sec = pcon.execute(SEC_SHARES_SQL).df()
    yf_shares = {cid: (s, d) for cid, s, d in pcon.execute("""
        SELECT corporate_id, yf_shares_outstanding, yf_shares_date FROM c.company_info
        WHERE yf_shares_outstanding > 0""").fetchall()}
    ca = classify_events(events, sec, yf_shares)
    pcon.execute('DELETE FROM corporate_actions')
    common.upsert(pcon, 'corporate_actions', ca.drop(columns='review'))
    ca[ca['review'].astype(bool)].drop(columns='review').sort_values(['date', 'ticker']).to_csv(ACTIONS_REVIEW_CSV, index=False)
    return ca


# ----
# CALCULATION
# ----

def clean_sec_shares(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Drop glitches in a company's SEC share counts (on today's split basis):
    counts below 10,000, counts more than 100x off the company's median, and
    points more than 2x off the median of their
    neighbours (cover pages tagged in thousands, etc.). The previous count
    is then carried forward instead.
    """
    ev = ev[ev['shares_today'] >= 1e4].sort_values(['corporate_id', 'd'])      # placeholders like 1 share (FOXA 2019)
    # orders of magnitude off the company's overall median (runs of mis-scaled values)
    overall = ev.groupby('corporate_id')['shares_today'].transform('median')
    ev = ev[(ev['shares_today'] / overall).between(0.01, 100)]
    med = ev.groupby('corporate_id')['shares_today'].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).median())
    ratio = ev['shares_today'] / med
    return ev[(ratio > 0.5) & (ratio < 2.0)]


def share_events(pcon) -> pd.DataFrame:
    """
    Per company the dates from which a share count applies, already on today's
    split basis: corporate_id, d, shares_today, src.
    """
    comp = pcon.execute("""
        SELECT corporate_id, primary_ticker, yf_shares_outstanding FROM c.company_info
        WHERE primary_ticker IS NOT NULL AND coalesce(status, 'active') = 'active'""").df()
    sec = pcon.execute(SEC_SHARES_SQL).df()
    # real splits only (spin-offs recorded by Yahoo as splits are price-only)
    splits = pcon.execute("""
        SELECT ticker, date, split_factor AS ratio FROM corporate_actions WHERE abs(split_factor - 1) > 1e-9""").df()
    last_price = dict(pcon.execute('SELECT ticker, max(date) FROM prices_daily GROUP BY 1').fetchall())

    sec = sec.merge(comp[['corporate_id', 'primary_ticker']], on='corporate_id')
    sec = sec[~sec['primary_ticker'].isin(config.MARKETCAP_YFINANCE_ONLY)]
    sec['d'] = pd.to_datetime(sec['d'])
    sec['basis'] = pd.to_datetime(sec['basis'])
    splits['date'] = pd.to_datetime(splits['date'])
    by_ticker = {t: g for t, g in splits.groupby('ticker')}

    def factor(ticker, d):
        g = by_ticker.get(ticker)
        return float(g.loc[g['date'] > d, 'ratio'].prod()) if g is not None else 1.0

    sec['shares_today'] = [s * factor(t, b) for s, t, b in zip(sec['shares'], sec['primary_ticker'], sec['basis'])]
    sec = clean_sec_shares(sec)

    # a jump of > 20 % between two counts with a completed acquisition in between (8-K item 2.01):
    # the new count applies from the 8-K, not only from the next cover page (DVN 2026-05-07)
    deals = pcon.execute("""
        SELECT corporate_id, coalesce(CAST(acceptance_time AS DATE), filed_date) AS d FROM c.sec_filings
        WHERE form_type = '8-K' AND items LIKE '%2.01%'""").df()
    deals['d'] = pd.to_datetime(deals['d'])
    by_cid = {cid: g['d'].sort_values() for cid, g in deals.groupby('corporate_id')}
    extra = []
    for cid, g in sec.sort_values('d').groupby('corporate_id'):
        prev = None
        for r in g.itertuples(index=False):
            if prev is not None and not (0.83 <= r.shares_today / prev.shares_today <= 1.2):
                dd = by_cid.get(cid)
                if dd is not None:
                    inside = dd[(dd > prev.d) & (dd < r.d)]
                    if not inside.empty:
                        extra.append({'corporate_id': cid, 'd': inside.iloc[-1], 'shares_today': r.shares_today,
                                      'src': 'sec_after_deal', 'primary_ticker': r.primary_ticker})
            prev = r
    if extra:
        sec = pd.concat([sec, pd.DataFrame(extra)], ignore_index=True)

    # before the first SEC count: the earliest SEC count, extended backwards (already on today's basis)
    first = sec.sort_values('d').drop_duplicates('corporate_id', keep='first')
    backfill = first.assign(d=pd.Timestamp('1900-01-01'), src='sec_backfilled')
    events = [sec[['corporate_id', 'd', 'shares_today', 'src']], backfill[['corporate_id', 'd', 'shares_today', 'src']]]
    yf_rows = []
    stale = pd.Timedelta(days=config.SHARES_STALE_DAYS)
    last_sec = sec.groupby('corporate_id')['d'].max()
    for cid, ticker, yf_shares in zip(comp['corporate_id'], comp['primary_ticker'], comp['yf_shares_outstanding']):
        if not (pd.notna(yf_shares) and yf_shares > 0):
            continue
        if cid not in last_sec.index:
            yf_rows.append((cid, pd.Timestamp('1900-01-01'), yf_shares))          # whole history
        elif ticker in last_price and pd.Timestamp(last_price[ticker]) > last_sec[cid] + stale:
            yf_rows.append((cid, last_sec[cid] + stale, yf_shares))               # after the SEC count got stale
    events.append(pd.DataFrame([(c, d, s, 'yfinance_current') for c, d, s in yf_rows],
                               columns=['corporate_id', 'd', 'shares_today', 'src']))
    ev = pd.concat(events, ignore_index=True)
    ev['d'] = pd.to_datetime(ev['d']).dt.date
    return ev


def build(pcon) -> int:
    """
    Recreate market_cap_daily from prices (main database) and companies
    (ATTACHed as c). Returns the row count.
    """
    build_corporate_actions(pcon)
    units = pd.DataFrame([(q, m, float(d)) for q, (m, d) in config.MINOR_UNITS.items()],
                         columns=['quoted', 'major', 'divisor'])
    pcon.register('_units', units)
    pcon.register('_events', share_events(pcon))
    try:
        pcon.execute('DELETE FROM market_cap_daily')
        pcon.execute("""
            INSERT INTO market_cap_daily
            WITH comp AS (SELECT corporate_id, primary_ticker FROM c.company_info
                          WHERE primary_ticker IS NOT NULL AND coalesce(status, 'active') = 'active'),
            px AS (
                SELECT comp.corporate_id, p.ticker, p.date, p.close, p.fx_to_usd,
                       coalesce(u.major, upper(p.currency)) AS currency, coalesce(u.divisor, 1.0) AS divisor,
                       -- products of the factors after this date: real price = close x yahoo_after,
                       -- real shares = shares on today's split basis / split_after
                       coalesce(exp(sum(ln(coalesce(ca.yahoo_factor, 1.0)))
                           OVER (PARTITION BY p.ticker ORDER BY p.date DESC
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), 1.0) AS yahoo_after,
                       coalesce(exp(sum(ln(coalesce(ca.split_factor, 1.0)))
                           OVER (PARTITION BY p.ticker ORDER BY p.date DESC
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)), 1.0) AS split_after
                FROM prices_daily p
                JOIN comp ON comp.primary_ticker = p.ticker
                LEFT JOIN corporate_actions ca ON ca.ticker = p.ticker AND ca.date = p.date
                LEFT JOIN _units u ON u.quoted = p.currency
                WHERE p.close IS NOT NULL)
            SELECT px.corporate_id, px.date, px.ticker AS primary_ticker,
                   e.shares_today / px.split_after AS shares_outstanding,
                   px.close * px.yahoo_after / px.split_after * e.shares_today / px.divisor AS market_cap,
                   px.currency,
                   px.close * px.yahoo_after / px.split_after * px.fx_to_usd * e.shares_today AS market_cap_usd,
                   e.src AS shares_source
            FROM px ASOF JOIN _events e ON e.corporate_id = px.corporate_id AND px.date >= e.d""")
    finally:
        pcon.unregister('_units')
        pcon.unregister('_events')
    return pcon.execute('SELECT count(*) FROM market_cap_daily').fetchone()[0]


CHECKS_CSV = config.REVIEW_DIR / 'marketcap_checks.csv'


JUMP_LIMIT = 0.30


def run_checks(pcon) -> pd.DataFrame:
    """
    Two checks -> marketcap_checks.csv (column `check`):
    - shares_vs_yfinance: latest share count used vs yfinance's current count, > 20 % off.
    - daily_jump: market cap changes by more than JUMP_LIMIT from one trading day to the next,
      except on days with a corporate action of the primary ticker (split / spin-off, corporate_actions).
      Catches wrong share counts, restatements and price glitches.
    """
    shares = pcon.execute("""
        SELECT m.corporate_id, m.primary_ticker, m.date, m.market_cap, m.currency, m.shares_source,
               m.shares_outstanding, c.yf_shares_outstanding
        FROM market_cap_daily m JOIN c.company_info c USING (corporate_id)
        QUALIFY row_number() OVER (PARTITION BY m.corporate_id ORDER BY m.date DESC) = 1""").df()
    shares = shares.dropna(subset=['yf_shares_outstanding'])
    shares['ratio_vs_yfinance'] = shares['shares_outstanding'] / shares['yf_shares_outstanding']
    shares = shares[(shares['ratio_vs_yfinance'] - 1).abs() > 0.2].assign(check='shares_vs_yfinance')

    jumps = pcon.execute(f"""
        WITH m AS (
            SELECT corporate_id, primary_ticker, date, market_cap, currency, shares_source, shares_outstanding,
                   lag(market_cap) OVER w AS prev_market_cap, lag(date) OVER w AS prev_date,
                   lag(shares_outstanding) OVER w AS prev_shares, lag(shares_source) OVER w AS prev_shares_source
            FROM market_cap_daily WINDOW w AS (PARTITION BY corporate_id ORDER BY date))
        SELECT m.*, market_cap / prev_market_cap - 1 AS change,
               shares_outstanding / prev_shares - 1 AS shares_change,     -- ~0: price move, else share count
               CASE WHEN abs(shares_outstanding / prev_shares - 1) < 0.05 THEN 'price' ELSE 'shares' END AS driver
        FROM m
        WHERE prev_market_cap > 0 AND abs(market_cap / prev_market_cap - 1) > {JUMP_LIMIT}
          AND NOT EXISTS (SELECT 1 FROM corporate_actions ca
                          WHERE ca.ticker = m.primary_ticker AND ca.date = m.date)""").df()
    jumps = jumps.assign(check='daily_jump')

    out = pd.concat([shares, jumps], ignore_index=True)
    first = ['check', 'corporate_id', 'primary_ticker', 'date']
    out = out[first + [c for c in out.columns if c not in first]].sort_values(['check', 'primary_ticker', 'date'])
    out.to_csv(CHECKS_CSV, index=False)
    return out


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    con = common.connect('companies')
    try:
        comp = con.execute("""
            SELECT ci.corporate_id, ci.primary_ticker, ci.sec_filer,
                   EXISTS (SELECT 1 FROM fundamentals_quarterly f
                           WHERE f.corporate_id = ci.corporate_id AND f.shares_outstanding > 0) AS has_sec_shares
            FROM company_info ci WHERE coalesce(ci.status, 'active') = 'active'""").df()
        # all companies: fallback for stale SEC counts and the plausibility check
        for t in ensure_yf_shares(con, comp['corporate_id'].astype(int).tolist()):
            result.fail(t, 'no shares outstanding from yfinance', log)
    finally:
        con.close()

    pcon = common.connect('prices')
    try:
        common.attach(pcon, 'companies', 'c')
        result.rows = build(pcon)
        stats = pcon.execute("""
            SELECT shares_source, count(DISTINCT corporate_id) FROM market_cap_daily GROUP BY 1""").fetchall()
        log.info('companies by shares source: ' + ', '.join(f'{s}: {n}' for s, n in stats))
        result.updated = pcon.execute('SELECT count(DISTINCT corporate_id) FROM market_cap_daily').fetchone()[0]
        checks = run_checks(pcon)
        n = checks['check'].value_counts()
        log.info(f"checks -> {CHECKS_CSV.name}: {n.get('shares_vs_yfinance', 0)} companies whose latest share count "
                 f"is > 20% off yfinance, {n.get('daily_jump', 0)} days with a market cap change > "
                 f"{JUMP_LIMIT:.0%} without a corporate action")
        pcon.execute('DETACH c')
    finally:
        pcon.close()
    return result
