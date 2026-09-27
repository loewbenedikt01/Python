"""
Step 'intraday': 1-minute bars from Alpaca for the US tickers (ticker_us),
with 5m / 10m / 1h bars resampled from them. Parquet under data/intraday/:

    <interval>/ticker=<TICKER>/<year>.parquet      interval = 1m, 5m, 10m, 1h

Columns: ts (UTC, start of the bar), open, high, low, close, volume, vwap,
trade_count; 'ticker' comes from the folder name (read_parquet(...,
hive_partitioning = true)).

- Feed: SIP (all US exchanges) is tried first, IEX if SIP is refused. The
  last 16 minutes are never requested (SIP is delayed 15 minutes on the free plan).
- Split-adjusted only (Alpaca adjustment='split'), not for dividends. When a
  split happens after the ticker's history was downloaded, the whole history
  is downloaded again (intraday_state.split_basis_date).
- Regular trading hours only (config.REGULAR_HOURS_ONLY): 09:30 up to and
  including the closing-auction minute, 16:00 New York time (13:00 on
  early-close days: 3 July, the day after Thanksgiving, 24 December when they
  are weekdays). The 16:00 bar holds the closing cross (10-20 % of the day's
  volume) and the official close; without it the daily volume was only
  70-80 % of prices_daily. Early-close days simply have fewer bars.
- Resampled bars are anchored at the open: 5m 09:30, 09:35, ...; 10m 09:30,
  09:40, ...; 1h 09:30, 10:30, ..., 15:30 (the last hour bar is 15:30-16:00).
  The closing-auction minute is folded into the session's last bar, so there
  is no extra bar at 16:00 and the last bar closes at the official close.
  Only minutes with trades exist, so no empty bars are created: open = first,
  close = last, high / low = max / min, volume and trade_count summed, VWAP
  volume-weighted.
- Incremental per ticker from its last stored minute. First run:
  config.INTRADAY_HISTORY_DAYS.
- Check against prices_daily per ticker and day: intraday high / low more than
  2 % off, or the day's volume ratio (intraday / daily) more than 15 % off the
  ticker's median ratio -> data/review/intraday_checks.csv.
"""

from __future__ import annotations

import concurrent.futures as cf
import shutil
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import requests

import config
from pipeline import common


log = common.get_logger('pipeline.intraday')

API = 'https://data.alpaca.markets/v2/stocks/bars'
BASE = config.INTRADAY_DIR
DERIVED_MINUTES = {'5m': 5, '10m': 10, '1h': 60}
CHECKS_CSV = config.REVIEW_DIR / 'intraday_checks.csv'
PRICE_TOL, VOLUME_TOL = 0.02, 0.15
NY = 'America/New_York'
INTRADAY_WORKERS = 4          # parallel tickers; Alpaca's 200 requests / minute limit is shared


# ----
# DOWNLOAD
# ----

def _headers() -> dict:
    return {'APCA-API-KEY-ID': common.get_key('ALPACA_KEY_ID'), 'APCA-API-SECRET-KEY': common.get_key('ALPACA_SECRET')}


def fetch_bars(ticker: str, start: datetime, end: datetime, feed: str) -> pd.DataFrame:
    """
    1-minute bars (split-adjusted) for one ticker, all pages.
    """
    rows, token = [], None
    symbol = ticker.replace('-', '.')                   # Alpaca: BF.B, BRK.B (Yahoo: BF-B)
    while True:
        params = {'symbols': symbol, 'timeframe': '1Min', 'adjustment': 'split', 'feed': feed, 'limit': 10000,
                  'start': start.strftime('%Y-%m-%dT%H:%M:%SZ'), 'end': end.strftime('%Y-%m-%dT%H:%M:%SZ')}
        if token:
            params['page_token'] = token
        data = common.http_get(API, 'alpaca', params=params, headers=_headers()).json()
        rows += (data.get('bars') or {}).get(symbol, [])
        token = data.get('next_page_token')
        if not token:
            break
    if not rows:
        return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'])
    df = pd.DataFrame(rows).rename(columns={'t': 'ts', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
                                            'v': 'volume', 'vw': 'vwap', 'n': 'trade_count'})
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df[['ts', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']]


def session_close_minute(dates) -> pd.Series:
    """
    Minutes after midnight (New York) of the close for each date: 16:00, or 13:00 on early-close days.
    """
    d = pd.to_datetime(pd.Series(dates))
    thanksgiving_friday = (d.dt.month == 11) & (d.dt.weekday == 4) & d.dt.day.between(23, 29)
    early = (((d.dt.month == 7) & (d.dt.day == 3)) | ((d.dt.month == 12) & (d.dt.day == 24)) | thanksgiving_friday)
    early &= d.dt.weekday < 5
    return pd.Series(16 * 60, index=d.index).where(~early, 13 * 60)


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep bars from 09:30 up to and including the closing-auction minute (16:00, 13:00 on early-close days).
    """
    if df.empty:
        return df
    local = df['ts'].dt.tz_convert(NY)
    minutes = (local.dt.hour * 60 + local.dt.minute).values
    close = session_close_minute(local.dt.date.values).values
    return df[(minutes >= 9 * 60 + 30) & (minutes <= close)]


# ----
# RESAMPLING
# ----

def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Bars of `minutes` anchored at the 09:30 open, built only from existing 1-minute bars.
    """
    if df.empty:
        return df.copy()
    d = df.sort_values('ts').copy()
    local = d['ts'].dt.tz_convert(NY)
    minute = local.dt.hour * 60 + local.dt.minute
    since_open = minute - (9 * 60 + 30)
    # the closing-auction minute (16:00 / 13:00) belongs to the session's last bar
    close = pd.Series(session_close_minute(local.dt.date.values).values, index=d.index)
    since_open = since_open.where(minute != close, since_open - 1)
    open_ts = local.dt.normalize() + pd.Timedelta(minutes=9 * 60 + 30)
    d['bucket'] = (open_ts + pd.to_timedelta((since_open // minutes) * minutes, unit='m')).dt.tz_convert('UTC')
    d['pv'] = d['vwap'] * d['volume']
    g = d.groupby('bucket', sort=True)
    out = pd.DataFrame({
        'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(), 'close': g['close'].last(),
        'volume': g['volume'].sum(), 'trade_count': g['trade_count'].sum(), 'pv': g['pv'].sum(),
        'vwap_mean': g['vwap'].mean()})
    out['vwap'] = (out['pv'] / out['volume']).where(out['volume'] > 0, out['vwap_mean'])
    out = out.drop(columns=['pv', 'vwap_mean']).reset_index().rename(columns={'bucket': 'ts'})
    return out[['ts', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']]


# ----
# FILES
# ----

def _path(interval: str, ticker: str, year: int):
    return BASE / interval / f'ticker={ticker}' / f'{year}.parquet'


def read_year(interval: str, ticker: str, year: int) -> pd.DataFrame:
    p = _path(interval, ticker, year)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def write_minutes(ticker: str, new: pd.DataFrame) -> int:
    """
    Merge new 1-minute bars into the yearly files and rebuild the derived files of those years.
    """
    if new.empty:
        return 0
    for year, part in new.groupby(new['ts'].dt.tz_convert(NY).dt.year):
        old = read_year('1m', ticker, year)
        merged = pd.concat([old, part]) if not old.empty else part
        merged = merged.drop_duplicates('ts', keep='last').sort_values('ts').reset_index(drop=True)
        p = _path('1m', ticker, year)
        p.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(p, index=False)
        for interval, minutes in DERIVED_MINUTES.items():
            q = _path(interval, ticker, year)
            q.parent.mkdir(parents=True, exist_ok=True)
            resample(merged, minutes).to_parquet(q, index=False)
    return len(new)


def delete_ticker(ticker: str) -> None:
    for interval in ['1m', *DERIVED_MINUTES]:
        shutil.rmtree(BASE / interval / f'ticker={ticker}', ignore_errors=True)


def last_minute(ticker: str):
    folder = BASE / '1m' / f'ticker={ticker}'
    files = sorted(folder.glob('*.parquet')) if folder.exists() else []
    if not files:
        return None
    return pd.read_parquet(files[-1], columns=['ts'])['ts'].max()


# ----
# CHECK AGAINST DAILY
# ----

def check_against_daily(tickers: list[str]) -> pd.DataFrame:
    pattern = str(BASE / '1m' / 'ticker=*' / '*.parquet').replace('\\', '/')
    con = duckdb.connect()
    common.attach(con, 'prices', 'p')
    df = con.execute(f"""
        WITH m AS (
            SELECT ticker, CAST(timezone('{NY}', ts) AS DATE) AS date,
                   max(high) AS i_high, min(low) AS i_low, sum(volume) AS i_volume, count(*) AS minutes
            FROM read_parquet('{pattern}', hive_partitioning = true)
            WHERE ticker IN (SELECT unnest(?)) GROUP BY 1, 2)
        SELECT m.*, d.high AS d_high, d.low AS d_low, d.volume AS d_volume
        FROM m JOIN p.prices_daily d ON d.ticker = m.ticker AND d.date = m.date""", [tickers]).df()
    con.close()
    df = flag_deviations(df)
    df[df['flagged']].drop(columns='flagged').sort_values(['ticker', 'date']).to_csv(CHECKS_CSV, index=False)
    return df


def flag_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price: intraday high / low vs daily more than PRICE_TOL off. Volume: the day's ratio intraday / daily
    volume vs the ticker's own median ratio more than VOLUME_TOL off (daily volume includes pre- and
    after-market trading, so a normal ratio is below 1 and differs per ticker).
    """
    df = df.copy()
    df['high_dev'] = (df['i_high'] / df['d_high'] - 1).abs()
    df['low_dev'] = (df['i_low'] / df['d_low'] - 1).abs()
    df['volume_ratio'] = (df['i_volume'] / df['d_volume']).where(df['d_volume'] > 0)
    df['median_ratio'] = df.groupby('ticker')['volume_ratio'].transform('median')
    df['volume_dev'] = (df['volume_ratio'] / df['median_ratio'] - 1).abs()
    df['flagged'] = (df['high_dev'] > PRICE_TOL) | (df['low_dev'] > PRICE_TOL) | (df['volume_dev'] > VOLUME_TOL)
    return df


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    files = common.load_ticker_files()
    tickers = ctx.selected(files.loc[files['group_name'] == config.US_GROUP, 'ticker'].drop_duplicates())

    now = datetime.now(timezone.utc)
    end = now - timedelta(minutes=16)
    first_start = now - timedelta(days=config.INTRADAY_HISTORY_DAYS)
    pcon = common.connect('prices')
    try:
        state = {t: (b, f) for t, b, f in pcon.execute(
            'SELECT ticker, split_basis_date, feed FROM intraday_state').fetchall()}
        # only real splits (Alpaca adjusts those); spin-offs that Yahoo records as a split are price-only
        # (corporate_actions, from the marketcap step); tickers not classified there: Yahoo's splits
        splits = pcon.execute("""
            SELECT ticker, max(date) FILTER (WHERE abs(split_factor - 1) > 1e-9) FROM corporate_actions GROUP BY 1
            UNION ALL
            SELECT ticker, max(date) FROM prices_daily
            WHERE stock_splits > 0 AND stock_splits <> 1
              AND ticker NOT IN (SELECT ticker FROM corporate_actions) GROUP BY 1""").fetchall()
        last_split = dict(splits)
        status = dict(pcon.execute('SELECT ticker, status FROM instruments').fetchall())
    finally:
        pcon.close()

    def one(t):
        """Update one ticker; returns (state row or None, minutes written, skipped)."""
        if status.get(t) in ('removed', 'no_data'):
            return None, 0, True
        basis, feed = state.get(t, (None, None))
        split = last_split.get(t)
        if basis is not None and split is not None and split > basis:
            log.info(f'{t}: split on {split} after the intraday history was downloaded -> full reload')
            delete_ticker(t)
            basis = None
        last = last_minute(t) if basis is not None else None
        start = (last + timedelta(minutes=1)).to_pydatetime() if last is not None else first_start
        if start >= end:
            return None, 0, True
        bars, used = None, None
        for f in ([feed] if feed else []) + [x for x in config.ALPACA_FEEDS if x != feed]:
            try:
                bars, used = fetch_bars(t, start, end, f), f
                break
            except requests.HTTPError as e:
                # only a refused subscription (403 / 422) falls back to the next feed; network errors or
                # other responses fail the ticker (retried next run) instead of silently storing IEX data
                code = getattr(e.response, 'status_code', None)
                if code not in (403, 422) or f == config.ALPACA_FEEDS[-1]:
                    raise
                log.warning(f'{t}: feed {f} refused (HTTP {code}), trying the next')
        if config.REGULAR_HOURS_ONLY:
            bars = regular_hours(bars)
        n = write_minutes(t, bars)
        row = {'ticker': t, 'split_basis_date': basis or now.date(), 'feed': used,
               'last_ts': (bars['ts'].max() if n else last), 'updated_at': datetime.now()}
        return row, n, False

    rows_state = []
    with cf.ThreadPoolExecutor(max_workers=INTRADAY_WORKERS) as ex:
        futures = {ex.submit(one, t): t for t in tickers}
        for i, f in enumerate(cf.as_completed(futures), 1):
            t = futures[f]
            try:
                row, n, skipped = f.result()
                result.skipped += int(skipped)
                if row:
                    rows_state.append(row)
                result.rows += n
                result.updated += int(n > 0)
            except Exception as e:
                result.fail(t, common.format_error(e), log)
            if i % 25 == 0:
                log.info(f'  ...{i}/{len(tickers)} ({result.rows:,} minutes)')

    pcon = common.connect('prices')
    try:
        if rows_state:
            st = pd.DataFrame(rows_state)
            st['last_ts'] = pd.to_datetime(st['last_ts'], utc=True).dt.tz_localize(None)
            common.upsert(pcon, 'intraday_state', st)
    finally:
        pcon.close()

    # always all US tickers, so a --tickers run doesn't shrink the review file to its selection
    all_us = files.loc[files['group_name'] == config.US_GROUP, 'ticker'].drop_duplicates().tolist()
    checked = check_against_daily(all_us)
    result.message = (f'{result.rows:,} new minutes; check: {int(checked["flagged"].sum())} of {len(checked)} '
                      f'ticker-days off (> {PRICE_TOL:.0%} price or > {VOLUME_TOL:.0%} vs own median volume ratio)')
    log.info(result.message)
    return result
