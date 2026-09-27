"""
Step 'prices': daily prices for every instrument from yfinance, then FX
rates and USD columns (pipeline/fx.py).

- auto_adjust=False, actions=True: close is split-adjusted, adj_close is
  split + dividend adjusted, dividends and splits are stored.
- Dates are the exchange's local trading dates (yfinance default), never UTC.
- Incremental per ticker: a new ticker gets history from config.START_DATE;
  an existing ticker is re-downloaded from its last stored date minus
  PRICE_OVERLAP_DAYS trading days, which also overwrites a partial row
  stored while the market was open. If adj_close in the overlap differs
  from the stored value by more than ADJ_CLOSE_TOLERANCE (dividend / split),
  the ticker's full history is downloaded again.
- Tickers without data: after NO_DATA_AFTER_FAILURES failed runs in a row
  status = 'no_data'; they are skipped and re-checked every NO_DATA_RECHECK_DAYS.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

import config
from pipeline import common, fx


log = common.get_logger('pipeline.prices')

COLUMNS = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
           'Volume': 'volume', 'Dividends': 'dividends', 'Stock Splits': 'stock_splits'}


# ----
# DOWNLOAD
# ----

def download(tickers: list[str], start) -> dict[str, pd.DataFrame]:
    """
    {ticker: frame with date + price columns}; tickers without rows are missing.
    """
    out = {}
    for i in range(0, len(tickers), config.YF_BATCH_SIZE):
        batch = tickers[i:i + config.YF_BATCH_SIZE]
        common.limiter('yfinance').wait()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = yf.download(batch, start=str(start), auto_adjust=False, actions=True,
                              group_by='ticker', threads=True, progress=False)
        if raw is None or raw.empty:
            continue
        for t in batch:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t]
            else:
                df = raw
            df = df.rename(columns=COLUMNS)
            if 'close' not in df.columns:              # failed ticker inside a batch
                continue
            df = df[[c for c in COLUMNS.values() if c in df.columns]].dropna(subset=['close'])
            if df.empty:
                continue
            df = df.reset_index().rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date          # local trading date, no tz shift
            out[t] = df
    return out


def adj_close_changed(stored: pd.DataFrame, new: pd.DataFrame, before: date,
                      tolerance: float = config.ADJ_CLOSE_TOLERANCE) -> bool:
    """
    True if adj_close differs by more than `tolerance` (relative) on any
    overlapping date before `before` (the last stored date, which may have
    been a partial intraday row).
    """
    stored = stored.assign(date=pd.to_datetime(stored['date']).dt.date)
    new = new.assign(date=pd.to_datetime(new['date']).dt.date)
    m = stored[stored['date'] < before][['date', 'adj_close']].merge(
        new[['date', 'adj_close']], on='date', suffixes=('_old', '_new'))
    m = m.dropna(subset=['adj_close_old', 'adj_close_new'])
    m = m[m['adj_close_old'] != 0]
    if m.empty:
        return False
    rel = (m['adj_close_new'] - m['adj_close_old']).abs() / m['adj_close_old'].abs()
    return bool((rel > tolerance).any())


def _overlap_start(last: date) -> date:
    # PRICE_OVERLAP_DAYS trading days back (~1.5 calendar days per trading day incl. weekends)
    return last - timedelta(days=int(config.PRICE_OVERLAP_DAYS * 1.5) + 2)


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    pcon = common.connect('prices')
    try:
        pairs = fx.ensure_pairs(pcon)
        today = date.today()
        recheck = today - timedelta(days=config.NO_DATA_RECHECK_DAYS)

        inst = pcon.execute("""
            SELECT i.ticker, i.asset_class, i.currency, i.status, i.last_checked, i.consecutive_failures,
                   max(p.date) AS last_date
            FROM instruments i LEFT JOIN prices_daily p USING (ticker)
            WHERE NOT ends_with(i.ticker, '_FRED')
            GROUP BY ALL""").df()
        for c in ('last_date', 'last_checked'):
            inst[c] = [d.date() if pd.notna(d) else None for d in pd.to_datetime(inst[c])]
        # --tickers limits the instruments, but FX pairs are always updated (needed for USD)
        wanted = set(ctx.selected(inst['ticker'])) | set(pairs.values())
        inst = inst[inst['ticker'].isin(wanted)]

        skip = ((inst['status'] == 'no_data') & inst['last_checked'].map(lambda d: d is not None and d > recheck)) \
            | (inst['status'] == 'removed')
        result.skipped += int(skip.sum())
        inst = inst[~skip]
        log.info(f'{len(inst)} tickers to update ({int(skip.sum())} no_data tickers skipped)')

        # groups by download start date: new tickers from START_DATE, others from their overlap
        inst['start'] = [_overlap_start(d) if pd.notna(d) else None for d in inst['last_date']]
        new_tickers = inst.loc[inst['last_date'].isna(), 'ticker'].tolist()
        existing = inst[inst['last_date'].notna()]

        got: dict[str, pd.DataFrame] = {}
        full_reload: list[str] = []
        if new_tickers:
            log.info(f'{len(new_tickers)} new tickers: full history from {config.START_DATE}')
            got.update(download(new_tickers, config.START_DATE))
        for start, grp in existing.groupby('start'):
            tickers = grp['ticker'].tolist()
            new = download(tickers, start)
            stored = pcon.execute("""
                SELECT ticker, date, adj_close FROM prices_daily
                WHERE ticker IN (SELECT unnest(?)) AND date >= ?""", [tickers, start]).df()
            for t in tickers:
                if t not in new:
                    continue
                last = grp.loc[grp['ticker'] == t, 'last_date'].iloc[0]
                if adj_close_changed(stored[stored['ticker'] == t], new[t], last):
                    full_reload.append(t)
                else:
                    got[t] = new[t]
        if full_reload:
            log.info(f'{len(full_reload)} tickers with changed adj_close (dividend/split): full reload')
            reloaded = download(full_reload, config.START_DATE)
            pcon.execute('DELETE FROM prices_daily WHERE ticker IN (SELECT unnest(?))', [list(reloaded)])
            got.update(reloaded)

        # write prices
        meta = inst.set_index('ticker')
        _fill_missing_currency(pcon, meta, [t for t in got if pd.isna(meta.at[t, 'currency'])])
        frames = []
        for t, df in got.items():
            df = df.copy()
            df.insert(0, 'ticker', t)
            df['asset_class'] = meta.at[t, 'asset_class']
            df['currency'] = meta.at[t, 'currency']
            df['source'] = 'yfinance'
            df['updated_at'] = datetime.now()
            frames.append(df)
        if frames:
            result.rows += common.upsert(pcon, 'prices_daily', pd.concat(frames, ignore_index=True))
        result.updated = len(got)

        # ticker status: an empty download only counts as a failure if there is no recent data
        ok = set(got)
        recent = today - timedelta(days=config.RECENT_DATA_DAYS)
        failed = [t for t in inst['ticker'] if t not in ok and not (
            pd.notna(meta.at[t, 'last_date']) and meta.at[t, 'last_date'] >= recent)]
        _update_status(pcon, ok, failed, today)
        for t in failed:
            result.fail(t, 'no price data from yfinance', log)
        result.skipped += len(inst) - len(ok) - len(failed)

        # rows stored before the instrument had a currency
        pcon.execute("""
            UPDATE prices_daily AS p SET currency = i.currency FROM instruments i
            WHERE p.ticker = i.ticker AND p.currency IS NULL AND i.currency IS NOT NULL""")

        # instruments: first/last date
        pcon.execute("""
            UPDATE instruments AS i SET first_date = p.first_date, last_date = p.last_date
            FROM (SELECT ticker, min(date) AS first_date, max(date) AS last_date
                  FROM prices_daily GROUP BY ticker) p
            WHERE i.ticker = p.ticker""")

        # FX rates + USD columns
        fx.update(pcon, pairs)
        result.message = f'{len(full_reload)} full reloads'
    finally:
        pcon.close()
    return result


def _fill_missing_currency(pcon, meta: pd.DataFrame, tickers: list[str]) -> None:
    """
    Currency for tickers that have prices but got none from yfinance .info:
    config.CURRENCY_OVERRIDES, FX pairs -> quote currency of the pair,
    others -> yfinance price metadata.
    """
    found = {}
    for t in tickers:
        if t in config.CURRENCY_OVERRIDES:
            found[t] = config.CURRENCY_OVERRIDES[t]
            continue
        pair = fx.pair_currencies(t)
        if pair:
            found[t] = pair[1]
            continue
        try:
            common.limiter('yfinance').wait()
            tk = yf.Ticker(t)
            tk.history(period='5d')                      # fills the price metadata
            found[t] = (tk.get_history_metadata() or {}).get('currency')
        except Exception as e:
            log.warning(f'{t}: no currency ({common.format_error(e)})')
    for t in [t for t, c in found.items() if not c]:
        log.warning(f'{t}: Yahoo reports no currency; add it to config.CURRENCY_OVERRIDES')
    found = {t: c for t, c in found.items() if c}
    if found:
        log.info(f'currency filled from price data: {found}')
        common.upsert(pcon, 'instruments', pd.DataFrame({'ticker': list(found), 'currency': list(found.values())}))
        for t, c in found.items():
            meta.at[t, 'currency'] = c


def _update_status(pcon, ok: set[str], failed: list[str], today: date) -> None:
    if ok:
        pcon.execute("""
            UPDATE instruments SET status = 'active', consecutive_failures = 0, last_checked = ?
            WHERE ticker IN (SELECT unnest(?))""", [today, list(ok)])
    if failed:
        pcon.execute(f"""
            UPDATE instruments SET
                consecutive_failures = coalesce(consecutive_failures, 0) + 1,
                last_checked = ?,
                status = CASE WHEN coalesce(consecutive_failures, 0) + 1 >= {config.NO_DATA_AFTER_FAILURES}
                              THEN 'no_data' ELSE status END
            WHERE ticker IN (SELECT unnest(?))""", [today, failed])
        newly = pcon.execute("""
            SELECT ticker FROM instruments WHERE ticker IN (SELECT unnest(?)) AND status = 'no_data'""",
            [failed]).fetchall()
        if newly:
            log.warning(f'status no_data: {", ".join(t for (t,) in newly)}')
