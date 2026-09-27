"""
Step 'fred': FRED bond yields and macro series (prices.duckdb).

- Bond yields: every bond ticker in config.FRED_BOND_MAP gets an extra
  instrument '<ticker>_FRED' (e.g. '^TNX_FRED' from DGS10) in instruments /
  prices_daily; only close is filled. '^' bond tickers without a mapping are
  logged; bond ETFs are skipped silently.
- Macro: the series in _tickers/macro.py -> macro_series + macro_observations.

Update rule per series (also used for the FX helper series of the prices step):
  FRED last_updated unchanged      -> skip (only the metadata request)
  new series                       -> full download
  changed, daily                   -> re-download the last MACRO_OVERLAP_DAYS
  changed, weekly/monthly/...      -> full re-download (revisions can go back years)
macro_series.category separates macro / bond_yield / fx_helper series.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.fred')

API = 'https://api.stlouisfed.org/fred'
SERIES_COLS = ['series_id', 'title', 'frequency', 'units', 'seasonal_adjustment', 'last_updated', 'category']
# a series in several groups keeps the first category of this list (your macro.py list wins)
CATEGORY_PRIORITY = ['macro', 'bond_yield', 'fx_helper']


# ----
# DOWNLOADS
# ----

def fetch_series_info(series_id: str) -> dict:
    r = common.http_get(f'{API}/series', 'fred', params={
        'series_id': series_id, 'api_key': common.get_key('FRED_API_KEY'), 'file_type': 'json'})
    s = r.json()['seriess'][0]
    return {
        'series_id':           series_id,
        'title':               s.get('title'),
        'frequency':           s.get('frequency'),
        'frequency_short':     s.get('frequency_short'),
        'units':               s.get('units'),
        'seasonal_adjustment': s.get('seasonal_adjustment'),
        'last_updated':        pd.to_datetime(s.get('last_updated'), utc=True).tz_localize(None),
    }


def fetch_observations(series_id: str, start: str | date | None = None) -> pd.DataFrame:
    """
    Columns series_id, date, value; FRED's '.' (no value) becomes NaN and is dropped.
    """
    r = common.http_get(f'{API}/series/observations', 'fred', params={
        'series_id': series_id, 'api_key': common.get_key('FRED_API_KEY'), 'file_type': 'json',
        'observation_start': str(start or config.START_DATE)})
    obs = pd.DataFrame(r.json().get('observations', []))
    if obs.empty:
        return pd.DataFrame(columns=['series_id', 'date', 'value'])
    obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
    obs = obs.dropna(subset=['value'])
    return pd.DataFrame({'series_id': series_id, 'date': pd.to_datetime(obs['date']).dt.date, 'value': obs['value']})


# ----
# INCREMENTAL UPDATE
# ----

def update_mode(stored_updated, new_updated, frequency_short: str | None, has_obs: bool) -> str:
    """
    'full', 'overlap' or 'skip' for one series (see module docstring).
    """
    if not has_obs or stored_updated is None or pd.isna(stored_updated):
        return 'full'
    if pd.Timestamp(new_updated) <= pd.Timestamp(stored_updated):
        return 'skip'
    return 'overlap' if frequency_short == 'D' else 'full'


def update_series(pcon, series_ids: list[str], category: str,
                  overlap_days: int = config.MACRO_OVERLAP_DAYS,
                  result: common.StepResult | None = None,
                  first_start: str | date | None = None) -> dict[str, str]:
    """
    Update each series by update_mode(); returns {series_id: mode, 'failed' or 'no_data'}.
    A full download starts at `first_start` (default START_DATE) and replaces
    all stored observations of that series. A series that failed
    NO_DATA_AFTER_FAILURES runs in a row gets status 'no_data' and is only
    tried again every NO_DATA_RECHECK_DAYS.
    """
    today = date.today()
    recheck = today - timedelta(days=config.NO_DATA_RECHECK_DAYS)
    parked = {sid for sid, checked in pcon.execute("""
        SELECT series_id, last_checked FROM macro_series
        WHERE series_id IN (SELECT unnest(?)) AND status = 'no_data'""", [series_ids]).fetchall()
        if checked is not None and checked > recheck}
    rows = pcon.execute("""
        SELECT series_id, last_updated, category FROM macro_series WHERE series_id IN (SELECT unnest(?))""",
        [series_ids]).fetchall()
    stored = {sid: upd for sid, upd, _ in rows}
    stored_cat = {sid: cat for sid, _, cat in rows}
    last_obs = dict(pcon.execute("""
        SELECT series_id, max(date) FROM macro_observations
        WHERE series_id IN (SELECT unnest(?)) GROUP BY series_id""", [series_ids]).fetchall())

    modes = {}
    for sid in series_ids:
        if sid in parked:
            modes[sid] = 'no_data'
            if result is not None:
                result.skipped += 1
            continue
        try:
            info = fetch_series_info(sid)
            mode = update_mode(stored.get(sid), info['last_updated'], info['frequency_short'], sid in last_obs)
            if mode == 'full':
                obs = fetch_observations(sid, first_start or config.START_DATE)
                pcon.execute('DELETE FROM macro_observations WHERE series_id = ?', [sid])
                n = common.upsert(pcon, 'macro_observations', obs)
            elif mode == 'overlap':
                obs = fetch_observations(sid, last_obs[sid] - timedelta(days=overlap_days))
                n = common.upsert(pcon, 'macro_observations', obs)
            else:
                n = 0
            info['category'] = _category(category, stored_cat.get(sid))
            row = {k: info[k] for k in SERIES_COLS}
            row.update(status='active', consecutive_failures=0, last_checked=today)
            common.upsert(pcon, 'macro_series', pd.DataFrame([row]))
            modes[sid] = mode
            if result is not None:
                result.rows += n
                if mode == 'skip':
                    result.skipped += 1
                else:
                    result.updated += 1
        except Exception as e:
            modes[sid] = 'failed'
            _record_failure(pcon, sid, _category(category, stored_cat.get(sid)), today)
            if result is not None:
                result.fail(sid, common.format_error(e), log)
            else:
                log.warning(f'{sid}: {common.format_error(e)}')
    return modes


def _category(new: str, stored: str | None) -> str:
    return min([c for c in (new, stored) if c],
               key=lambda c: CATEGORY_PRIORITY.index(c) if c in CATEGORY_PRIORITY else 99)


def _record_failure(pcon, sid: str, category: str, today: date) -> None:
    pcon.execute("""
        INSERT INTO macro_series (series_id, category, status, consecutive_failures, last_checked)
        VALUES (?, ?, 'active', 1, ?)
        ON CONFLICT (series_id) DO UPDATE SET
            consecutive_failures = coalesce(macro_series.consecutive_failures, 0) + 1,
            last_checked = EXCLUDED.last_checked""", [sid, category, today])
    pcon.execute(f"""
        UPDATE macro_series SET status = 'no_data'
        WHERE series_id = ? AND consecutive_failures >= {config.NO_DATA_AFTER_FAILURES}""", [sid])


# ----
# BOND YIELD COPIES
# ----

def write_bond_copies(pcon, bond_names: dict[str, str]) -> int:
    """
    Copy the FRED yield series into prices_daily as '<ticker>_FRED' instruments.
    """
    rows = 0
    for ticker, sid in config.FRED_BOND_MAP.items():
        obs = pcon.execute('SELECT date, value FROM macro_observations WHERE series_id = ? ORDER BY date', [sid]).df()
        if obs.empty:
            continue
        copy = f'{ticker}_FRED'
        name = bond_names.get(ticker, ticker).replace('CBOE ', '')
        common.upsert(pcon, 'instruments', pd.DataFrame([{
            'ticker': copy, 'name': f'{name} (FRED: {sid})', 'asset_class': 'bond', 'group_name': 'fred_bonds',
            'currency': 'USD', 'source': 'fred', 'status': 'active',
            'first_date': obs['date'].min(), 'last_date': obs['date'].max(), 'updated_at': datetime.now()}]))
        pcon.execute('DELETE FROM prices_daily WHERE ticker = ?', [copy])
        rows += common.upsert(pcon, 'prices_daily', pd.DataFrame({
            'ticker': copy, 'date': pd.to_datetime(obs['date']).dt.date, 'close': obs['value'],
            'asset_class': 'bond', 'currency': 'USD', 'source': f'fred:{sid}', 'updated_at': datetime.now()}))
    return rows


# ----
# STEP
# ----

def run(ctx: common.RunContext) -> common.StepResult:
    result = common.StepResult()
    files = common.load_ticker_files()
    bonds = files[files['asset_class'] == 'bond']
    macro = files[files['asset_class'].isin(config.FRED_ASSET_CLASSES)].drop_duplicates('ticker')

    unmapped = [t for t in bonds['ticker'] if t.startswith('^') and t not in config.FRED_BOND_MAP]
    for t in unmapped:
        log.warning(f'{t}: yield ticker without a FRED series in config.FRED_BOND_MAP')

    bond_tickers = ctx.selected([t for t in config.FRED_BOND_MAP if t in set(bonds['ticker'])])
    macro_ids = ctx.selected(macro['ticker'])

    pcon = common.connect('prices')
    try:
        if bond_tickers:
            update_series(pcon, [config.FRED_BOND_MAP[t] for t in bond_tickers], 'bond_yield', result=result)
            write_bond_copies(pcon, dict(zip(bonds['ticker'], bonds['name'])))
        if macro_ids:
            log.info(f'{len(macro_ids)} macro series')
            update_series(pcon, macro_ids, 'macro', result=result)
        if not ctx.tickers:
            # series no longer in _tickers/macro.py: kept with their data, not requested or listed any more
            gone = pcon.execute("""
                UPDATE macro_series SET status = 'removed'
                WHERE category = 'macro' AND coalesce(status, 'active') <> 'removed'
                  AND series_id NOT IN (SELECT unnest(?)) RETURNING series_id""", [list(macro['ticker'])]).fetchall()
            if gone:
                log.info(f'{len(gone)} series no longer in macro.py -> status removed: {", ".join(g[0] for g in gone)}')
    finally:
        pcon.close()
    return result
