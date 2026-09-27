"""
FX rates and USD columns for prices_daily (used by the prices step).

Yahoo FX tickers and their direction:
    'EURUSD=X' -> price = USD per 1 EUR   (base EUR, quote USD)
    'USDJPY=X' -> price = JPY per 1 USD   (base USD, quote JPY)
    'JPY=X'    -> price = JPY per 1 USD   (Yahoo shorthand for USDJPY=X)
fx_rates_daily holds, per currency and day, USD per 1 unit of the currency.
Where Yahoo's history starts late, FRED daily rates fill the earlier years
(EUR before 1999: DEM rate / 1.95583).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import config
from pipeline import common, fred


log = common.get_logger('pipeline.fx')

FX_GAPS_CSV = config.REVIEW_DIR / 'fx_gaps.csv'


# ----
# PAIRS
# ----

def pair_currencies(ticker: str) -> tuple[str, str] | None:
    """
    (base, quote) of a Yahoo FX ticker: price = quote units per 1 base unit.
    """
    if not ticker.endswith('=X'):
        return None
    code = ticker[:-2].upper()
    if len(code) == 3:
        return 'USD', code
    if len(code) == 6:
        return code[:3], code[3:]
    return None


def usd_per_unit(ticker: str, currency: str, price: float) -> float | None:
    """
    Convert a Yahoo FX price into USD per 1 unit of `currency`.
    """
    pair = pair_currencies(ticker)
    if pair is None or not price:
        return None
    base, quote = pair
    if (base, quote) == (currency, 'USD'):
        return price
    if (base, quote) == ('USD', currency):
        return 1.0 / price
    return None


def major_and_divisor(quoted: str | None) -> tuple[str | None, float]:
    """
    'GBp' -> ('GBP', 100); 'EUR' -> ('EUR', 1).
    """
    if quoted in config.MINOR_UNITS:
        return config.MINOR_UNITS[quoted]
    return (quoted.upper() if isinstance(quoted, str) else None), 1.0


def choose_pairs(currencies: set[str], fx_tickers: list[str]) -> dict[str, str]:
    """
    Yahoo ticker used for each currency: CCYUSD=X, then USDCCY=X, then CCY=X.
    Currencies without any pair get the helper 'CCY=X'.
    """
    available = set(fx_tickers)
    out = {}
    for c in sorted(currencies - {'USD'}):
        for t in (f'{c}USD=X', f'USD{c}=X', f'{c}=X'):
            if t in available:
                out[c] = t
                break
        else:
            out[c] = f'{c}=X'
    return out


def ensure_pairs(pcon) -> dict[str, str]:
    """
    Make sure every quote currency in instruments has an FX pair; missing
    pairs are added to instruments as helpers (group 'fx_helper').
    """
    quoted = [r[0] for r in pcon.execute('SELECT DISTINCT currency FROM instruments WHERE currency IS NOT NULL').fetchall()]
    currencies = {major_and_divisor(q)[0] for q in quoted} - {None}
    fx_tickers = [r[0] for r in pcon.execute("SELECT ticker FROM instruments WHERE asset_class = 'forex'").fetchall()]
    pairs = choose_pairs(currencies, fx_tickers)
    helpers = [t for t in pairs.values() if t not in fx_tickers]
    if helpers:
        log.info(f'adding helper FX pairs: {", ".join(helpers)}')
        common.upsert(pcon, 'instruments', pd.DataFrame({
            'ticker': helpers, 'name': [f'{t[:-2]} FX helper' for t in helpers], 'asset_class': 'forex',
            'group_name': 'fx_helper', 'currency': 'USD', 'is_helper': True, 'source': 'yfinance',
            'updated_at': datetime.now()}))
    return pairs


# ----
# RATES TABLE
# ----

def fred_rates(currency: str, obs: dict[str, pd.DataFrame]) -> list[tuple[str, pd.Series]]:
    """
    FRED series for a currency as (source, USD-per-unit Series indexed by date), in order of use.
    """
    out = []
    for series_id, kind in config.FRED_FX_SERIES.get(currency, []):
        df = obs.get(series_id)
        if df is None or df.empty:
            continue
        s = df.set_index(pd.to_datetime(df['date']))['value'].astype(float).sort_index()
        if kind == 'usd_per':
            rate, src = s, f'fred:{series_id}'
        elif kind == 'per_usd':
            rate, src = 1.0 / s, f'fred:{series_id}'
        elif kind in ('per_usd_monthly', 'dem_per_usd_monthly'):
            # monthly average -> the same value on every business day of that month
            monthly = 1.0 / s
            src = f'fred:{series_id} monthly'
            if kind == 'dem_per_usd_monthly':       # USD per EUR = (DEM per EUR) / (DEM per USD)
                monthly, src = monthly / config.EUR_PER_DEM, f'fred:{series_id}/DEM monthly'
            days = pd.bdate_range(monthly.index.min(), monthly.index.max() + pd.offsets.MonthEnd(0))
            rate = monthly.reindex(days.to_period('M').to_timestamp()).set_axis(days)
        else:
            continue
        out.append((src, rate.dropna()))
    return out


def peg_rates(currency: str) -> list[tuple[str, pd.Series]]:
    """
    Fixed peg from config.FX_PEGS as a business-day series up to the end of the peg.
    """
    if currency not in config.FX_PEGS:
        return []
    value, end, note = config.FX_PEGS[currency]
    start = pd.Timestamp(config.START_DATE) - pd.Timedelta(days=14)
    days = pd.bdate_range(start, end)
    return [(note, pd.Series(value, index=days))]


def combine_rates(currency: str, yahoo: pd.Series | None, yahoo_src: str,
                  fred_list: list[tuple[str, pd.Series]],
                  gap_days: int = config.FX_FFILL_LIMIT) -> pd.DataFrame:
    """
    Yahoo where it exists. Each FRED series (in order) fills the dates that
    are not yet covered: before the first covered date, and inside gaps
    longer than `gap_days` (shorter gaps are forward-filled later).
    """
    parts = []
    if yahoo is not None and not yahoo.dropna().empty:
        y = yahoo.dropna()
        y.index = pd.to_datetime(y.index)
        parts.append(pd.DataFrame({'date': y.index, 'usd_per_unit': y.values, 'source': yahoo_src}))
    for src, s in fred_list:
        s = s.dropna()
        if s.empty:
            continue
        if parts:
            covered = pd.DatetimeIndex(sorted(pd.concat(parts)['date']))
            pos = covered.searchsorted(s.index, side='right') - 1
            prev = pd.Series(covered[pos.clip(min=0)], index=s.index).where(pos >= 0)
            exact = s.index.isin(covered)
            keep = ~exact & (prev.isna() | ((s.index - prev).dt.days > gap_days)).values
            s = s[keep]
        if not s.empty:
            parts.append(pd.DataFrame({'date': s.index, 'usd_per_unit': s.values, 'source': src}))
    if not parts:
        return pd.DataFrame(columns=['currency', 'date', 'usd_per_unit', 'source'])
    df = pd.concat(parts).sort_values('date')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.insert(0, 'currency', currency)
    return df


def rebuild_rates(pcon, pairs: dict[str, str]) -> None:
    """
    Recreate fx_rates_daily from the stored Yahoo FX closes + FRED series.
    """
    series_ids = sorted({sid for c in pairs for sid, _ in config.FRED_FX_SERIES.get(c, [])})
    obs = {}
    if series_ids:
        stored = pcon.execute("""
            SELECT series_id, date, value FROM macro_observations WHERE series_id IN (SELECT unnest(?))""",
            [series_ids]).df()
        obs = {sid: g for sid, g in stored.groupby('series_id')}

    frames = []
    for currency, ticker in pairs.items():
        y = pcon.execute('SELECT date, close FROM prices_daily WHERE ticker = ? ORDER BY date', [ticker]).df()
        yahoo = None
        if not y.empty:
            yahoo = pd.Series([usd_per_unit(ticker, currency, p) for p in y['close']], index=y['date'])
        frames.append(combine_rates(currency, yahoo, f'yahoo:{ticker}',
                                    peg_rates(currency) + fred_rates(currency, obs)))

    rates = pd.concat(frames) if frames else pd.DataFrame()
    pcon.execute('DELETE FROM fx_rates_daily')
    if not rates.empty:
        common.upsert(pcon, 'fx_rates_daily', rates)


# ----
# USD COLUMNS
# ----

def _register_units(pcon) -> None:
    """
    Temp view `_units` (quoted, major, divisor) for minor currency units.
    """
    units = pd.DataFrame([(q, m, float(d)) for q, (m, d) in config.MINOR_UNITS.items()],
                         columns=['quoted', 'major', 'divisor'])
    pcon.register('_units', units)


# price rows with the major currency and divisor of their quote currency
_QUOTED = """
    SELECT p.ticker, p.date, coalesce(u.major, upper(p.currency)) AS major,
           coalesce(u.divisor, 1.0) AS divisor
    FROM prices_daily p LEFT JOIN _units u ON u.quoted = p.currency
"""


def no_usd_tickers(pcon) -> list[str]:
    """
    Instruments whose values are not money (yields, volatility indices, ...).
    """
    inst = pcon.execute('SELECT ticker, name FROM instruments').df()
    mask = (inst['ticker'].isin(config.NO_USD_TICKERS)
            | inst['ticker'].str.endswith(config.NO_USD_SUFFIXES)
            | inst['name'].fillna('').str.contains(config.NO_USD_NAME_PATTERN, regex=True))
    return sorted(inst.loc[mask, 'ticker'])


def apply_usd(pcon, tickers: list[str] | None = None) -> None:
    """
    (Re)calculate fx_to_usd and the *_usd columns. fx_to_usd = USD per 1
    unit of the quote currency (GBp: GBPUSD / 100). The FX rate is the
    latest one on or before the price date, at most FX_FFILL_LIMIT days old.
    Instruments from no_usd_tickers() keep empty USD columns.
    """
    _register_units(pcon)
    pcon.register('_no_usd', pd.DataFrame({'ticker': no_usd_tickers(pcon)}, dtype=str))
    try:
        where = 'WHERE p.ticker IN (SELECT unnest(?))' if tickers is not None else ''
        pcon.execute(f"""
            UPDATE prices_daily AS t SET
                fx_to_usd     = x.fx,
                open_usd      = t.open * x.fx,
                high_usd      = t.high * x.fx,
                low_usd       = t.low * x.fx,
                close_usd     = t.close * x.fx,
                adj_close_usd = t.adj_close * x.fx
            FROM (
                SELECT q.ticker, q.date,
                       CASE WHEN q.ticker IN (SELECT ticker FROM _no_usd) THEN NULL
                            WHEN q.major = 'USD' THEN 1.0 / q.divisor
                            WHEN q.date - f.date <= {config.FX_FFILL_LIMIT} THEN f.usd_per_unit / q.divisor
                       END AS fx
                FROM ({_QUOTED} {where}) q
                ASOF LEFT JOIN fx_rates_daily f ON f.currency = q.major AND q.date >= f.date
            ) AS x
            WHERE t.ticker = x.ticker AND t.date = x.date""", [tickers] if tickers is not None else [])
    finally:
        pcon.unregister('_units')
        pcon.unregister('_no_usd')


def report_gaps(pcon) -> pd.DataFrame:
    """
    Currencies whose FX history starts after the first price in that currency,
    with the number of price rows that have no USD value.
    """
    _register_units(pcon)
    try:
        gaps = pcon.execute(f"""
            WITH q AS ({_QUOTED}),
            per_ccy AS (
                SELECT q.major AS currency, min(q.date) AS first_price,
                       count(*) FILTER (WHERE p.close IS NOT NULL AND p.fx_to_usd IS NULL) AS rows_without_usd
                FROM q JOIN prices_daily p USING (ticker, date)
                WHERE q.major IS NOT NULL AND q.major <> 'USD'
                  AND q.ticker NOT IN (SELECT unnest(?)) GROUP BY 1),
            first_fx AS (SELECT currency, min(date) AS first_fx FROM fx_rates_daily GROUP BY 1)
            SELECT c.currency, c.first_price, f.first_fx, c.rows_without_usd
            FROM per_ccy c LEFT JOIN first_fx f USING (currency)
            WHERE f.first_fx IS NULL OR f.first_fx > c.first_price OR c.rows_without_usd > 0
            ORDER BY 1""", [no_usd_tickers(pcon)]).df()
    finally:
        pcon.unregister('_units')
    if not gaps.empty:
        gaps.to_csv(FX_GAPS_CSV, index=False)
        for r in gaps.itertuples():
            log.warning(f'FX gap {r.currency}: prices from {r.first_price}, FX from {r.first_fx}, '
                        f'{r.rows_without_usd} rows without USD')
    elif FX_GAPS_CSV.exists():
        FX_GAPS_CSV.unlink()
    return gaps


def update(pcon, pairs: dict[str, str]) -> None:
    """
    FRED FX series -> fx_rates_daily -> USD columns -> gap report.
    """
    series_ids = sorted({sid for c in pairs for sid, _ in config.FRED_FX_SERIES.get(c, [])})
    # two weeks before START_DATE, so the first price days have a rate to carry forward
    first = (pd.Timestamp(config.START_DATE) - pd.Timedelta(days=14)).date()
    fred.update_series(pcon, series_ids, category='fx_helper', first_start=first)
    rebuild_rates(pcon, pairs)
    apply_usd(pcon)
    report_gaps(pcon)
