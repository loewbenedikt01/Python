"""
Price glitches in prices_daily (Yahoo), run by the prices step before the USD
columns are computed. Rows are kept; only values are corrected or flagged.

1. Unit glitches, GBp / GBX tickers: a day-to-day jump of ~100x or ~1/100x
   (+-10 %) is a switch between pence and pounds. The stretch in the wrong unit
   is scaled back to the unit of the latest data (Yahoo quotes the ticker in
   pence today), open / high / low / close / adj_close x 100 or / 100, and
   marked price_corrected = TRUE (sticky: a later run doesn't reset it).
2. One-day spikes (equities, ETFs, sectors, indices): a move of more than 30 %
   that reverses within 1-3 trading days (back within 15 % of the level before
   the move; the days in between stay > 30 % away) -> price_suspect = TRUE on
   the spike days. Recomputed on every run. Suspect days are left out of
   market cap and the dashboard charts. Not applied to sentiment (VIX spikes),
   bond yields (near zero), commodities (oil futures in April 2020), crypto or
   volatility indices filed under indices (^VIX, ^VXD, ^VXN; fx.no_usd_tickers),
   where such moves are real. Also not applied to US listings (config.US_GROUP):
   Yahoo's US data is consolidated and clean, and all 21 US cases the rule found
   were real events (C / BAC 2009-01-21, AIG 2008, BIIB 2020-11-04, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from pipeline import common


log = common.get_logger('pipeline.price_checks')

UNIT_CURRENCIES = ('GBp', 'GBX')
UNIT_FACTOR, UNIT_TOL = 100.0, 0.10
SPIKE_MOVE, SPIKE_BACK, SPIKE_MAX_DAYS = 0.30, 0.15, 3
SPIKE_CLASSES = ('equities', 'etfs', 'sectors', 'indices')
PRICE_COLS = ['open', 'high', 'low', 'close', 'adj_close']


# ----
# PURE FUNCTIONS (one ticker, sorted by date)
# ----

def unit_factors(close: pd.Series) -> pd.Series:
    """
    Per day the factor that puts the close into the unit of the latest data: 1, 100 or 1/100.
    Walks backwards from the latest day; each ~100x jump flips the unit of the days before it.
    """
    c = close.to_numpy(dtype=float)
    f = np.ones(len(c))
    cur = 1.0
    for i in range(len(c) - 1, 0, -1):
        f[i] = cur
        if c[i - 1] > 0 and c[i] > 0:
            r = c[i] / c[i - 1]
            if abs(r / UNIT_FACTOR - 1) <= UNIT_TOL:
                cur *= UNIT_FACTOR               # day before is in pounds: x 100
            elif abs(r * UNIT_FACTOR - 1) <= UNIT_TOL:
                cur /= UNIT_FACTOR               # day before is 100x too high: / 100
    if len(c):
        f[0] = cur
    f[~np.isin(np.round(f, 6), [1.0, UNIT_FACTOR, 1 / UNIT_FACTOR])] = 1.0   # 10,000x etc.: leave alone
    return pd.Series(f, index=close.index)


def spike_days(close: pd.Series) -> pd.Series:
    """
    True on the days of a > SPIKE_MOVE move that reverses within 1..SPIKE_MAX_DAYS trading days.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    out = np.zeros(n, dtype=bool)
    t = 1
    while t < n:
        prev = c[t - 1]
        if prev > 0 and c[t] > 0 and abs(c[t] / prev - 1) > SPIKE_MOVE:
            for k in range(1, SPIKE_MAX_DAYS + 1):
                if t + k >= n or not c[t + k] > 0:
                    break
                if abs(c[t + k] / prev - 1) < SPIKE_BACK:          # back at the level before the move
                    out[t:t + k] = True
                    t += k - 1
                    break
                if abs(c[t + k] / prev - 1) <= SPIKE_MOVE:         # partly back only: a real move
                    break
        t += 1
    return pd.Series(out, index=close.index)


# ----
# DATABASE
# ----

def apply(pcon) -> dict:
    """
    Correct unit glitches and recompute price_suspect over the whole table. Returns counts.
    """
    # 1. unit glitches
    df = pcon.execute(f"""
        SELECT ticker, date, close FROM prices_daily
        WHERE currency IN (SELECT unnest(?)) AND close IS NOT NULL ORDER BY ticker, date""",
        [list(UNIT_CURRENCIES)]).df()
    fixes = []
    for t, g in df.groupby('ticker', sort=False):
        f = unit_factors(g['close'])
        bad = f != 1.0
        if bad.any():
            fixes.append(pd.DataFrame({'ticker': t, 'date': g.loc[bad, 'date'], 'factor': f[bad]}))
    n_fixed = 0
    if fixes:
        fx = pd.concat(fixes, ignore_index=True)
        pcon.register('_unit_fix', fx)
        sets = ', '.join(f'{c} = p.{c} * u.factor' for c in PRICE_COLS)
        pcon.execute(f"""
            UPDATE prices_daily AS p SET {sets}, price_corrected = TRUE
            FROM _unit_fix u WHERE p.ticker = u.ticker AND p.date = u.date""")
        pcon.unregister('_unit_fix')
        n_fixed = len(fx)
        log.info(f'{n_fixed} days in {fx["ticker"].nunique()} GBp tickers scaled to the right unit: '
                 f'{", ".join(sorted(fx["ticker"].unique()))}')

    # 2. spikes (recomputed)
    from pipeline import fx
    df = pcon.execute("""
        SELECT ticker, date, close FROM prices_daily
        WHERE asset_class IN (SELECT unnest(?)) AND ticker NOT IN (SELECT unnest(?))
          AND ticker NOT IN (SELECT ticker FROM instruments WHERE group_name = ?)
          AND close IS NOT NULL ORDER BY ticker, date""",
        [list(SPIKE_CLASSES), fx.no_usd_tickers(pcon), config.US_GROUP]).df()
    flagged = []
    for t, g in df.groupby('ticker', sort=False):
        s = spike_days(g['close'])
        if s.any():
            flagged.append(g.loc[s.to_numpy(), ['ticker', 'date']])
    sp = pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame(columns=['ticker', 'date'])
    pcon.execute('UPDATE prices_daily SET price_suspect = FALSE WHERE price_suspect')
    if not sp.empty:
        pcon.register('_spikes', sp)
        pcon.execute("""UPDATE prices_daily AS p SET price_suspect = TRUE
                        FROM _spikes s WHERE p.ticker = s.ticker AND p.date = s.date""")
        pcon.unregister('_spikes')
    log.info(f'{len(sp)} one-day spikes in {sp["ticker"].nunique() if not sp.empty else 0} tickers '
             f'-> price_suspect (left out of market cap and charts)')
    return {'unit_fixed': n_fixed, 'suspect': len(sp)}
