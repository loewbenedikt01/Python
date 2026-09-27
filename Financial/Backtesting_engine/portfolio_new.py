import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
DATABASE_DIR = Path(r'C:\Users\benel\Coding\Python\Database')

PARQUET_FILES = {
    'equities':     'equities.parquet',
    'macro':        'macro.parquet',
    'bonds':        'bonds.parquet',
    'forex':        'forex.parquet',
    'commodities':  'commodities.parquet',
    'crypto':       'crypto.parquet',
    'sentiment':    'sentiment.parquet',
    'sectors':      'sectors.parquet',
    'indices':      'indices.parquet',
}


# ── Universe country sets ─────────────────────────────────────────────────────
_UNIVERSES: dict[str, set[str] | None] = {
    'US':                 {'United States'},
    'North America':      {'United States', 'Canada', 'Mexico'},
    'Europe':             {'Austria', 'Belgium', 'Denmark', 'Finland', 'France',
                           'Germany', 'Italy', 'Netherlands', 'Norway', 'Poland',
                           'Portugal', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom'},
    'Asia':               {'Australia', 'China', 'Hong Kong', 'India', 'Indonesia',
                           'Japan', 'Malaysia', 'Singapore', 'South Korea',
                           'Taiwan', 'Thailand'},
    'Developed Markets':  {'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark',
                           'Finland', 'France', 'Germany', 'Hong Kong', 'Italy',
                           'Japan', 'Netherlands', 'Norway', 'Portugal', 'Singapore',
                           'South Korea', 'Spain', 'Sweden', 'Switzerland',
                           'United Kingdom', 'United States'},
    'Emerging Markets':   {'Brazil', 'China', 'India', 'Indonesia', 'Malaysia',
                           'Mexico', 'Taiwan', 'Thailand'},
    'Global':             None,   # None = all countries
}


def load_equities() -> pd.DataFrame:
    """
    Load equities.parquet — single file with all countries.
    Returns a DataFrame with (Date, Ticker) MultiIndex.
    """
    path = DATABASE_DIR / PARQUET_FILES['equities']
    if not path.exists():
        raise FileNotFoundError(f'equities.parquet not found at {path}')
    return pd.read_parquet(path)


def _select_universe(equities: pd.DataFrame, universe: str) -> pd.DataFrame:
    """
    Filter the combined equities DataFrame to the requested universe.

    Parameters
    ----------
    equities : DataFrame with (Date, Ticker) MultiIndex and a 'Country' column
    universe : one of the keys in _UNIVERSES

    Returns
    -------
    Filtered DataFrame (same structure as input).
    """
    countries = _UNIVERSES.get(universe)
    if countries is None and universe not in _UNIVERSES:
        raise ValueError(
            f"Unknown universe '{universe}'. "
            f"Choose: {', '.join(_UNIVERSES)}"
        )

    if countries is None:
        return equities

    ticker_country = equities['Country'].groupby(level='Ticker').last()
    tickers_in     = ticker_country[ticker_country.isin(countries)].index
    return equities.loc[pd.IndexSlice[:, tickers_in], :]



# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES
# Each function takes (close, date, **params) and returns a pd.Series of
# ticker → score, sorted descending (best candidates first).
# ══════════════════════════════════════════════════════════════════════════════

def _momentum(
    close: pd.DataFrame,
    date:  pd.Timestamp,
    lookback:    int = 252,   # ~12 months
    skip:        int = 21,    # ~1 month (avoid short-term reversal)
    min_history: int = 300,
) -> pd.Series:
    """
    Classic Jegadeesh-Titman 12-1 cross-sectional momentum.
    score = price[t - skip] / price[t - lookback] - 1
    """
    history = close.loc[:date].dropna(axis=1, thresh=min_history)
    if len(history) < lookback:
        return pd.Series(dtype=float)
    price_skip = history.iloc[-skip]
    price_long = history.iloc[-lookback]
    valid  = price_skip.notna() & price_long.notna() & (price_long > 0)
    scores = (price_skip[valid] / price_long[valid]) - 1
    return scores.sort_values(ascending=False)


def _momentum_6_1(
    close: pd.DataFrame,
    date:  pd.Timestamp,
    min_history: int = 150,
) -> pd.Series:
    """6-month minus 1-month momentum."""
    return _momentum(close, date, lookback=126, skip=21, min_history=min_history)


def _mean_reversion(
    close: pd.DataFrame,
    date:  pd.Timestamp,
    lookback:    int = 21,    # 1-month reversal
    min_history: int = 60,
) -> pd.Series:
    """
    Short-term mean reversion: rank by worst recent 1-month return
    (buy losers expecting reversal). Score = negative momentum.
    """
    history = close.loc[:date].dropna(axis=1, thresh=min_history)
    if len(history) < lookback:
        return pd.Series(dtype=float)
    price_now  = history.iloc[-1]
    price_past = history.iloc[-lookback]
    valid  = price_now.notna() & price_past.notna() & (price_past > 0)
    scores = (price_now[valid] / price_past[valid]) - 1
    return (-scores).sort_values(ascending=False)   # invert: buy worst performers


# ── Registry ──────────────────────────────────────────────────────────────────
# Maps the strategy name used in test_new.py → implementation function
_STRATEGY_REGISTRY: dict[str, callable] = {
    'Momentum':       _momentum,
    'Momentum_6_1':   _momentum_6_1,
    'Mean_Reversion': _mean_reversion,
}


def run_strategy(
    close:    pd.DataFrame,
    date:     pd.Timestamp,
    strategy: str,
    eligible: list[str] | None = None,
    **params,
) -> pd.Series:
    """
    Compute a stock-selection score for all (or eligible) tickers.

    Parameters
    ----------
    close     : wide price DataFrame (Date x Ticker)
    date      : scoring date
    strategy  : name from _STRATEGY_REGISTRY, e.g. 'Momentum'
    eligible  : optional subset of tickers (region/universe filter)
    **params  : forwarded to the strategy function (e.g. lookback=252)

    Returns
    -------
    pd.Series of ticker → score, sorted descending.
    """
    fn = _STRATEGY_REGISTRY.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {', '.join(_STRATEGY_REGISTRY)}"
        )
    if eligible is not None:
        close = close[[t for t in eligible if t in close.columns]]
    return fn(close, date, **params)



# ------------------------------------------
# Portfolio Computation
# ------------------------------------------

def create_portfolio(
    universe:   str,
    weighting:  str,
    start_date: str | None = None,
    end_date:   str | None = None,
    equities:   pd.DataFrame | None = None,
) -> dict:
    """
    Build a portfolio from a named universe of stocks.

    Parameters
    ----------
    universe   : 'US' | 'North America' | 'Europe' | 'Asia' |
                 'Developed Markets' | 'Emerging Markets' | 'Global'
    weighting  : 'equal'      — 1/N weight per available ticker each day
                 'market_cap' — weight proportional to Market_Cap column
    start_date : clip from this date (YYYY-MM-DD)
    end_date   : clip to this date (YYYY-MM-DD)
    equities   : pre-loaded equities DataFrame (pass to avoid re-reading disk)

    Returns
    -------
    dict with keys:
      'close'    — pd.DataFrame wide (Date x Ticker)
      'weights'  — pd.DataFrame wide (Date x Ticker)
      'exposure' — pd.Series  (1.0 = fully invested)
      'regimes'  — pd.DataFrame placeholder for regime states
    """
    if equities is None:
        equities = load_equities()

    df = _select_universe(equities, universe)
    if df.empty:
        raise ValueError(f"No data found for universe '{universe}'.")

    close = df['Close'].unstack('Ticker')

    if start_date:
        close = close.loc[start_date:]
    if end_date:
        close = close.loc[:end_date]

    if weighting == 'market_cap' and 'Market_Cap' in df.columns:
        mcap_wide    = df['Market_Cap'].unstack('Ticker').reindex(close.index)[close.columns]
        base_weights = mcap_wide.div(mcap_wide.sum(axis=1), axis=0).fillna(0.0)
    else:
        valid        = close.notna()
        n_valid      = valid.sum(axis=1).replace(0, np.nan)
        base_weights = valid.div(n_valid, axis=0).fillna(0.0)

    exposure = pd.Series(1.0, index=close.index)
    weights  = base_weights.multiply(exposure, axis=0)

    return {
        'close':    close,
        'weights':  weights,
        'exposure': exposure,
        'regimes':  pd.DataFrame(index=close.index),
    }
