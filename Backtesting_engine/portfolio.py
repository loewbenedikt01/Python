import pandas as pd
import numpy as np

# ── Developed / Emerging market country definitions ───────────────────────────
_DM_ROTW = {'Australia', 'Canada', 'Switzerland', 'United Kingdom'}
_DM_ASIA = {'Hong Kong', 'Japan', 'Singapore', 'South Korea'}


def _select_universe(data: dict, universe: str) -> pd.DataFrame:
    """
    Combine the relevant DataFrames for the requested universe.
    Expected keys in data: 'us', 'eu', 'de', 'asia', 'rotw'.
    """
    us   = data.get('us')
    eu   = data.get('eu')
    de   = data.get('de')
    asia = data.get('asia')
    rotw = data.get('rotw')

    if universe == 'US':
        parts = [us]

    elif universe == 'North America':
        na_rotw = rotw[rotw['Country'].isin({'Canada', 'Mexico'})] if rotw is not None else None
        parts = [us, na_rotw]

    elif universe == 'Europe':
        eu_rotw = rotw[rotw['Country'].isin({'Switzerland', 'United Kingdom'})] if rotw is not None else None
        parts = [eu, de, eu_rotw]

    elif universe == 'Asia':
        parts = [asia]

    elif universe == 'Developed Markets':
        dm_rotw = rotw[rotw['Country'].isin(_DM_ROTW)] if rotw is not None else None
        dm_asia = asia[asia['Country'].isin(_DM_ASIA)] if asia is not None else None
        parts = [us, eu, de, dm_rotw, dm_asia]

    elif universe == 'Emerging Markets':
        em_rotw = rotw[~rotw['Country'].isin(_DM_ROTW)] if rotw is not None else None
        em_asia = asia[~asia['Country'].isin(_DM_ASIA)] if asia is not None else None
        parts = [em_rotw, em_asia]

    elif universe == 'Global':
        parts = [us, eu, de, asia, rotw]

    else:
        raise ValueError(
            f"Unknown universe '{universe}'. "
            "Choose: US, North America, Europe, Asia, Developed Markets, Emerging Markets, Global"
        )

    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep='last')].sort_index()


def create_portfolio(
    data:       dict,
    universe:   str,
    weighting:  str        = 'equal',   # 'equal' | 'market_cap'
    start_date: str | None = None,
    end_date:   str | None = None,
    min_mcap:   float | None = None,    # minimum market cap filter
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a portfolio from the given universe.

    Parameters
    ----------
    data       : dict of DataFrames keyed by 'us', 'eu', 'de', 'asia', 'rotw', ...
                 Each DataFrame has a (Date, Ticker) MultiIndex and at least a Close column.
    universe   : 'US' | 'North America' | 'Europe' | 'Asia' |
                 'Developed Markets' | 'Emerging Markets' | 'Global'
    weighting  : 'equal'      - each ticker gets 1/N weight on each date
                 'market_cap' - weights proportional to Market_Cap column
    start_date : clip to this start date (YYYY-MM-DD)
    end_date   : clip to this end date (YYYY-MM-DD)
    min_mcap   : exclude tickers whose most-recent Market_Cap is below this value

    Returns
    -------
    close   : pd.DataFrame - wide (Date x Ticker), adjusted close prices
    weights : pd.DataFrame - wide (Date x Ticker), weights summing to 1 per row
    """
    df = _select_universe(data, universe)
    if df.empty:
        raise ValueError(f"No data found for universe '{universe}'.")

    close = df['Close'].unstack('Ticker')

    # ── date clip ─────────────────────────────────────────────────────────────
    if start_date:
        close = close.loc[start_date:]
    if end_date:
        close = close.loc[:end_date]

    # ── minimum market-cap filter ─────────────────────────────────────────────
    if min_mcap is not None and 'Market_Cap' in df.columns:
        mcap_wide = df['Market_Cap'].unstack('Ticker').reindex(close.index)
        last_mcap = mcap_wide.apply(
            lambda col: col.dropna().iloc[-1] if col.dropna().size else np.nan
        )
        keep  = last_mcap[last_mcap >= min_mcap].index
        close = close[close.columns.intersection(keep)]

    # ── weights ───────────────────────────────────────────────────────────────
    if weighting == 'market_cap' and 'Market_Cap' in df.columns:
        mcap_wide = (
            df['Market_Cap']
            .unstack('Ticker')
            .reindex(close.index)[close.columns]
        )
        weights = mcap_wide.div(mcap_wide.sum(axis=1), axis=0)
    else:
        valid   = close.notna()
        n_valid = valid.sum(axis=1).replace(0, np.nan)
        weights = valid.div(n_valid, axis=0).astype(float)

    weights = weights.fillna(0.0)

    return close, weights
