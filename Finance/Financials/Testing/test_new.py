"""
test_new.py
===========
Cross-sectional momentum backtest with optional region and regime filters.
Strategies are implemented in portfolio_new.py and selected via STRATEGY below.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[2]
_DATABASE = _ROOT / 'Database'
_ENGINE   = _ROOT / 'Backtesting_engine'
_REGIMES  = _ROOT / 'Regimes'

sys.path.insert(0, str(_ENGINE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_REGIMES))

import risk_metrics_new
import plotting_new
import portfolio_new

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_NAME    = 'momentum_top20'
START_DATE       = '2010-01-01'
END_DATE         = '2025-12-31'
INITIAL_CAPITAL  = 1_000_000

N_STOCKS         = 20    # stocks selected per rebalance
MIN_HISTORY_DAYS = 300   # minimum price history for a ticker to be eligible

# ── Equity file ───────────────────────────────────────────────────────────────
EQUITY_FILE = _DATABASE / 'equities.parquet'

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# Available names (defined in portfolio_new):
#   'Momentum'        12-month minus 1-month (Jegadeesh-Titman, default)
#   'Momentum_6_1'    6-month minus 1-month
#   'Mean_Reversion'  short-term 1-month reversal
# ══════════════════════════════════════════════════════════════════════════════
STRATEGY = [
    'Momentum',
    'Momentum_6_1',
    'Mean_Reversion'
]
# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS  ←  toggle here
# ══════════════════════════════════════════════════════════════════════════════

USE_UNIVERSE_FILTER = False   # True → restrict to UNIVERSE countries
USE_REGIME_FILTER   = False   # True → go to cash when gated regimes are blocked

# ── Universe filter ───────────────────────────────────────────────────────────
# Active only when USE_UNIVERSE_FILTER = True.
# Pick ONE from portfolio_new._UNIVERSES:
UNIVERSE = [
    'US',
    #'Europe',
    #'North America',
    #'Asia',
    #'Global',
    #'Developed Markets',
    #'Emerging Markets',
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_close() -> pd.DataFrame:
    """Wide Close price DataFrame: index = Date, columns = Ticker."""
    if not EQUITY_FILE.exists():
        raise FileNotFoundError(f'equities.parquet not found at {EQUITY_FILE}')
    df    = pd.read_parquet(EQUITY_FILE, columns=['Close'])
    close = df['Close'].unstack('Ticker')
    print(f'  Loaded equities.parquet  ({close.shape[1]} tickers)')
    return close


def _active_universe() -> str:
    """Return the first (uncommented) entry from UNIVERSE."""
    return UNIVERSE[0] if isinstance(UNIVERSE, list) else UNIVERSE


def load_eligible_tickers() -> list[str] | None:
    """Return tickers restricted to UNIVERSE, or None if filter is off."""
    u = _active_universe()
    if not USE_UNIVERSE_FILTER or u == 'Global':
        return None
    df          = pd.read_parquet(EQUITY_FILE, columns=['Country'])
    country_map = df['Country'].groupby(level='Ticker').last()
    allowed     = portfolio_new._UNIVERSES.get(u, set())
    return country_map[country_map.isin(allowed)].index.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(close: pd.DataFrame, allocations: list[tuple]) -> pd.Series:
    """
    Simulate portfolio NAV with equal-weight holdings between annual rebalances.
    Empty ticker lists (cash periods) hold value flat until next rebalance.
    """
    nav_parts = []
    capital   = float(INITIAL_CAPITAL)

    for i, (rebal_date, tickers) in enumerate(allocations):
        next_date = (allocations[i + 1][0] if i + 1 < len(allocations)
                     else pd.Timestamp(END_DATE))

        if not tickers:
            period_dates = close.loc[rebal_date:next_date].index
            if not period_dates.empty:
                nav_parts.append(pd.Series(capital, index=period_dates))
            continue

        period = close.loc[rebal_date:next_date, tickers].dropna(how='all')
        if period.empty:
            continue

        prices_start = period.iloc[0].dropna()
        tickers_ok   = prices_start.index.tolist()
        if not tickers_ok:
            continue

        alloc_each = capital / len(tickers_ok)
        shares     = alloc_each / prices_start
        daily_nav  = period[tickers_ok].mul(shares, axis=1).sum(axis=1)
        nav_parts.append(daily_nav)
        capital = float(daily_nav.iloc[-1])

    if not nav_parts:
        raise RuntimeError('No valid periods simulated.')

    nav = pd.concat(nav_parts)
    return nav.loc[~nav.index.duplicated(keep='last')].sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_strategy(strategy: str, close_all: pd.DataFrame, close: pd.DataFrame,
                      eligible: list[str] | None, flag_str: str):
    """Run the full backtest loop for a single strategy name. Returns (portfolio_df, metrics)."""
    rebal_dates = list(pd.date_range(start=START_DATE, end=END_DATE, freq='BYS'))
    allocations = []

    print(f'\n--- Strategy: {strategy} ---')
    print(f'{"Date":<14}  {"Status":<10}  {"Info"}')
    print('-' * 72)

    for date in rebal_dates:
        avail = close.loc[:date].index
        if avail.empty:
            continue
        actual = avail[-1]

        scores = portfolio_new.run_strategy(
            close_all, actual, strategy,
            eligible=eligible,
            min_history=MIN_HISTORY_DAYS,
        )

        if len(scores) < N_STOCKS:
            allocations.append((actual, []))
            print(f'{str(actual.date()):<14}  {"SKIP":<10}  only {len(scores)} eligible')
            continue

        top = scores.index[:N_STOCKS].tolist()
        allocations.append((actual, top))
        top5 = '  '.join(f'{t}({scores[t]:+.0%})' for t in top[:5])
        print(f'{str(actual.date()):<14}  {"INVEST":<10}  {top5} ...')

    print()
    nav = simulate(close, allocations)
    portfolio_df = pd.DataFrame({'nav': nav, 'daily_return': nav.pct_change()})

    metrics = risk_metrics_new.compute(portfolio_df)
    print(risk_metrics_new.summary(metrics))

    title = f'{strategy}  Top-{N_STOCKS}{flag_str}'
    plotting_new.plot(portfolio_df, metrics, title=title)
    return portfolio_df, metrics


def main():
    universe  = _active_universe()
    strategies = STRATEGY if isinstance(STRATEGY, list) else [STRATEGY]

    filters = []
    if USE_UNIVERSE_FILTER: filters.append(f'universe={universe}')
    if USE_REGIME_FILTER:   filters.append('regime-gated')
    flag_str = '  [' + '  |  '.join(filters) + ']' if filters else '  [no filters]'

    print(f'=== Top-{N_STOCKS}  |  {START_DATE} -> {END_DATE} ===')
    print(flag_str + '\n')

    # ── Load prices ───────────────────────────────────────────────────────────
    print('Loading prices ...')
    close_all = load_close()
    close     = close_all.loc[START_DATE:END_DATE]

    # ── Universe filter ───────────────────────────────────────────────────────
    eligible = load_eligible_tickers()
    if eligible is not None:
        eligible = [t for t in eligible if t in close.columns]
        print(f'  Universe filter ({universe}): {len(eligible)} tickers')

    print(f'  Universe: {close.shape[1]} tickers  |  {len(close)} trading days')

    results = {}
    for strat in strategies:
        results[strat] = _run_one_strategy(strat, close_all, close, eligible, flag_str)

    return results


if __name__ == '__main__':
    main()
