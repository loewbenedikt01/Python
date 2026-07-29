"""
initial_test.py
===============
Cross-sectional momentum backtest with optional region and regime filters.

Strategy
--------
  Signal    : 12-month minus 1-month price return (Jegadeesh-Titman)
              score(t) = price[t-21] / price[t-252] - 1  (skips last month)
  Selection : top N_STOCKS by momentum score at each annual rebalance date
  Weighting : equal weight
  Rebalance : annual

Filters (toggle with USE_* flags below)
----------------------------------------
  USE_REGION_FILTER  — restrict universe to specific continents
  USE_REGIME_FILTER  — skip to cash when gated regimes are in blocked states
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[2]   # …/Python/
_DATABASE = _ROOT / 'Database'
_ENGINE   = _ROOT / 'Backtesting_engine'
_REGIMES  = _ROOT / 'Regimes'

sys.path.insert(0, str(_ENGINE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_REGIMES))

import risk_metrics
import plotting

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_NAME   = 'momentum_top20'
START_DATE      = '2010-01-01'
END_DATE        = '2025-12-31'
INITIAL_CAPITAL = 1_000_000

N_STOCKS         = 20    # stocks to hold per period
MOMENTUM_LONG    = 252   # lookback in trading days (~12 months)
MOMENTUM_SKIP    = 21    # skip recent days to avoid short-term reversal (~1 month)
MIN_HISTORY_DAYS = 300   # minimum price history for a stock to be eligible

# ── Equity file ───────────────────────────────────────────────────────────────
EQUITY_FILE = _DATABASE / 'equities.parquet'   # single combined file

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS  ←  toggle here
# ══════════════════════════════════════════════════════════════════════════════

USE_REGION_FILTER = False   # True → only include tickers from REGIONS below
USE_REGIME_FILTER = False   # True → go to cash when any gated regime is blocked

# ── Region filter ─────────────────────────────────────────────────────────────
# Active only when USE_REGION_FILTER = True.
# Match values from the 'Continent' column in the parquet files.
REGIONS = [
    'North America',
    'Europe',
    'Asia',
    # 'South America',
    # 'Oceania',
    # 'Africa',
]

# ── Regime gates ──────────────────────────────────────────────────────────────
# Active only when USE_REGIME_FILTER = True.
# On each rebalance date: if any gated regime (non-None) is NOT in 'allowed',
# the portfolio holds cash for that year instead of picking momentum stocks.
# Set a regime to None to compute it (for display) but not block allocation.
REGIME_GATES = {
    'equity':        None, #{'allowed': {'Bull', 'Neutral'}},   # gate: no Bear markets
    'bond':          None,
    'commodity':     None,
    'forex':         None,
    'crypto':        None,
    'growth':        None,   # e.g. {'allowed': {'Expansion', 'Recovery'}}
    'inflation':     None,
    'liquidity':     None,
    'risk_appetite': None,
    'hidden_markov': None,
}

# Column name that holds the regime label inside each regime's output DataFrame
_REGIME_COL = {
    'equity':        'global_regime',
    'bond':          'bond_regime',
    'commodity':     'commodity_regime',
    'forex':         'forex_regime',
    'crypto':        'crypto_regime',
    'growth':        'growth_regime',
    'inflation':     'inflation_regime',
    'liquidity':     'liquidity_regime',
    'risk_appetite': 'risk_regime',
    'hidden_markov': 'hmm_regime',
}


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


def load_continent_map() -> pd.Series:
    """Series mapping Ticker -> Continent, built from the parquet Continent column."""
    if not EQUITY_FILE.exists():
        return pd.Series(dtype=str)
    df = pd.read_parquet(EQUITY_FILE, columns=['Continent'])
    return df['Continent'].groupby(level='Ticker').last()


def load_regimes(equity_df: pd.DataFrame) -> dict:
    """
    Compute all regime signals. Only called when USE_REGIME_FILTER = True.
    Returns {name: DataFrame} — each DataFrame has a regime-label column.
    """
    from Simple.equity    import compute_equity_regime
    from Simple.bond      import compute_bond_regime
    from Simple.commodity import compute_commodity_regime
    from Simple.forex     import compute_forex_regime
    from Simple.crypto    import compute_crypto_regime
    from Multi_Model.growth        import compute_growth_regime
    from Multi_Model.inflation     import compute_inflation_regime
    from Multi_Model.liquidity     import compute_liquidity_regime
    from Multi_Model.risk_appetite import compute_risk_appetite_regime
    from Hidden_Markov_Model.hmm   import compute_hmm_regime

    bonds       = pd.read_parquet(_DATABASE / 'bonds.parquet')
    commodities = pd.read_parquet(_DATABASE / 'commodities.parquet')
    forex       = pd.read_parquet(_DATABASE / 'forex.parquet')
    crypto      = pd.read_parquet(_DATABASE / 'crypto.parquet')

    continent_dfs, equity_global = compute_equity_regime(equity_df)
    bond_df      = compute_bond_regime(bonds)
    commodity_df = compute_commodity_regime(commodities)
    forex_df     = compute_forex_regime(forex)
    crypto_df    = compute_crypto_regime(crypto)
    equity_regime = (continent_dfs, equity_global)

    growth_df = compute_growth_regime(
        equity_regime, bond_df, commodity_df, forex_df, macro_df=None
    )

    return {
        'equity':        equity_global,
        'bond':          bond_df,
        'commodity':     commodity_df,
        'forex':         forex_df,
        'crypto':        crypto_df,
        'growth':        growth_df,
        'inflation':     compute_inflation_regime(commodity_df, bond_df, macro_df=None),
        'liquidity':     compute_liquidity_regime(crypto_df, bond_df, forex_df),
        'risk_appetite': compute_risk_appetite_regime(
                             equity_global, bond_df, forex_df,
                             commodity_df, crypto_df, sentiment_df=None),
        'hidden_markov': compute_hmm_regime(
                             equity_global, bond_df, forex_df, commodity_df, crypto_df),
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGIME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_states(regimes: dict, date: pd.Timestamp) -> dict[str, str]:
    """Most-recent regime label for each module as of `date`."""
    states = {}
    for name, df in regimes.items():
        if df is None or df.empty:
            continue
        col = _REGIME_COL.get(name)
        if col and col in df.columns:
            past = df.loc[:date, col].dropna()
            if not past.empty:
                states[name] = past.iloc[-1]
    return states


def regimes_allow_allocation(states: dict[str, str]) -> tuple[bool, list[str]]:
    """
    Returns (allowed, blocked_list).
    allowed      = True when every gated regime is in its allowed set
    blocked_list = names of gated regimes that are currently blocking
    """
    blocked = []
    for name, gate in REGIME_GATES.items():
        if gate is None:
            continue
        state = states.get(name)
        if state is not None and state not in gate['allowed']:
            blocked.append(f'{name}={state}')
    return (len(blocked) == 0), blocked


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum(close: pd.DataFrame, date: pd.Timestamp,
                     eligible: list[str] | None = None) -> pd.Series:
    """
    12-1 month momentum for all (or eligible) tickers as of `date`.

    score = price[t - MOMENTUM_SKIP] / price[t - MOMENTUM_LONG] - 1

    Parameters
    ----------
    eligible : optional list of tickers to restrict scoring to (region filter)
    """
    cols    = close.columns if eligible is None else [t for t in eligible if t in close.columns]
    history = close.loc[:date, cols].dropna(axis=1, thresh=MIN_HISTORY_DAYS)

    if len(history) < MOMENTUM_LONG:
        return pd.Series(dtype=float)

    price_skip = history.iloc[-MOMENTUM_SKIP]
    price_long = history.iloc[-MOMENTUM_LONG]

    valid  = price_skip.notna() & price_long.notna() & (price_long > 0)
    scores = (price_skip[valid] / price_long[valid]) - 1

    return scores.sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(close: pd.DataFrame, allocations: list[tuple]) -> pd.Series:
    """
    Simulate portfolio NAV with equal-weight holdings between annual rebalances.
    Empty ticker lists (cash periods) hold value flat until next rebalance.

    Parameters
    ----------
    close       : wide price DataFrame (Date x Ticker)
    allocations : list of (rebalance_date, [ticker, ...]) — empty list = cash

    Returns
    -------
    Daily NAV Series indexed by Date.
    """
    nav_parts = []
    capital   = float(INITIAL_CAPITAL)

    for i, (rebal_date, tickers) in enumerate(allocations):
        next_date = (allocations[i + 1][0] if i + 1 < len(allocations)
                     else pd.Timestamp(END_DATE))

        # ── Cash period (blocked by regime) ──────────────────────────────────
        if not tickers:
            period_dates = close.loc[rebal_date:next_date].index
            if not period_dates.empty:
                nav_parts.append(pd.Series(capital, index=period_dates))
            continue

        # ── Invested period ───────────────────────────────────────────────────
        period = close.loc[rebal_date:next_date, tickers].dropna(how='all')
        if period.empty:
            continue

        prices_start = period.iloc[0].dropna()
        tickers_ok   = prices_start.index.tolist()
        if not tickers_ok:
            continue

        alloc_each = capital / len(tickers_ok)
        shares     = alloc_each / prices_start

        daily_nav = period[tickers_ok].mul(shares, axis=1).sum(axis=1)
        nav_parts.append(daily_nav)
        capital = float(daily_nav.iloc[-1])

    if not nav_parts:
        raise RuntimeError('No valid periods simulated.')

    nav = pd.concat(nav_parts)
    return nav.loc[~nav.index.duplicated(keep='last')].sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    flags = []
    if USE_REGION_FILTER: flags.append(f'regions={REGIONS}')
    if USE_REGIME_FILTER: flags.append('regime-gated')
    flag_str = '  [' + '  |  '.join(flags) + ']' if flags else '  [no filters]'

    print(f'=== Momentum Top-{N_STOCKS}  |  12M-1M  |  {START_DATE} -> {END_DATE} ===')
    print(flag_str + '\n')

    # ── Load prices ───────────────────────────────────────────────────────────
    print('Loading prices ...')
    close_all = load_close()
    close     = close_all.loc[START_DATE:END_DATE]

    # ── Region filter: build eligible ticker list ─────────────────────────────
    eligible = None   # None = all tickers
    if USE_REGION_FILTER:
        print('Loading continent map ...')
        continent_map = load_continent_map()
        eligible = continent_map[continent_map.isin(REGIONS)].index.tolist()
        eligible = [t for t in eligible if t in close.columns]
        print(f'  Region filter: {len(eligible)} tickers in {REGIONS}')

    print(f'  Universe: {close.shape[1]} tickers  |  {len(close)} trading days\n')

    # ── Regime filter: pre-compute regime signals ─────────────────────────────
    regimes = None
    if USE_REGIME_FILTER:
        print('Computing regimes (this may take a moment) ...')
        equity_df = pd.read_parquet(EQUITY_FILE)
        regimes   = load_regimes(equity_df)
        print()

    # ── Annual rebalance loop ─────────────────────────────────────────────────
    rebal_dates = list(pd.date_range(start=START_DATE, end=END_DATE, freq='BYS'))
    allocations = []

    print(f'{"Date":<14}  {"Status":<10}  {"Info"}')
    print('-' * 72)

    for date in rebal_dates:
        avail = close.loc[:date].index
        if avail.empty:
            continue
        actual = avail[-1]

        # ── Regime gate check ─────────────────────────────────────────────────
        if USE_REGIME_FILTER and regimes:
            states  = get_regime_states(regimes, actual)
            allowed, blocked = regimes_allow_allocation(states)
            if not allowed:
                allocations.append((actual, []))   # cash
                print(f'{str(actual.date()):<14}  {"CASH":<10}  blocked: {", ".join(blocked)}')
                continue

        # ── Momentum selection ────────────────────────────────────────────────
        scores = compute_momentum(close_all, actual, eligible=eligible)

        if len(scores) < N_STOCKS:
            allocations.append((actual, []))
            print(f'{str(actual.date()):<14}  {"SKIP":<10}  only {len(scores)} eligible tickers')
            continue

        top = scores.index[:N_STOCKS].tolist()
        allocations.append((actual, top))

        regime_str = ''
        if USE_REGIME_FILTER and regimes:
            states     = get_regime_states(regimes, actual)
            regime_str = '  ' + '  '.join(f'{k}={v}' for k, v in states.items()
                                           if REGIME_GATES.get(k) is not None)

        top5 = '  '.join(f'{t}({scores[t]:+.0%})' for t in top[:5])
        print(f'{str(actual.date()):<14}  {"INVEST":<10}  {top5} ...{regime_str}')

    print()

    # ── Simulate ──────────────────────────────────────────────────────────────
    print('Simulating portfolio ...')
    nav = simulate(close, allocations)

    portfolio_df = pd.DataFrame({
        'nav':          nav,
        'daily_return': nav.pct_change(),
    })

    # ── Risk metrics ──────────────────────────────────────────────────────────
    print('Computing risk metrics ...')
    metrics = risk_metrics.compute(portfolio_df)
    print(risk_metrics.summary(metrics))

    # ── Plots ─────────────────────────────────────────────────────────────────
    title = f'Momentum Top-{N_STOCKS}  (12M-1M, Annual Rebalance){flag_str}'
    plotting.plot(portfolio_df, metrics, regimes=regimes, title=title)

    return portfolio_df, metrics


if __name__ == '__main__':
    main()
