





import pandas as pd
import numpy as np

# ── Universe definitions ──────────────────────────────────────────────────────
_DM_ROTW = {'Australia', 'Canada', 'Switzerland', 'United Kingdom'}
_DM_ASIA = {'Hong Kong', 'Japan', 'Singapore', 'South Korea'}


# ── Paths ─────────────────────────────────────────────────────────────────────
DATABASE_DIR = r'C:\Users\benel\Coding\Python\Database'

PARQUET_FILES = {
    'macro':        'macro.parquet',
    'us':           'US_equities.parquet',
    'de':           'DE_equities.parquet',
    'eu':           'EU_equities.parquet',
    'asia':         'ASIA_equities.parquet',
    'rotw':         'ROTW_equities.parquet',
    'bonds':        'bonds.parquet',
    'forex':        'forex.parquet',
    'commodities':  'commodities.parquet',
    'crypto':       'crypto.parquet',
    'sentiment':    'sentiment.parquet',
    'sectors':      'sectors.parquet',
    'indices':      'indices.parquet',
}

# ── Equity Regime ─────────────────────────────────────────────────────────────
EQUITY_BULL_THRESHOLD  = 0.60
EQUITY_BEAR_THRESHOLD  = 0.50
EQUITY_MIN_STOCKS      = 5     # min tickers with valid MA200 to trust a continent's breadth
EQUITY_MA_WINDOW       = 200
EQUITY_SMOOTH_WINDOW   = 10
EQUITY_TREND_WINDOW    = 63
EQUITY_CONFIRM_WINDOW  = 7
EQUITY_CONFIRM_MIN     = 5

# ── Bond Regime ───────────────────────────────────────────────────────────────
BOND_BULL_THRESHOLD         = 3.0
BOND_BEAR_THRESHOLD         = 1.0
BOND_MA200_WINDOW              = 200
BOND_MA50_WINDOW            = 50
BOND_TREND_WINDOW           = 63

BOND_SCORE_SMOOTH           = 20   # bond signals are slow; wider smooth before hysteresis
BOND_CONFIRM_WINDOW         = 21   # monthly window — bond regimes hold for months not days
BOND_CONFIRM_MIN            = 16   # ~76% agreement



BOND_CURVE_STEEP_THRESHOLD  = 1.0
BOND_CURVE_FLAT_THRESHOLD   = -0.5
BOND_MOMENTUM_THRESHOLD     = 0.5   # % change in 10yr yield over BOND_MOMENTUM_WINDOW
BOND_REAL_YIELD_HIGH        = 1.02  # TIP/IEF ratio above this → real yields falling
BOND_REAL_YIELD_LOW         = 0.98  # TIP/IEF ratio below this → real yields rising
BOND_CURVE_SMOOTH           = 5
BOND_MOMENTUM_SMOOTH        = 10
BOND_MOMENTUM_WINDOW        = 63    # days for rate momentum diff
BOND_REAL_YIELD_SMOOTH      = 10
BOND_REAL_YIELD_NORM_WINDOW = 252
BOND_TLT_SMOOTH             = 5
BOND_SHY_REL_WINDOW         = 50

# ── Forex Regime ──────────────────────────────────────────────────────────────
FOREX_MA50_WINDOW          = 50
FOREX_MA200_WINDOW         = 200
FOREX_MOMENTUM_WINDOW      = 63    # days for rate-of-change signals
FOREX_STRENGTH_WINDOW      = 20    # days for cross-pair pct change in currency strength
FOREX_SCORE_SMOOTH         = 15    # smooth 0-N score before hysteresis
FOREX_EM_NORM_WINDOW       = 252   # days for EM z-score normalisation
FOREX_USD_BULL_ENTRY       = 2.5   # smoothed score >= 2.5/3 → USD Bull
FOREX_USD_BEAR_ENTRY       = 0.5
FOREX_CARRY_BULL_ENTRY     = 3.0   # weighted score: level*2 + momentum + nzdjpy (0-4)
FOREX_CARRY_BEAR_ENTRY     = 1.0
FOREX_JPY_BULL_ENTRY       = 2.5   # JPY Bull = risk-off (safe-haven demand)
FOREX_JPY_BEAR_ENTRY       = 0.5
FOREX_EUROPE_BULL_ENTRY    = 2.5
FOREX_EUROPE_BEAR_ENTRY    = 0.5
FOREX_EM_STRESS_THRESHOLD  = 1.0   # z-score above this → EM Stress
FOREX_EM_RELIEF_THRESHOLD  = -0.5  # z-score below this → EM Relief
FOREX_CONFIRM_WINDOW       = 7    # sub-regime confirmation
FOREX_CONFIRM_MIN          = 5
FOREX_MASTER_CONFIRM_WINDOW = 15  # master regime needs wider window — two sub-regimes must agree
FOREX_MASTER_CONFIRM_MIN    = 11  # ~75% agreement
FOREX_MIN_PAIRS            = 10   # min pairs with valid data before master regime is assigned

# ── Commodity Regime ──────────────────────────────────────────────────────────
COMMODITY_MA_WINDOW       = 200
COMMODITY_SMOOTH_WINDOW   = 20
COMMODITY_MOMENTUM_WINDOW = 63
COMMODITY_BASKET_WINDOW   = 63   # quarterly return for broad CRB trend
COMMODITY_SCORE_SMOOTH    = 20
COMMODITY_BULL_ENTRY      = 2.7   # 2.7/4 signals needed to enter Bull (~68%)
COMMODITY_BULL_EXIT       = 2.1   # exit Bull when score < 2.1 (sooner than 2.0 → more Neutral)
COMMODITY_BEAR_ENTRY      = 1.3   # 1.3/4 signals → Bear (raises threshold from 0.8)
COMMODITY_BEAR_EXIT       = 1.9   # exit Bear when score > 1.9 (sooner than 2.0 → more Neutral)
COMMODITY_CONFIRM_WINDOW  = 15
COMMODITY_CONFIRM_MIN     = 11

# ── Crypto Regime ─────────────────────────────────────────────────────────────
CRYPTO_BULL_ENTRY        = 3.5  # smoothed score must reach >= 3.5 to enter Bull
CRYPTO_BULL_EXIT         = 2.5  # exit Bull below 2.5 → Neutral (not Bear directly)
CRYPTO_BEAR_ENTRY        = 1.5  # smoothed score must drop <= 1.5 to enter Bear
CRYPTO_BEAR_EXIT         = 2.5  # exit Bear above 2.5 → Neutral (not Bull directly)
CRYPTO_MA_WINDOW         = 200
CRYPTO_BTC_SMOOTH        = 5    # smooth BTC price before MA200 comparison
CRYPTO_ETH_SMOOTH        = 5
CRYPTO_MOMENTUM_WINDOW   = 63   # days for BTC return signal
CRYPTO_MOMENTUM_SMOOTH   = 10
CRYPTO_ETH_BTC_MA        = 50   # MA window for ETH/BTC ratio trend
CRYPTO_SMOOTH_WINDOW     = 10
CRYPTO_SCORE_SMOOTH      = 10   # smooth the 0-4 bull score before hysteresis (wider than bonds)
CRYPTO_CONFIRM_WINDOW    = 10   # wider than equities/bonds — crypto is 5-10x more volatile
CRYPTO_CONFIRM_MIN       = 7

# ── Growth Regime (Multi-Model) ───────────────────────────────────────────────
GROWTH_THRESHOLD          = 0.55   # normalised score above this = above-trend growth
GROWTH_SCORE_SMOOTH       = 21     # monthly smoothing — growth is slow-moving
GROWTH_DELTA_WINDOW       = 21     # monthly delta for direction (accelerating/decelerating)
GROWTH_CONFIRM_WINDOW     = 21     # monthly confirmation window
GROWTH_CONFIRM_MIN        = 15     # 71% agreement before regime switch
GROWTH_WEIGHT_YIELD_CURVE = 2.0   # most reliable leading indicator (6-12mo lead)
GROWTH_WEIGHT_COPPER_GOLD = 2.0   # 3-6mo lead — best real-time growth proxy
GROWTH_WEIGHT_EQUITY      = 1.5   # coincident — majority of stocks in uptrend
GROWTH_WEIGHT_BOND        = 1.0   # coincident — credit conditions supportive
GROWTH_WEIGHT_USD         = 1.0   # coincident-lagging — weak dollar = global growth supportive
GROWTH_WEIGHT_EM          = 0.5   # coincident — no EM stress = global growth intact
GROWTH_WEIGHT_PMI         = 1.5   # coincident — only used when macro_df has PMI

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_COMPONENTS    = 4
HMM_N_ITER          = 200
HMM_COVARIANCE_TYPE = 'full'


_BOND_REGIME_TO_NUM = {'Bond Bull': 2, 'Bond Neutral': 1, 'Bond Bear': 0, 'Unknown': -1}
_BOND_NUM_TO_REGIME = {v: k for k, v in _BOND_REGIME_TO_NUM.items()}
_COMMODITY_REGIME_TO_NUM = {'Commodity Bull': 2, 'Commodity Neutral': 1, 'Commodity Bear': 0, 'Unknown': -1}
_COMMODITY_NUM_TO_REGIME = {v: k for k, v in _COMMODITY_REGIME_TO_NUM.items()}


def _hysteresis(score_series: pd.Series, bull_entry: float, bear_entry: float,
                bull_exit: float | None = None, bear_exit: float | None = None) -> list:
    """Generic hysteresis filter — returns 'Bull' / 'Neutral' / 'Bear' / 'Unknown'.
    bull_exit / bear_exit set the dead-band: once in Bull the regime only exits when
    the score drops *below* bull_exit (≤ bull_entry). Omit for a symmetric threshold.
    """
    bull_exit = bull_exit if bull_exit is not None else bull_entry
    bear_exit = bear_exit if bear_exit is not None else bear_entry
    regimes, state = [], None

    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue

        if state is None:
            if   score >= bull_entry: state = 'Bull'
            elif score <= bear_entry: state = 'Bear'
            else:                     state = 'Neutral'
        elif state == 'Bull':
            if   score <= bear_entry: state = 'Bear'
            elif score <  bull_exit:  state = 'Neutral'
        elif state == 'Bear':
            if   score >= bull_entry: state = 'Bull'
            elif score >  bear_exit:  state = 'Neutral'
        else:  # Neutral
            if   score >= bull_entry: state = 'Bull'
            elif score <= bear_entry: state = 'Bear'
        regimes.append(state)
    return regimes

def _equity_regime(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Returns
    -------
    continent_dfs : dict {continent_name → DataFrame}
        Each DataFrame (Date index) has columns:
          n_available    — tickers with a valid MA on that day
          n_above_ma     — tickers whose close > MA
          breadth        — n_above_ma / n_available  (0–1 decimal, NaN if < EQUITY_MIN_STOCKS)
          breadth_smooth — EQUITY_SMOOTH_WINDOW rolling mean of breadth
          score          — 1 (Bull) | 0 (Neutral) | -1 (Bear)

    global_df : DataFrame
          global_score   — 1 | 0 | -1
          global_regime  — 'Bull' | 'Neutral' | 'Bear' | 'Unknown'
    """
    close         = df['Close'].unstack('Ticker')
    continent_map = df['Continent'].groupby(level='Ticker').last()

    # ── MA per ticker (one pass for all tickers) ──────────────────────────────
    ma = (
        df['Close'].dropna()
        .groupby(level='Ticker')
        .transform(lambda s: s.rolling(EQUITY_MA_WINDOW, min_periods=EQUITY_MA_WINDOW).mean())
        .reindex(df.index)
        .unstack('Ticker')
    )
    valid = ma.notna()           # ticker has enough history for MA on this day
    above = (close > ma) & valid # ticker is above its MA on this day

    continent_dfs = {}
    score_matrix  = pd.DataFrame(index=close.index)  # one score column per continent

    # ── Per-continent breadth + hysteresis ────────────────────────────────────
    for continent in sorted(continent_map.unique()):
        tickers = continent_map[continent_map == continent].index.intersection(close.columns)
        if tickers.empty:
            continue

        n_available = valid[tickers].sum(axis=1)
        n_above_ma  = above[tickers].sum(axis=1)

        # Breadth: ignore days where too few stocks have MA history (e.g. early data)
        breadth = (
            (n_above_ma / n_available.replace(0, np.nan))
            .where(n_available >= EQUITY_MIN_STOCKS)
            .ffill(limit=3)   # bridge 1-3 day holiday gaps without masking real moves
        )
        breadth_smooth = (
            breadth.dropna()
            .rolling(EQUITY_SMOOTH_WINDOW, min_periods=EQUITY_SMOOTH_WINDOW // 2)
            .mean()
            .reindex(close.index)
        )

        raw = pd.Series(
            _hysteresis(
                breadth_smooth,
                bull_entry=EQUITY_BULL_THRESHOLD,
                bear_entry=EQUITY_BEAR_THRESHOLD,
            ),
            index=close.index,
        ).map({'Bull': 1, 'Neutral': 0, 'Bear': -1, 'Unknown': np.nan})

        # Confirmation: only switch regime when the new state holds for at least
        # EQUITY_CONFIRM_MIN of the last EQUITY_CONFIRM_WINDOW trading days.
        # Runs on the non-NaN subset so weekends don't dilute the window.
        trading_idx = breadth_smooth.dropna().index
        confirmed = (
            raw.loc[trading_idx]
            .rolling(EQUITY_CONFIRM_WINDOW)
            .apply(lambda x: x[-1] if (x == x[-1]).sum() >= EQUITY_CONFIRM_MIN else np.nan, raw=True)
            .ffill()
            .reindex(close.index)
            .ffill()   # forward-fill weekends
        )

        continent_dfs[continent] = pd.DataFrame({
            'n_available':    n_available,
            'n_above_ma':     n_above_ma,
            'breadth':        breadth,
            'breadth_smooth': breadth_smooth,
            'score':          confirmed,
        })
        score_matrix[continent] = confirmed

    # ── Global regime ─────────────────────────────────────────────────────────
    def _global_score(row: pd.Series) -> float:
        known = row.dropna()
        if known.empty:
            return np.nan
        if (known == 1).all():
            return  1.0
        if (known == -1).all():
            return -1.0
        return 0.0

    global_score  = score_matrix.apply(_global_score, axis=1)
    global_regime = global_score.map({1.0: 'Bull', 0.0: 'Neutral', -1.0: 'Bear'}).fillna('Unknown')
    global_df     = pd.DataFrame({'global_score': global_score, 'global_regime': global_regime})

    return continent_dfs, global_df




# Universe construction

def _select_universe(data: dict, universe: str) -> pd.DataFrame:
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
    data:         dict,
    universe:     str,
    weighting:    str         = 'equal',       # 'equal' | 'market_cap'
    regimes:      dict | None = None,          # {'equity': equity_df, 'growth': growth_df, ...}
    regime_rules: dict | None = None,          # override _DEFAULT_REGIME_RULES
    start_date:   str  | None = None,
    end_date:     str  | None = None,
    min_mcap:     float | None = None,
) -> dict:
    """
    Build a portfolio from a universe of stocks, optionally scaled by regime signals.

    Parameters
    ----------
    data         : dict of raw DataFrames — keys: 'us', 'eu', 'de', 'asia', 'rotw', ...
                   Each has a (Date, Ticker) MultiIndex and at minimum a Close column.
    universe     : 'US' | 'North America' | 'Europe' | 'Asia' |
                   'Developed Markets' | 'Emerging Markets' | 'Global'
    weighting    : 'equal'      — 1/N on each date
                   'market_cap' — proportional to Market_Cap column
    regimes      : dict of regime DataFrames from the Regimes module, e.g.
                   {'equity': equity_df, 'growth': growth_df, 'bond': bond_df}
                   Multiple regimes are multiplied together to get a combined exposure.
    regime_rules : map from regime state string → exposure scalar (0.0–1.0).
                   Defaults to _DEFAULT_REGIME_RULES. Pass your own to override.
    start_date   : clip history from this date (YYYY-MM-DD)
    end_date     : clip history to this date (YYYY-MM-DD)
    min_mcap     : exclude tickers whose most-recent Market_Cap is below this value

    Returns
    -------
    dict with keys:
      'close'    — pd.DataFrame wide (Date x Ticker), adjusted close prices
      'weights'  — pd.DataFrame wide (Date x Ticker), final weights (regime-adjusted)
      'exposure' — pd.Series, daily total portfolio exposure (1.0 = fully invested)
      'regimes'  — pd.DataFrame, one column per regime showing the active state per day
    """
    rules = {**_DEFAULT_REGIME_RULES, **(regime_rules or {})}

    # ── 1. Universe selection ─────────────────────────────────────────────────
    df = _select_universe(data, universe)
    if df.empty:
        raise ValueError(f"No data found for universe '{universe}'.")

    close = df['Close'].unstack('Ticker')

    # ── 2. Date clip ──────────────────────────────────────────────────────────
    if start_date:
        close = close.loc[start_date:]
    if end_date:
        close = close.loc[:end_date]

    # ── 3. Market-cap filter ──────────────────────────────────────────────────
    if min_mcap is not None and 'Market_Cap' in df.columns:
        mcap_wide = df['Market_Cap'].unstack('Ticker').reindex(close.index)
        last_mcap = mcap_wide.apply(
            lambda col: col.dropna().iloc[-1] if col.dropna().size else np.nan
        )
        keep  = last_mcap[last_mcap >= min_mcap].index
        close = close[close.columns.intersection(keep)]

    # ── 4. Base weights (before regime adjustment) ────────────────────────────
    if weighting == 'market_cap' and 'Market_Cap' in df.columns:
        mcap_wide = (
            df['Market_Cap']
            .unstack('Ticker')
            .reindex(close.index)[close.columns]
        )
        base_weights = mcap_wide.div(mcap_wide.sum(axis=1), axis=0).fillna(0.0)
    else:
        valid        = close.notna()
        n_valid      = valid.sum(axis=1).replace(0, np.nan)
        base_weights = valid.div(n_valid, axis=0).fillna(0.0)

    # ── 5. Regime exposure scaling ────────────────────────────────────────────
    regime_states = pd.DataFrame(index=close.index)
    exposure      = pd.Series(1.0, index=close.index)   # start fully invested

    if regimes:
        for name, regime_df in regimes.items():
            col = _REGIME_COL.get(name)
            if col is None or regime_df is None or col not in regime_df.columns:
                continue

            # align regime series to portfolio index, forward-fill gaps (weekends etc.)
            state = regime_df[col].reindex(close.index).ffill()
            scale = state.map(rules).fillna(rules.get('Unknown', 0.6))

            regime_states[name] = state
            exposure = exposure * scale   # multiply exposures across all regime signals

    weights = base_weights.multiply(exposure, axis=0)

    return {
        'close':    close,
        'weights':  weights,
        'exposure': exposure,
        'regimes':  regime_states,
    }
