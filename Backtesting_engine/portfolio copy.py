import pandas as pd
import numpy as np

# ── Universe definitions ──────────────────────────────────────────────────────
_DM_ROTW = {'Australia', 'Canada', 'Switzerland', 'United Kingdom'}
_DM_ASIA = {'Hong Kong', 'Japan', 'Singapore', 'South Korea'}


# ── Date Range ────────────────────────────────────────────────────────────────
START_DATA  = '2000-01-01'   # data loaded from here (warmup for MAs)
START_DATE  = '2001-01-01'   # regime analysis starts here (post warmup)
END_DATE    = '2025-12-31'

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
EQUITY_REGION_WEIGHTS = {
    'north_america': 0.63,
    'europe':        0.17,
    'asia':          0.13,
    'south_america': 0.04,
    'oceania':       0.03,
}
EQUITY_BULL_THRESHOLD  = 0.60
EQUITY_BEAR_THRESHOLD  = 0.50
EQUITY_MIN_STOCKS      = 5     # min tickers with valid MA200 to trust a continent's breadth
EQUITY_MA_WINDOW       = 200
EQUITY_SMOOTH_WINDOW   = 10
EQUITY_TREND_WINDOW    = 63
EQUITY_CONFIRM_WINDOW  = 7
EQUITY_CONFIRM_MIN     = 5

# ── Bond Regime ───────────────────────────────────────────────────────────────
BOND_BULL_ENTRY             = 3.0   # smoothed score must reach ≥ 3 to enter Bull
BOND_BULL_EXIT              = 2.0   # exit Bull only when score drops below 2 (1-unit dead band)
BOND_BEAR_ENTRY             = 1.0   # smoothed score must drop ≤ 1 to enter Bear
BOND_BEAR_EXIT              = 2.0   # exit Bear only when score rises above 2
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
BOND_MA50_WINDOW            = 50
BOND_MA200_WINDOW           = 200
BOND_SHY_REL_WINDOW         = 50
BOND_SCORE_SMOOTH           = 20   # bond signals are slow; wider smooth before hysteresis
BOND_CONFIRM_WINDOW         = 21   # monthly window — bond regimes hold for months not days
BOND_CONFIRM_MIN            = 16   # ~76% agreement

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

# ── Colours ───────────────────────────────────────────────────────────────────
REGIME_COLORS = {
    # Equity (3-state — matches bond and crypto structure)
    'Bull':    '#2ecc71',
    'Neutral': '#ffc04d',
    'Bear':    '#e74c3c',
    # Bond
    'Bond Bull':          '#2ecc71',
    'Bond Neutral':       '#ffc04d',
    'Bond Bear':          '#e74c3c',
    # Master
    'Goldilocks':         '#2ecc71',
    'Overheating':        '#f39c12',
    'Stagflation':        '#e74c3c',
    'Deflation':          '#3498db',
    'Transition':         '#7f8c8d',
    # Generic
    'Unknown':            '#555577',
}

CURVE_COLORS = {
    'Steep':    '#2ecc71',
    'Normal':   '#3498db',
    'Flat':     '#ffc04d',
    'Inverted': '#e74c3c',
}

_REGIME_TO_NUM = {'Bull': 2, 'Neutral': 1, 'Bear': 0, 'Unknown': -1}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _apply_hysteresis(smooth: pd.Series) -> list:
    regimes = []
    state   = None
    mid     = (EQUITY_BULL_THRESHOLD + EQUITY_BEAR_THRESHOLD) / 2

    for val in smooth:
        if pd.isna(val):
            regimes.append('Unknown')
            continue

        if state is None:
            state = 'Bull' if val >= mid else 'Bear'

        elif state == 'Bull':
            if   val <  EQUITY_BEAR_THRESHOLD: state = 'Bear'
            elif val <  EQUITY_BULL_THRESHOLD: state = 'Neutral'

        elif state == 'Bear':
            if   val >= EQUITY_BULL_THRESHOLD: state = 'Bull'
            elif val >= EQUITY_BEAR_THRESHOLD: state = 'Neutral'

        else:  # Neutral
            if   val >= EQUITY_BULL_THRESHOLD: state = 'Bull'
            elif val <  EQUITY_BEAR_THRESHOLD: state = 'Bear'

        regimes.append(state)

    return regimes


def compute_equity_regime(df: pd.DataFrame) -> pd.DataFrame:
    close         = df['Close'].unstack('Ticker')
    continent_map = df['Continent'].groupby(level='Ticker').last()

    # dropna before groupby: combined parquet has NaN-Close rows for cross-market
    # non-trading days; one NaN inside the window makes the whole rolling result NaN.
    close_valid = df['Close'].dropna()
    ma200 = (
        close_valid
        .groupby(level='Ticker')
        .transform(lambda s: s.rolling(EQUITY_MA_WINDOW, min_periods=EQUITY_MA_WINDOW).mean())
        .reindex(df.index)
        .unstack('Ticker')
    )
    valid = ma200.notna()
    above = (close > ma200) & valid

    regime_df   = pd.DataFrame(index=close.index)
    region_cols = []

    for cont in sorted(continent_map.unique()):
        col     = 'breadth_' + cont.lower().replace(' ', '_')
        tickers = continent_map[continent_map == cont].index.intersection(close.columns)
        if tickers.empty:
            continue
        n_above = above[tickers].sum(axis=1)
        n_valid = valid[tickers].sum(axis=1)
        # EQUITY_MIN_STOCKS guards against holiday spikes (1-4 stocks trading → spurious 0% or 100%)
        regime_df[col] = (n_above / n_valid.replace(0, np.nan)).where(n_valid >= EQUITY_MIN_STOCKS)
        region_cols.append(col)

    # ffill(limit=3): a region dipping below EQUITY_MIN_STOCKS for 1-3 days (public holiday)
    # produces a NaN spike that the rolling window treats as a genuine data point.
    # Filling forward up to 3 days bridges the gap without masking real regime changes.
    for col in region_cols:
        regime_df[col] = regime_df[col].ffill(limit=3)

    # Adaptive smoothing: small-universe regions (South America, Oceania) swing
    # 10pp per extra stock crossing its MA. Apply a longer window for regions with
    # fewer than 50 tickers to reduce noise without masking real trend changes.
    for col in region_cols:
        cont_name = col.replace('breadth_', '').replace('_', ' ').title()
        n = len(continent_map[continent_map == cont_name].index)
        window = 10 if n >= 50 else 21
        regime_df[col] = (
            regime_df[col].dropna()
            .rolling(window, min_periods=max(3, window // 3))
            .mean()
            .reindex(regime_df.index)
        )

    weighted    = pd.Series(0.0, index=regime_df.index)
    weight_used = pd.Series(0.0, index=regime_df.index)
    for cont, w in EQUITY_REGION_WEIGHTS.items():
        col = 'breadth_' + cont
        if col not in regime_df.columns:
            continue
        mask         = regime_df[col].notna()
        weighted    += regime_df[col].fillna(0) * w * mask
        weight_used += w * mask
    regime_df['breadth_weighted'] = weighted / weight_used.replace(0, np.nan)

    # Roll on trading-day-only subset: NaN weekends inside a 10-row window would
    # prevent min_periods from ever being met on the full calendar index.
    bw_td  = regime_df['breadth_weighted'].dropna()
    smooth = bw_td.rolling(EQUITY_SMOOTH_WINDOW, min_periods=EQUITY_SMOOTH_WINDOW // 2).mean()
    trend  = bw_td.rolling(EQUITY_TREND_WINDOW,  min_periods=EQUITY_TREND_WINDOW  // 2).mean()

    # 31-day delta (half of 63d trend window) with a 1pp deadband: avoids the sign
    # flipping constantly near peaks/troughs where a 10-day diff ticks ± every session.
    delta_long = smooth.diff(EQUITY_TREND_WINDOW // 2)

    regime_df['breadth_smooth'] = smooth.reindex(regime_df.index)
    regime_df['breadth_trend']  = trend.reindex(regime_df.index)
    regime_df['breadth_delta']  = delta_long.reindex(regime_df.index)

    # Direction kept as metadata — monthly slope of the 63d trend MA.
    # Shown in the plot as an indicator but does NOT drive regime state.
    trend_slope = regime_df['breadth_trend'].diff(21)
    regime_df['breadth_direction'] = np.where(
        trend_slope > 0, 'Rising', 'Falling'
    )

    raw_regime = _apply_hysteresis(regime_df['breadth_smooth'])
    regime_df['equity_regime_raw'] = raw_regime

    # Confirmation window on trading-day index so weekends don't dilute the 5-row window.
    # Order matters: ffill on trading days first (fills intra-week gaps in the confirmed
    # series), then reindex to the calendar index, then ffill only to bridge weekends.
    # Mapping to regime labels happens after reindex so the ffill works on integers not strings.
    raw_series    = pd.Series(raw_regime, index=regime_df.index)
    trading_idx   = regime_df['breadth_smooth'].dropna().index
    regime_num_td = raw_series.loc[trading_idx].map(_REGIME_TO_NUM).astype(float)
    confirmed_td  = (
        regime_num_td
        .rolling(EQUITY_CONFIRM_WINDOW)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= EQUITY_CONFIRM_MIN else np.nan, raw=True)
        .ffill()                          # fill on trading days only
    )
    regime_df['equity_regime'] = (
        confirmed_td
        .map(_NUM_TO_REGIME)
        .reindex(regime_df.index)         # extend to calendar index (weekends → NaN)
        .ffill()                          # fill only those weekend gaps
        .fillna('Unknown')
    )

    regime_df['regional_dispersion'] = regime_df[region_cols].std(axis=1)
    regime_df['regime_conviction']   = np.select(
        [regime_df['regional_dispersion'] < 0.08,
         regime_df['regional_dispersion'] < 0.15],
        ['High', 'Medium'], default='Low',
    )

    if 'breadth_north_america' in regime_df.columns and 'breadth_europe' in regime_df.columns:
        regime_df['us_eu_divergence'] = (
            regime_df['breadth_north_america'] - regime_df['breadth_europe']
        )

    # ── snapshot print ────────────────────────────────────────────────────────
    valid_rows = regime_df.dropna(subset=['breadth_smooth'])
    latest     = valid_rows.iloc[-1]
    date_str   = valid_rows.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Equity Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Regime:":<28} {latest["equity_regime"]}')
    print(f'  {"Conviction:":<28} {latest["regime_conviction"]}')
    print(f'  {"Weighted Breadth:":<28} {latest["breadth_weighted"]:.1%}')
    print(f'  {"Breadth Trend (63d avg):":<28} {latest["breadth_trend"]:.1%}')
    print(f'  {"Direction (metadata):":<28} {latest.get("breadth_direction", "?")}')

    print('\n  --- Breadth by Continent (% above 200-MA) ---')
    for col in sorted(region_cols):
        val = latest.get(col, np.nan)
        if pd.isna(val):
            continue
        print(f'  {col.replace("breadth_","").replace("_"," ").title():<22} {val:>5.1%}  {"#"*int(val*20)}')

    if 'us_eu_divergence' in latest.index and not pd.isna(latest['us_eu_divergence']):
        print(f'\n  US - EU Divergence:  {latest["us_eu_divergence"]:+.1%}')

    print('\n  --- Regime Distribution (full history) ---')
    confirmed = valid_rows[valid_rows['equity_regime'] != 'Unknown']['equity_regime']
    dist      = confirmed.value_counts(normalize=True)
    for r in ['Bull', 'Neutral', 'Bear']:
        pct = dist.get(r, 0.0)
        print(f'  {r:<22} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return regime_df

_BOND_REGIME_TO_NUM = {'Bond Bull': 2, 'Bond Neutral': 1, 'Bond Bear': 0, 'Unknown': -1}
_BOND_NUM_TO_REGIME = {v: k for k, v in _BOND_REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_bond_hysteresis(score_series: pd.Series) -> list:
    regimes = []
    state   = None

    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue

        if state is None:
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score <= BOND_BEAR_ENTRY: state = 'Bond Bear'
            else:                          state = 'Bond Neutral'

        elif state == 'Bond Bull':
            # Exit Bull only when score drops to BOND_BULL_EXIT (1 unit below entry)
            # prevents exiting on small dips that don't represent a real regime change
            if   score <= BOND_BEAR_ENTRY: state = 'Bond Bear'
            elif score <  BOND_BULL_EXIT:  state = 'Bond Neutral'

        elif state == 'Bond Bear':
            # Exit Bear only when score rises to BOND_BEAR_EXIT (1 unit above entry)
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score >  BOND_BEAR_EXIT:  state = 'Bond Neutral'

        else:  # Bond Neutral — still requires full entry threshold to commit
            if   score >= BOND_BULL_ENTRY: state = 'Bond Bull'
            elif score <= BOND_BEAR_ENTRY: state = 'Bond Bear'

        regimes.append(state)

    return regimes


def compute_bond_regime(df_bonds: pd.DataFrame) -> pd.DataFrame:
    close   = df_bonds['Close'].unstack('Ticker')
    bond_df = pd.DataFrame(index=close.index)

    irx = close['^IRX']
    fvx = close['^FVX']
    tnx = close['^TNX']
    tyx = close['^TYX']

    bond_df['curve_2s10s']        = tnx - irx
    bond_df['curve_5s30s']        = tyx - fvx
    bond_df['curve_level']        = tnx
    bond_df['curve_2s10s_smooth'] = _smooth(bond_df['curve_2s10s'], window=BOND_CURVE_SMOOTH)

    bond_df['curve_regime'] = np.select([
        bond_df['curve_2s10s_smooth'] >  BOND_CURVE_STEEP_THRESHOLD,
        bond_df['curve_2s10s_smooth'] >  0,
        bond_df['curve_2s10s_smooth'] >  BOND_CURVE_FLAT_THRESHOLD,
    ], ['Steep', 'Normal', 'Flat'], default='Inverted')

    bond_df['rate_ma50']  = _smooth(tnx, window=BOND_MA50_WINDOW)
    bond_df['rate_ma200'] = _smooth(tnx, window=BOND_MA200_WINDOW)

    bond_df['rate_momentum']        = tnx.dropna().diff(BOND_MOMENTUM_WINDOW).reindex(close.index)
    bond_df['rate_momentum_smooth'] = _smooth(bond_df['rate_momentum'], window=BOND_MOMENTUM_SMOOTH)

    bond_df['rate_trend'] = np.where(tnx > bond_df['rate_ma200'], 'Rising', 'Falling')
    bond_df['policy_direction'] = np.select([
        bond_df['rate_momentum_smooth'] >  BOND_MOMENTUM_THRESHOLD,
        bond_df['rate_momentum_smooth'] < -BOND_MOMENTUM_THRESHOLD,
    ], ['Tightening', 'Easing'], default='Neutral')

    tip      = close['TIP'].dropna()
    ief      = close['IEF'].dropna()
    tip_norm = (tip / tip.rolling(BOND_REAL_YIELD_NORM_WINDOW, min_periods=BOND_REAL_YIELD_NORM_WINDOW // 2).mean()).reindex(close.index)
    ief_norm = (ief / ief.rolling(BOND_REAL_YIELD_NORM_WINDOW, min_periods=BOND_REAL_YIELD_NORM_WINDOW // 2).mean()).reindex(close.index)

    bond_df['real_yield_proxy']        = tip_norm / ief_norm
    bond_df['real_yield_proxy_smooth'] = _smooth(bond_df['real_yield_proxy'], window=BOND_REAL_YIELD_SMOOTH)
    bond_df['real_yield_regime'] = np.where(
        bond_df['real_yield_proxy_smooth'] > BOND_REAL_YIELD_HIGH, 'Falling Real',
        np.where(bond_df['real_yield_proxy_smooth'] < BOND_REAL_YIELD_LOW, 'Rising Real', 'Neutral')
    )

    tlt = close['TLT'].dropna()
    shy = close['SHY'].dropna()

    bond_df['tlt_ma200'] = _smooth(tlt, window=BOND_MA200_WINDOW)
    tlt_smooth = _smooth(tlt, window=BOND_TLT_SMOOTH).reindex(close.index)
    bond_df['tlt_trend'] = np.where(tlt_smooth > bond_df['tlt_ma200'], 'Bull', 'Bear')

    shy_rel = (shy / _smooth(shy, window=BOND_SHY_REL_WINDOW)).reindex(close.index)
    tlt_rel = (tlt / _smooth(tlt, window=BOND_SHY_REL_WINDOW)).reindex(close.index)
    bond_df['duration_spread'] = shy_rel - tlt_rel

    bond_df['sig_curve_ok']       = (bond_df['curve_2s10s_smooth']     >  0                 ).astype(float)
    bond_df['sig_yields_falling'] = (bond_df['rate_momentum_smooth']    <  0                 ).astype(float)
    bond_df['sig_tlt_uptrend']    = (tlt_smooth                         >  bond_df['tlt_ma200']).astype(float)
    bond_df['sig_real_falling']   = (bond_df['real_yield_proxy_smooth'] >  BOND_REAL_YIELD_HIGH).astype(float)

    sig_cols = ['sig_curve_ok', 'sig_yields_falling', 'sig_tlt_uptrend', 'sig_real_falling']
    # fillna(0): sig_real_falling is NaN before ~2003 (TIP/IEF data); treat as abstaining
    bond_df['bond_bull_score']        = bond_df[sig_cols].fillna(0).sum(axis=1)
    bond_df['bond_bull_score_smooth'] = _smooth(bond_df['bond_bull_score'], window=BOND_SCORE_SMOOTH)

    # Feed smoothed continuous score — prevents single-signal integer flips from
    # instantly crossing the bull/bear thresholds every day
    raw_regime = _apply_bond_hysteresis(bond_df['bond_bull_score_smooth'])
    bond_df['bond_regime_raw'] = raw_regime

    regime_num    = bond_df['bond_regime_raw'].map(_BOND_REGIME_TO_NUM).astype(float)
    confirmed_num = (
        regime_num
        .rolling(BOND_CONFIRM_WINDOW)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= BOND_CONFIRM_MIN else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    bond_df['bond_regime'] = confirmed_num.map(_BOND_NUM_TO_REGIME).fillna('Unknown')

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = bond_df.dropna(subset=['curve_2s10s_smooth', 'rate_ma200'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Bond Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Bond Regime:":<28} {latest["bond_regime"]}')
    print(f'  {"Curve Regime:":<28} {latest["curve_regime"]}')
    print(f'  {"2s10s Spread (smooth):":<28} {latest["curve_2s10s_smooth"]:+.2f}%')
    print(f'  {"10yr Yield:":<28} {latest["curve_level"]:.2f}%')
    print(f'  {"Rate Trend (vs MA200):":<28} {latest["rate_trend"]}')
    print(f'  {"Policy Direction:":<28} {latest["policy_direction"]}')
    print(f'  {"Real Yield Regime:":<28} {latest["real_yield_regime"]}')
    print(f'  {"Duration Trend (TLT):":<28} {latest["tlt_trend"]}')
    print(f'  {"Bull Score (raw):":<28} {int(latest["bond_bull_score"])}/4')
    print(f'  {"Bull Score (smooth):":<28} {latest["bond_bull_score_smooth"]:.2f}/4')

    print('\n  --- Signal Breakdown ---')
    for col, label in [
        ('sig_curve_ok',       'Curve Not Inverted'),
        ('sig_yields_falling', 'Yields Falling (63d smooth)'),
        ('sig_tlt_uptrend',    'TLT Above MA200'),
        ('sig_real_falling',   'Real Yields Falling'),
    ]:
        print(f'    {"YES" if latest[col] == 1.0 else "NO ":<4} {label}')

    print('\n  --- Bond Regime Distribution (full history) ---')
    for regime in ['Bond Bull', 'Bond Neutral', 'Bond Bear']:
        pct = (bond_df['bond_regime'] == regime).mean()
        print(f'  {regime:<16} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return bond_df

_REGIME_TO_NUM = {'Commodity Bull': 2, 'Commodity Neutral': 1, 'Commodity Bear': 0, 'Unknown': -1}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _apply_hysteresis(score_series: pd.Series) -> list:
    regimes = []
    state   = None
    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue
        if state is None:
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
            else:                               state = 'Commodity Neutral'
        elif state == 'Commodity Bull':
            if   score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
            elif score <  COMMODITY_BULL_EXIT:  state = 'Commodity Neutral'
        elif state == 'Commodity Bear':
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score >  COMMODITY_BEAR_EXIT:  state = 'Commodity Neutral'
        else:  # Neutral
            if   score >= COMMODITY_BULL_ENTRY: state = 'Commodity Bull'
            elif score <= COMMODITY_BEAR_ENTRY: state = 'Commodity Bear'
        regimes.append(state)
    return regimes


def _confirm(raw: pd.Series) -> pd.Series:
    num = raw.map(_REGIME_TO_NUM).astype(float)
    confirmed = (
        num
        .rolling(COMMODITY_CONFIRM_WINDOW, min_periods=COMMODITY_CONFIRM_MIN)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= COMMODITY_CONFIRM_MIN else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    return confirmed.map(_NUM_TO_REGIME).fillna('Unknown')


# ── 1. Broad commodity trend (CRB proxy) ──────────────────────────────────────

def compute_commodity_trend(close: pd.DataFrame) -> pd.Series:
    """Average 63d return across a liquid futures basket — overall commodity momentum."""
    basket  = ['CL=F', 'HG=F', 'GC=F', 'ZC=F', 'ZW=F', 'NG=F']
    returns = {t: close[t].dropna().pct_change(COMMODITY_BASKET_WINDOW).reindex(close.index)
               for t in basket if t in close.columns}
    if not returns:
        return pd.Series(np.nan, index=close.index)
    return _smooth(pd.DataFrame(returns).mean(axis=1), window=COMMODITY_SMOOTH_WINDOW)


# ── 2. Copper / Gold ratio (growth vs fear) ───────────────────────────────────

def compute_copper_gold_ratio(close: pd.DataFrame):
    """Rising ratio = growth > fear = risk-on. Falling = recession / safe-haven demand."""
    if 'HG=F' not in close.columns or 'GC=F' not in close.columns:
        nan = pd.Series(np.nan, index=close.index)
        return nan, nan, nan, nan
    ratio        = (close['HG=F'].dropna() / close['GC=F'].dropna()).reindex(close.index)
    ratio_ma200  = _smooth(ratio, window=COMMODITY_MA_WINDOW)
    ratio_smooth = _smooth(ratio, window=COMMODITY_SMOOTH_WINDOW)
    sig          = (ratio_smooth > ratio_ma200).astype(float)
    return ratio, ratio_smooth, ratio_ma200, sig


# ── 3. Energy regime (inflation signal) ───────────────────────────────────────

def compute_energy_regime(close: pd.DataFrame) -> pd.Series:
    """WTI trend, momentum, Brent confirmation → score 0-3."""
    if 'CL=F' not in close.columns:
        return pd.Series(np.nan, index=close.index)
    # Floor at $1: April 2020 WTI went negative (-$37) due to storage constraints —
    # a physically impossible settlement that corrupts MA and momentum calculations
    wti       = close['CL=F'].dropna().clip(lower=1.0)
    wti_ma200 = _smooth(wti, window=COMMODITY_MA_WINDOW)
    wti_sm    = _smooth(wti, window=COMMODITY_SMOOTH_WINDOW)
    wti_mom   = _smooth(
        wti.pct_change(COMMODITY_MOMENTUM_WINDOW).reindex(close.index), window=10
    )
    sig_trend = (wti_sm    > wti_ma200).astype(float)
    sig_mom   = (wti_mom   > 0        ).astype(float)
    brent     = close.get('BZ=F')
    if brent is not None:
        brent_ma200 = _smooth(brent, window=COMMODITY_MA_WINDOW)
        sig_brent   = (_smooth(brent, window=COMMODITY_SMOOTH_WINDOW) > brent_ma200).astype(float)
    else:
        sig_brent = sig_trend  # fallback — WTI counts twice
    return (sig_trend + sig_mom + sig_brent).reindex(close.index)


# ── 4. Gold regime (real yield / safe-haven signal) ───────────────────────────

def compute_gold_regime(close: pd.DataFrame) -> pd.Series:
    """Gold trend, momentum, silver confirmation → score 0-3."""
    if 'GC=F' not in close.columns:
        return pd.Series(np.nan, index=close.index)
    gold       = close['GC=F'].dropna()
    gold_ma200 = _smooth(gold, window=COMMODITY_MA_WINDOW)
    gold_sm    = _smooth(gold, window=COMMODITY_SMOOTH_WINDOW)
    gold_mom   = _smooth(
        gold.pct_change(COMMODITY_MOMENTUM_WINDOW).reindex(close.index), window=10
    )
    sig_trend  = (gold_sm  > gold_ma200).astype(float)
    sig_mom    = (gold_mom > 0         ).astype(float)
    silver     = close.get('SI=F')
    if silver is not None:
        silver_ma200 = _smooth(silver, window=COMMODITY_MA_WINDOW)
        sig_silver   = (_smooth(silver, window=COMMODITY_SMOOTH_WINDOW) > silver_ma200).astype(float)
    else:
        sig_silver = sig_trend
    return (sig_trend + sig_mom + sig_silver).reindex(close.index)


# ── 5. Agriculture regime (food inflation / supply shock) ─────────────────────

def compute_agriculture_regime(close: pd.DataFrame) -> pd.Series:
    """Fraction of corn / wheat / soybeans above their MA200 — breadth-style."""
    ags  = ['ZC=F', 'ZW=F', 'ZS=F']
    sigs = []
    for ticker in ags:
        if ticker in close.columns:
            s     = close[ticker].dropna()
            ma200 = _smooth(s, window=COMMODITY_MA_WINDOW)
            sm    = _smooth(s, window=COMMODITY_SMOOTH_WINDOW)
            sigs.append((sm > ma200).astype(float).reindex(close.index))
    if not sigs:
        return pd.Series(np.nan, index=close.index)
    return pd.concat(sigs, axis=1).mean(axis=1)


# ── Master commodity regime ────────────────────────────────────────────────────

def compute_commodity_regime(df_commodities: pd.DataFrame) -> pd.DataFrame:
    """
    4-group master regime: Industrial (Cu/Au) + Energy + Precious + Agriculture.
    Each group casts one vote → score 0-4 → smooth → hysteresis → confirmation.

    Regime states: Commodity Bull / Commodity Neutral / Commodity Bear
    """
    close = df_commodities['Close'].unstack('Ticker')

    ratio, ratio_smooth, ratio_ma200, sig_copper_gold = compute_copper_gold_ratio(close)
    energy_score = compute_energy_regime(close)
    gold_score   = compute_gold_regime(close)
    ag_signal    = compute_agriculture_regime(close)
    broad_trend  = compute_commodity_trend(close)

    commodity_df = pd.DataFrame(index=close.index)

    # ── Store raw prices and sub-signals for the plot ─────────────────────────
    if 'CL=F' in close.columns:
        wti = close['CL=F'].dropna().clip(lower=1.0)
        commodity_df['oil']        = wti.reindex(close.index)
        commodity_df['oil_ma200']  = _smooth(wti, window=COMMODITY_MA_WINDOW)
        commodity_df['oil_smooth'] = _smooth(wti, window=COMMODITY_SMOOTH_WINDOW)
    if 'GC=F' in close.columns:
        gold = close['GC=F'].dropna()
        commodity_df['gold']       = gold.reindex(close.index)
        commodity_df['gold_ma200'] = _smooth(gold, window=COMMODITY_MA_WINDOW)

    commodity_df['copper_gold_ratio']        = ratio
    commodity_df['copper_gold_ratio_smooth'] = ratio_smooth
    commodity_df['copper_gold_ratio_ma200']  = ratio_ma200
    commodity_df['energy_score']             = energy_score
    commodity_df['gold_score']               = gold_score
    commodity_df['ag_signal']                = ag_signal
    commodity_df['broad_trend']              = broad_trend

    # ── 4-vote master score ───────────────────────────────────────────────────
    commodity_df['sig_industrial'] = sig_copper_gold.reindex(close.index)
    commodity_df['sig_energy']     = (energy_score >= 2).astype(float).where(energy_score.notna())
    commodity_df['sig_gold']       = (gold_score   >= 2).astype(float).where(gold_score.notna())
    commodity_df['sig_ag']         = (ag_signal    > 0.5).astype(float).where(ag_signal.notna())

    sig_cols = ['sig_industrial', 'sig_energy', 'sig_gold', 'sig_ag']
    commodity_df['bull_score']        = commodity_df[sig_cols].fillna(0).sum(axis=1)
    commodity_df['bull_score_smooth'] = _smooth(commodity_df['bull_score'], window=COMMODITY_SCORE_SMOOTH)

    raw_regime = pd.Series(
        _apply_hysteresis(commodity_df['bull_score_smooth']),
        index=commodity_df.index,
    )
    commodity_df['commodity_regime_raw'] = raw_regime
    commodity_df['commodity_regime']     = _confirm(raw_regime)

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = commodity_df.dropna(subset=['bull_score'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Commodity Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Commodity Regime:":<28} {latest["commodity_regime"]}')
    print(f'  {"Bull Score (raw):":<28} {int(latest["bull_score"])}/4')
    print(f'  {"Bull Score (smooth):":<28} {latest["bull_score_smooth"]:.2f}/4')
    print(f'  {"Broad Trend (63d avg ret):":<28} {latest["broad_trend"]:+.2%}')
    if not pd.isna(latest.get('copper_gold_ratio_smooth', np.nan)):
        dev = latest['copper_gold_ratio_smooth'] / latest['copper_gold_ratio_ma200'] - 1
        print(f'  {"Cu/Au Ratio vs MA200:":<28} {dev:+.2%}')

    print('\n  --- Signal Breakdown ---')
    for col, label in [
        ('sig_industrial', 'Cu/Au Ratio Above MA200'),
        ('sig_energy',     'Energy Score ≥ 2/3'),
        ('sig_gold',       'Gold Score ≥ 2/3'),
        ('sig_ag',         'Agriculture Breadth > 50%'),
    ]:
        val = latest.get(col, np.nan)
        print(f'    {"YES" if val == 1.0 else "NO ":<4} {label}')

    print('\n  --- Commodity Regime Distribution (full history) ---')
    for regime in ['Commodity Bull', 'Commodity Neutral', 'Commodity Bear']:
        pct = (commodity_df['commodity_regime'] == regime).mean()
        print(f'  {regime:<22} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return commodity_df


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
