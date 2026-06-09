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
COMMODITY_TREND_WINDOW   = 200
COMMODITY_SMOOTH_WINDOW  = 10
COMMODITY_CONFIRM_WINDOW = 5

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
