import os

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
EQUITY_BULL_THRESHOLD  = 0.57
EQUITY_BEAR_THRESHOLD  = 0.53
EQUITY_MIN_STOCKS      = 5     # min tickers with valid MA200 to trust a continent's breadth
EQUITY_MA_WINDOW       = 200
EQUITY_SMOOTH_WINDOW   = 10
EQUITY_TREND_WINDOW    = 63
EQUITY_CONFIRM_WINDOW  = 5
EQUITY_CONFIRM_MIN     = 4

# ── Bond Regime ───────────────────────────────────────────────────────────────
BOND_BULL_ENTRY             = 3
BOND_BEAR_ENTRY             = 1
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
BOND_CONFIRM_WINDOW         = 5
BOND_CONFIRM_MIN            = 4

# ── Forex Regime ──────────────────────────────────────────────────────────────
FOREX_TREND_WINDOW     = 200
FOREX_SMOOTH_WINDOW    = 10
FOREX_CONFIRM_WINDOW   = 5

# ── Commodity Regime ──────────────────────────────────────────────────────────
COMMODITY_TREND_WINDOW   = 200
COMMODITY_SMOOTH_WINDOW  = 10
COMMODITY_CONFIRM_WINDOW = 5

# ── Crypto Regime ─────────────────────────────────────────────────────────────
CRYPTO_TREND_WINDOW    = 200
CRYPTO_SMOOTH_WINDOW   = 10
CRYPTO_CONFIRM_WINDOW  = 5

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_COMPONENTS    = 4
HMM_N_ITER          = 200
HMM_COVARIANCE_TYPE = 'full'

# ── Colours ───────────────────────────────────────────────────────────────────
REGIME_COLORS = {
    # Equity
    'Expanding Bull':     '#2ecc71',
    'Deteriorating Bull': '#f39c12',
    'Recovering Bear':    '#3498db',
    'Confirmed Bear':     '#e74c3c',
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
