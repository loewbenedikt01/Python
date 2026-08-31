
"""
Definitions of Metrics

Metrics are used for Calculation and Reporting

6 Main Metrics:
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Max. Drawdown, Cumulative Return
"""


from pathlib import Path

# ----
# Paths
# ----
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATABASE_PATH   = PROJECT_ROOT / "_database" / "database.parquet"
OUTPUT_ROOT     = PROJECT_ROOT / "_output"


# ----
# Investment Dates
# ----
START_DATE = '1998-01-01'
END_DATE = '2025-12-31'

# ----
# Configuration for Metrics
# ----
TRADING_DAYS_PER_YEAR   = 252
MONTHS_PER_YEAR         = 12
RISK_FREE_RATE          = 0.0

# ----
# Configuration for Portfolio construction
# ----
MAX_HOLDINGS            = 20        # investable universe size (universe.py per year)
TRANSACTION_COST_BPS    = 00        # charged on turnover at each rebalance
MIN_WEIGHT              = 0.01      # min 1% weight allocation per stock  
MAX_WEIGHT              = 0.10      # max 10% weight allocation per stock
HORIZON_TRADING_DAYS = {
    "Monthly":   21,
    "Quarterly": 63,
    "Yearly":    252,
}
MIN_OBS = 63                        # Skip a rebalance date when the lookback window has fewer than this many usable daily observations.
MIN_DATA                = 0.50      # min 50% data given to calculate, otherwise drop the ticker

# ----
# Configuration for MVO Weights Calculation
# ----
LOOKBACK_MONTHS_MVO     = 60        # 60 months lookback window for calculation

# ----
# Configuration for HRP Weights Calculation
# ----
LOOKBACK_MONTHS_HRP     = 60        # 60 months lookback window for calculation

# ----
# Configuration for XGB Weights Calculation
# ----
LOOKBACK_MONTHS_HRP     = 60        # 60 months lookback window for calculation
TRAIN_MONTHS_XGB        = 60
VAL_MONTHS_XGB          = 24
EARLY_STOPPING          = 10
BASE_SEED               = [41, 42, 43, 44, 45]
HYPERPARAMETER_GRID     = {
    'learning_rate'     : [0.005, 0.01, 0.05],
    'max_depth'         : [1, 2, 5, 10],
    'min_child_weight'  : [1, 3, 5, 10],
    'n_estimators'      : [100, 200, 300],
    'gamma'             : [0.001, 0.005, 0.01],
    'subsample'         : [0.7, 0.85, 1.0],
    'colsample_bytree'  : [0.7, 0.85, 1.0],
    
}

