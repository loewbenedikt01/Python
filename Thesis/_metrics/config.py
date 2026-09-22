
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
TRANSACTION_COST_BPS    = 10        # charged on turnover at each rebalance [0, 10, 20], [none, realistic, conservative]
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
# Configuration for XGB + RF
# ----
BASE_SEED               = [41, 42, 43, 44, 45]

# ----
# Configuration for XGB Weights Calculation
# ----
TRAINING_MONTHS_XGB     = 60        # fixed rolling training window (not expanding)
EMBARGO_MONTHS_XGB = {
    "Monthly":      1,              # gap between training end and validation start
    "Quarterly":    3,
    "Yearly":       12,
}                                   
VALIDATION_MONTHS_XGB   = 24        # fixed validation block ending at the rebalance date

XGB_FIXED = {
    'objective'        : 'reg:pseudohubererror',
    'n_estimators'     : 300,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
}

XGB_GRID = {
    'learning_rate'     : [0.005, 0.01, 0.03],
    'max_depth'         : [1, 2, 3],
    'min_child_weight'  : [1, 5, 10, 20],
    'reg_lambda'        : [1, 5, 10],
}

# ----
# Configuration for RF Weights Calculation
# ----
TRAINING_MONTHS_RF      = 60        # fixed rolling training window (not expanding)
EMBARGO_MONTHS_RF  = {
    "Monthly":      1,              # gap between training end and validation start
    "Quarterly":    3,
    "Yearly":       12,
}                                   
VALIDATION_MONTHS_RF    = 24        # fixed validation block ending at the rebalance date

RF_FIXED = {
    'n_estimators'     : 300,
    'max_samples'      : 0.7,
    'bootstrap'        : True,
    'criterion'        : "squared_error",
}

RF_GRID = {
    'max_features'      : ["sqrt", 0.33, 0.5],
    'max_depth'         : [3, 5, 10, None],
    'min_samples_leaf'  : [5, 10, 20, 50],
}

# ----
# Configuration for RF Weights Calculation
# ----
TRAINING_MONTHS_LSTM    = 60        # fixed rolling training window (not expanding)
EMBARGO_MONTHS_LSTM = {
    "Monthly":      1,              # gap between training end and validation start
    "Quarterly":    3,
    "Yearly":       12,
}                                   
VALIDATION_MONTHS_LSTM  = 24        # fixed validation block ending at the rebalance date

LSTM_FIXED = {
    'epochs'           : 30,
    'batch_size'       : 128,
    'lr'               : 1e-3,
    'l2'               : 1e-4,
    'seq_len'          : 6,
}

LSTM_GRID = {
    'units'             : [8, 16],
    'dropout'           : [0.1, 0.3],
}
