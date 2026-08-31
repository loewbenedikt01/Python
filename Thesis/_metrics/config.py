
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

# ----
# Configuration for MVO Weights Calculation
# ----
LOOKBACK_MONTHS_MVO     = 60        # 60 months lookback window for calculation
MIN_DATA                = 0.50      # min 50% data given to calculate, otherwise drop the ticker
