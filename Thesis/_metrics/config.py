
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
MAX_HOLDINGS            = 20         # investable universe size (universe.py per year)
TRANSACTION_COST_BPS    = 00         # charged on turnover at each rebalance
REBALANCE_FREQ          = "Monthly"  # default; models also run "Quarterly" / "Yearly"
