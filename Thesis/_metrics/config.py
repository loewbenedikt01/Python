
"""
Definitions of Metrics

Metrics are used for Calculation and Reporting

6 Main Metrics:
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Max. Drawdown, Cumulative Return
"""


from pathlib import Path


TRADING_DAYS_PER_YEAR   = 252
MONTHS_PER_YEAR         = 12
RISK_FREE_RATE          = 0.0


# ----
# Paths
# ----
# config.py lives in Thesis/_metrics/ -> parents[1] is the Thesis/ root.
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATABASE_PATH   = PROJECT_ROOT / "_database" / "database.parquet"
OUTPUT_ROOT     = PROJECT_ROOT / "_output"

BENCHMARK_TICKER = "^GSPC"
