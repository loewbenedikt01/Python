"""
Equal-weight baseline.

Assigns 1/N to the N names in each year's investable universe
(universe.tickers[year-1]).  The shared portfolio engine handles monthly
rebalancing, weight drift, turnover and transaction costs; `export` writes the
standard output tree under _output/equal_weight/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))

import export
from config import START_DATE, END_DATE
from portfolio import build_portfolio, universe_for

"""
For each run, change:

'MODEL_NAME' 
'TRANSACTION_COSTS' in 'config.py
"""

MODEL_NAME       = 'equal_weight_no_trans'           # change to no trans or whatever u run
START_YEAR = pd.Timestamp(START_DATE).year
END_YEAR   = pd.Timestamp(END_DATE).year

FREQUENCIES = [
    'Monthly',
    'Quarterly',
    'Yearly',
]


def equal_weight_targets(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    One target row per year (dated Jan 1) holding that year's universe at 1/N.
    build_portfolio upsamples this to monthly rebalances.
    """
    rows = {}
    for year in range(start_year, end_year + 1):
        uni = universe_for(year)
        rows[pd.Timestamp(year, 1, 1)] = pd.Series(1.0 / len(uni), index=uni)
    return pd.DataFrame(rows).T


def main() -> None:
    targets = equal_weight_targets()
    for frequency in FREQUENCIES:
        res = build_portfolio(targets, frequency=frequency)
        name = f"equal_weight/{MODEL_NAME}_{frequency.lower()}"        # _output/equal_weight/equal_weight_monthly/...

        export.build_report(
            name,
            res.log_returns,
            weights=res.weights,
            rebalance_status=res.rebalance_status,
            diagnostics={
                "turnover": res.turnover,
                "transaction_costs": res.transaction_costs,
            },
        )

        print(f"{name:24s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):7.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}")


if __name__ == "__main__":
    main()
