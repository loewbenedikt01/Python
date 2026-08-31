"""
Market-capitalisation weighted baseline.

Each calendar year Y starts from the end-of-(Y-1) market caps in universe.py.
Within the year the cap of each name drifts with its share price:

    mcap_i(t) = mcap0_i * price_i(t) / price_i(first trading day of year Y)

so a February rebalance uses the January price move, a March rebalance the
Jan-Feb move, and so on.  Weights are those drifted caps, renormalised to 1.
At each new year the caps reset to the fresh universe.py values.

The shared portfolio engine handles the actual rebalancing, drift between
rebalances, turnover and costs.  Output tree: _output/market_cap/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))

import export
from config import START_DATE, END_DATE
from portfolio import build_portfolio, load_prices
from universe import tickers as UNIVERSE

# ---- 
# Variables
# ----

MODEL_NAME = "market_cap_no_trans"          # change per run; costs live in config.py

FREQUENCIES = [
    "Monthly", 
    "Quarterly", 
    "Yearly"
]

# ----
# Main Part
# ----

_UNI_YEARS = sorted(UNIVERSE)

def _initial_caps(year: int) -> list[tuple[str, float]]:
    """(ticker, end-of-(year-1) market cap) for the names investable in `year`."""
    y = min(max(year - 1, _UNI_YEARS[0]), _UNI_YEARS[-1])
    return UNIVERSE[y]


def market_cap_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """
    A target-weight row for the first trading day of every month, weights =
    price-drifted market caps (see module docstring).  build_portfolio samples
    whichever rebalance frequency it is asked for.
    """
    cal = prices.loc[START_DATE:END_DATE].index
    month_firsts = cal[~cal.to_period("M").duplicated()]

    rows = {}
    for d in month_firsts:
        anchor = cal[cal.year == d.year][0]          # first trading day of the year
        caps = {}
        for ticker, mcap0 in _initial_caps(d.year):
            if ticker not in prices.columns:
                continue
            p_now, p_anchor = prices.at[d, ticker], prices.at[anchor, ticker]
            if pd.isna(p_now) or pd.isna(p_anchor) or p_anchor == 0:
                continue
            caps[ticker] = mcap0 * p_now / p_anchor
        s = pd.Series(caps, dtype=float)
        if s.sum() > 0:
            rows[d] = s / s.sum()
    return pd.DataFrame(rows).T


def main() -> None:
    prices = load_prices()
    targets = market_cap_targets(prices)

    for frequency in FREQUENCIES:
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"market_cap/{MODEL_NAME}_{frequency.lower()}"

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

        print(f"{name:34s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}")


if __name__ == "__main__":
    main()
