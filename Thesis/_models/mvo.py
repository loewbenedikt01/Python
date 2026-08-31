"""
Mean-Variance Optimization — maximum-Sharpe (tangency) portfolio.

At each rebalance the forecast horizon equals the rebalance interval: Monthly
rebalances "predict" one month ahead, Quarterly one quarter, Yearly one full
year.  mu and Sigma are estimated from daily log returns over a trailing
LOOKBACK_MONTHS_MVO window and scaled to that horizon (h trading days).  Under a
fixed weight box the tangency and minimum-variance portfolios are invariant to
that scaling, so the horizon enters through the rebalance calendar and the
lookback data rather than through the numbers themselves.

Optimisation is long-only, fully invested, with a hard weight box
[MIN_WEIGHT, MAX_WEIGHT] on every investable name.  The objective is the
max-Sharpe portfolio; when it is ill-defined (no name has a positive expected
excess return, or the solver fails) the model falls back to the
minimum-variance portfolio under the same box.

The covariance estimator is switchable in the Variables block below:
"sample" (plain pandas covariance) or "ledoit_wolf" (shrinkage).

The shared portfolio engine handles the actual rebalancing, drift between
rebalances, turnover and costs.  Output tree: _output/mvo/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))

import export
from config import (
    START_DATE,
    END_DATE,
    LOOKBACK_MONTHS_MVO,
    MIN_DATA,
    MIN_WEIGHT,
    MAX_WEIGHT,
    HORIZON_TRADING_DAYS,
    RISK_FREE_RATE,
    MIN_OBS,
)
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS

# ----
# Variables
# ----

MODEL_NAME = "mvo_ledoit_wolf_no_trans"          # change per run; costs live in config.py
COV_METHOD = "ledoit_wolf"           # "sample" | "ledoit_wolf"

FREQUENCIES = [
    "Monthly",
    "Quarterly",
    "Yearly",
]

# ----
# Optimisers
# ----

def _weight_box(n: int) -> tuple[float, float]:
    """
    [MIN_WEIGHT, MAX_WEIGHT] per name, widened only if that box cannot sum to 1
    for n names (n < 10 lifts the ceiling, n > 100 drops the floor).
    """
    lo, hi = MIN_WEIGHT, MAX_WEIGHT
    if n * hi < 1.0:
        hi = 1.0 / n
    if n * lo > 1.0:
        lo = 1.0 / n
    return lo, hi

def _solve(objective, n: int) -> np.ndarray | None:
    lo, hi = _weight_box(n)
    x0 = np.clip(np.full(n, 1.0 / n), lo, hi)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(lo, hi)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        return None
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    return w / s if s > 0 else None

def _max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    """Tangency portfolio, or None when it is ill-defined."""
    if not np.any(mu > 0):                       # no positive expected excess return
        return None

    def neg_sharpe(w):
        var = float(w @ cov @ w)
        return -(w @ mu) / np.sqrt(max(var, 1e-16))

    return _solve(neg_sharpe, len(mu))

def _min_variance(cov: np.ndarray) -> np.ndarray | None:
    return _solve(lambda w: float(w @ cov @ w), cov.shape[0])


# ----
# Target weights
# ----

def mvo_targets(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    One target-weight row per rebalance date for `frequency`: the max-Sharpe
    portfolio (fallback minimum variance) over that year's investable universe,
    from the trailing LOOKBACK_MONTHS_MVO window of daily log returns scaled to
    the horizon.  Returns (targets, method) where `method` logs "max_sharpe" or
    "min_variance" per date.
    """
    ret = np.log(prices / prices.shift(1))
    cal = prices.loc[START_DATE:END_DATE].index
    month_firsts = cal[~cal.to_period("M").duplicated()]
    reb_dates = month_firsts[month_firsts.month.isin(REBALANCE_MONTHS[frequency])]

    h = HORIZON_TRADING_DAYS[frequency]
    lookback = pd.DateOffset(months=LOOKBACK_MONTHS_MVO)

    rows: dict[pd.Timestamp, pd.Series] = {}
    method: dict[pd.Timestamp, str] = {}
    for d in reb_dates:
        uni = [t for t in universe_for(d.year) if t in ret.columns]
        win = ret.loc[d - lookback:d, uni]
        if len(win) < MIN_OBS:
            continue

        keep = win.columns[win.notna().sum() >= MIN_DATA * len(win)]
        win = win[keep].dropna(how="any")
        if len(keep) < 2 or len(win) < MIN_OBS:
            continue

        mu = (win.mean().to_numpy() - RISK_FREE_RATE) * h
        if COV_METHOD == "ledoit_wolf":
            cov = LedoitWolf().fit(win.to_numpy()).covariance_ * h
        else:
            cov = win.cov().to_numpy() * h

        w = _max_sharpe(mu, cov)
        method[d] = "max_sharpe" if w is not None else "min_variance"
        if w is None:
            w = _min_variance(cov)
        if w is None:
            method.pop(d)
            continue

        rows[d] = pd.Series(w, index=keep)

    targets = pd.DataFrame(rows).T
    return targets, pd.Series(method, name="method").sort_index()


# ----
# Main Part
# ----

def main() -> None:
    prices = load_prices()

    for frequency in FREQUENCIES:
        targets, method = mvo_targets(prices, frequency)
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"mvo/{MODEL_NAME}_{frequency.lower()}"

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

        n_fb = int((method == "min_variance").sum())
        print(f"{name:34s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"min-var fallback {n_fb}/{len(method)}")


if __name__ == "__main__":
    main()
