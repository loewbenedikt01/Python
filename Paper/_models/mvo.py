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
"sample" (plain pandas covariance) or "ledoit_wolf" (shrinkage) [used as main cov].

-> Before Running: 
    Things to adjust
        TRANSACTION_COST_BPS    
            [0 | 10 | 20]
    Do not run each Mode with different Transaction costs. Calculate the Transaction costs impact across a small mode and then further document.
    
    Two run options (set in the Regime block below):
        DETECTOR = "none"                 baseline: max-Sharpe at every rebalance
        DETECTOR = <detector> + SERIES    regime run: max-Sharpe in calm, minimum-variance
                                          when p_crisis > regime_def.CRISIS_THRESHOLD
    plus one control run:
        DETECTOR = "none" + ALWAYS_MIN_VAR = True    minimum variance at every rebalance
    SERIES per detector: changepoint "vix" | "gspc", hmm "gspc" | "vix",
    wasserstein "gspc_vix" | "gspc".  One detector/series per run.
"""

import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
 
sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))
sys.path.append(str(Path(__file__).resolve().parents[1]))
 
import export
from _regimes import regime_def
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
    TRANSACTION_COST_BPS,
)
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS

# ----
# Variables
# ----

COV_METHOD = "lw"           # "sample" | "lw" = "ledoit_wolf"

FREQUENCIES = [
    "Monthly",
    "Quarterly",
    "Yearly",
]

# ----
# Regime Implementation
# ----

regime_def.DETECTOR = "none"     # "none" (baseline) | "changepoint" | "hmm" | "wasserstein"
regime_def.SERIES   = None       # changepoint "vix"|"gspc", hmm "gspc"|"vix", wasserstein "gspc_vix"|"gspc"

# Control run, not a regime run: minimum variance at every rebalance
ALWAYS_MIN_VAR = True

if ALWAYS_MIN_VAR and regime_def.DETECTOR != "none":
    raise ValueError("[mvo] ALWAYS_MIN_VAR is a control run: set regime_def.DETECTOR = 'none'")
_run = "_minvar" if ALWAYS_MIN_VAR else regime_def.run_tag()

MODEL_NAME = f"mvo_t_{TRANSACTION_COST_BPS}_{COV_METHOD}{_run}"   # costs live in config.py


# ----
# Moments
# ----
 
def _moments(win: pd.DataFrame, w: np.ndarray | None = None):
    """
    (mu, Sigma) from daily log returns, optionally with per-day weights w
    (normalised to mean 1).  Rows are scaled by sqrt(w) before Ledoit-Wolf so
    the shrinkage target is estimated on the same weighted sample.
    """
    X = win.to_numpy()
    n = len(X)
    if w is None:
        w = np.ones(n)
    w = w / w.mean()
 
    mu = np.average(X, axis=0, weights=w)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    if COV_METHOD == "lw":
        cov = LedoitWolf(assume_centered=True).fit(Xc).covariance_
    else:
        cov = Xc.T @ Xc / (w.sum() - 1.0)
    return mu, cov
 
 
# ----
# Optimisers
# ----
 
def _weight_box(n: int) -> tuple[float, float]:
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
        objective, x0, method="SLSQP",
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
    if not np.any(mu > 0):
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
 
def mvo_targets(prices: pd.DataFrame, frequency: str):
    ret = np.log(prices / prices.shift(1))
    cal = prices.loc[START_DATE:END_DATE].index
    month_firsts = cal[~cal.to_period("M").duplicated()]
    reb_dates = month_firsts[month_firsts.month.isin(REBALANCE_MONTHS[frequency])]
 
    h = HORIZON_TRADING_DAYS[frequency]
    lookback = pd.DateOffset(months=LOOKBACK_MONTHS_MVO)
 
    rows:   dict[pd.Timestamp, pd.Series] = {}
    method: dict[pd.Timestamp, str] = {}
    diag:   dict[pd.Timestamp, dict] = {}
 
    for d in reb_dates:
        uni = [t for t in universe_for(d.year) if t in ret.columns]
        win = ret.loc[d - lookback:d, uni]
        if len(win) < MIN_OBS:
            continue
        keep = win.columns[win.notna().sum() >= MIN_DATA * len(win)]
        win = win[keep].dropna(how="any")
        if len(keep) < 2 or len(win) < MIN_OBS:
            continue
 
        mu_d, cov_d = _moments(win)
        mu  = (mu_d - RISK_FREE_RATE) * h
        cov = cov_d * h
 
        # regime run: minimum variance when the holding period is classed crisis;
        # control run: minimum variance always
        p_now  = regime_def.p_crisis(d)
        crisis = regime_def.in_crisis(d)
        w = None if (crisis or ALWAYS_MIN_VAR) else _max_sharpe(mu, cov)
        method[d] = "max_sharpe" if w is not None else "min_variance"
        if w is None:
            w = _min_variance(cov)
        if w is None:
            method.pop(d)
            continue
 
        rows[d] = pd.Series(w, index=keep)
        diag[d] = {"p_crisis": p_now, "crisis": crisis,
                   "vol_ann": float(np.sqrt(np.diag(cov).mean() * 252 / h))}
 
    targets = pd.DataFrame(rows).T
    diag_df = pd.DataFrame(diag).T.rename_axis("date")
    return targets, pd.Series(method, name="method").sort_index(), diag_df
 
 
# ----
# Main Part
# ----
 
def main() -> None:
    prices = load_prices()
    for frequency in FREQUENCIES:
        targets, method, diag = mvo_targets(prices, frequency)
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
                "method": method,
                "regime": diag,
            },
        )

        n_mv = int((method == "min_variance").sum())
        n_cr = int(diag["crisis"].sum()) if len(diag) else 0
        print(f"{name:40s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"min-var {n_mv}/{len(method)} (crisis {n_cr})")
 
 
if __name__ == "__main__":
    main()