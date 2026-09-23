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

-> Before Running: 
    Things to adjust
        TRANSACTION_COST_BPS    
            [0 | 10 | 20]
    Do not run each Mode with different Transaction costs. Calculate the Transaction costs impact across a small mode and then further document.
    
    Possible Run Modes:
        DETECTOR = "none" + MOMENT_MODE = "pooled" + USE_REGIME = False
        DETECTOR = "hmm" + MOMENT_MODE = "pooled" + USE_REGIME = False
        DETECTOR = "hmm" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "hmm" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "hmm" + MOMENT_MODE = "mixture" + USE_REGIME = False
        DETECTOR = "wasserstein" + MOMENT_MODE = "pooled" + USE_REGIME = False
        DETECTOR = "wasserstein" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "wasserstein" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "wasserstein" + MOMENT_MODE = "mixture" + USE_REGIME = False
        DETECTOR = "changepoint" + MOMENT_MODE = "pooled" + USE_REGIME = False
        DETECTOR = "changepoint" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "changepoint" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "changepoint" + MOMENT_MODE = "mixture" + USE_REGIME = False
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

regime_def.DETECTOR = "changepoint"        # "none" | "hmm" | "wasserstein" | "changepoint"

MOMENT_MODE     = "mixture"      # "pooled" | "weighted" | "mixture"
USE_REGIME      = False         # blend toward min-variance in crisis
RISK_MAX_SHRINK = 0.50          # a = p_crisis * RISK_MAX_SHRINK
 
DETECTOR     = regime_def.DETECTOR
regime_probs = regime_def.regime_probs
WEIGHT_FLOOR = regime_def.REGIME_WEIGHT_FLOOR      # same floor as the ML models

tag = {"pooled": "p", "weighted": "w", "mixture": "m"}[MOMENT_MODE]
tag += "_R" if USE_REGIME else ""
det = "" if DETECTOR == "none" else f"{DETECTOR}"

MODEL_NAME = f"mvo_t_{TRANSACTION_COST_BPS}_{COV_METHOD}_{det}_{tag}"          # change per run; costs live in config.py


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
 
 
def _n_eff(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())
 
 
def _similarity(p_hist: np.ndarray, p_now: float) -> np.ndarray:
    """Per-day resemblance to the regime expected at d, floored as in regime_def."""
    sim = p_hist * p_now + (1.0 - p_hist) * (1.0 - p_now)
    return WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * sim
 
 
def _regime_moments(win: pd.DataFrame, p_hist: np.ndarray, p_now: float):
    """
    Dispatch on MOMENT_MODE.  Returns (mu, Sigma, n_eff_report).
    With p_hist == p_now == 0 every mode returns the pooled moments.
    """
    if MOMENT_MODE == "pooled" or DETECTOR == "none":
        mu, cov = _moments(win)
        return mu, cov, float(len(win))
 
    if MOMENT_MODE == "weighted":
        w = _similarity(p_hist, p_now)
        mu, cov = _moments(win, w)
        return mu, cov, _n_eff(w)
 
    # ---- mixture: soft state assignment, then combine with p_next
    pis   = np.array([1.0 - p_now, p_now])
    masks = [1.0 - p_hist, p_hist]
    mus, covs, neffs = [], [], []
    mu_pool, cov_pool = _moments(win)
    for pk in masks:
        if pk.sum() <= 0 or _n_eff(pk) < MIN_OBS:
            # too little of that state in the window to estimate it
            mus.append(mu_pool); covs.append(cov_pool); neffs.append(np.nan)
            continue
        m, c = _moments(win, pk)
        mus.append(m); covs.append(c); neffs.append(_n_eff(pk))
 
    mu = sum(pi * m for pi, m in zip(pis, mus))
    cov = sum(pi * (c + np.outer(m - mu, m - mu))
              for pi, m, c in zip(pis, mus, covs))
    return mu, cov, float(np.nanmax(neffs))

 
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
 
        # daily regime probabilities over the estimation window, plus at d
        if DETECTOR == "none":
            p_hist, p_now = np.zeros(len(win)), 0.0
        else:
            p_hist = regime_probs(win.index)["p_crisis"].to_numpy()
            p_now  = float(regime_probs([d])["p_crisis"].iloc[0])
 
        mu_d, cov_d, n_eff = _regime_moments(win, p_hist, p_now)
        mu  = (mu_d - RISK_FREE_RATE) * h
        cov = cov_d * h
 
        w = _max_sharpe(mu, cov)
        method[d] = "max_sharpe" if w is not None else "min_variance"
        w_mv = None
        if w is None:
            w = w_mv = _min_variance(cov)
        if w is None:
            method.pop(d)
            continue
 
        # regime-dependent risk: shrink toward minimum variance in crisis
        a = 0.0
        if USE_REGIME and DETECTOR != "none" and p_now > 0:
            if w_mv is None:
                w_mv = _min_variance(cov)
            if w_mv is not None:
                a = float(p_now * RISK_MAX_SHRINK)
                w = (1.0 - a) * w + a * w_mv
 
        rows[d] = pd.Series(w, index=keep)
        diag[d] = {"p_crisis": p_now, "n_eff": n_eff, "shrink": a,
                   "vol_ann": float(np.sqrt(np.diag(cov).mean() * 252 / h))}
 
    targets = pd.DataFrame(rows).T
    diag_df = pd.DataFrame(diag).T.rename_axis("date")
    return targets, pd.Series(method, name="method").sort_index(), diag_df
 
 
# ----
# Main Part
# ----
 
def main() -> None:
    if DETECTOR == "none" and (MOMENT_MODE != "pooled" or USE_REGIME):
        print("[mvo] NOTE: regime channels set but DETECTOR='none' -> this run "
              "must reproduce the pooled baseline bit-for-bit (regression test).",
              flush=True)
 
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
 
        n_fb = int((method == "min_variance").sum())
        extra = ""
        if DETECTOR != "none" and len(diag):
            extra = (f"  mean p_crisis {diag['p_crisis'].mean():.2f}"
                     f"  min n_eff {diag['n_eff'].min():.0f}/{len(diag)}")
        print(f"{name:40s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"min-var fallback {n_fb}/{len(method)}{extra}")
 
 
if __name__ == "__main__":
    main()