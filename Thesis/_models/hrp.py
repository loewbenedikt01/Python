"""
Hierarchical Risk Parity (López de Prado, 2016).

At each rebalance the forecast horizon equals the rebalance interval.  The
covariance / correlation that drives the clustering is estimated from daily log
returns over a trailing LOOKBACK_MONTHS_HRP window and scaled to the horizon
(h trading days); HRP weights are invariant to that scaling.

Algorithm, over that year's investable universe:
  1. correlation-distance matrix  d_ij = sqrt((1 - rho_ij) / 2)
  2. hierarchical clustering (LINKAGE)
  3. quasi-diagonalisation
  4. recursive bisection by inverse cluster variance

Regime run
----------
HRP in calm; when p_crisis > regime_def.CRISIS_THRESHOLD the portfolio
switches to inverse-variance weights.  HRP has no expected-return input, so
the defensive analogue of MVO's switch to minimum variance is to drop the
clustering and allocate on variances alone: the correlation structure HRP
relies on is what degrades when correlations converge in a crisis.

-> Before Running: 
    Things to adjust
        TRANSACTION_COST_BPS    
            [0 | 10 | 20]
    Do not run each Mode with different Transaction costs. Calculate the Transaction costs impact across a small mode and then further document.
    
    Two run options (set in the Regime block below):
        DETECTOR = "none"                 baseline: HRP at every rebalance
        DETECTOR = <detector> + SERIES    regime run: HRP in calm, inverse-variance in crisis
    SERIES per detector: changepoint "vix" | "gspc", hmm "gspc" | "vix",
    wasserstein "gspc_vix" | "gspc".  One detector/series per run.
"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import export
from _regimes import regime_def
from config import (
    START_DATE,
    END_DATE,
    LOOKBACK_MONTHS_HRP,
    MIN_DATA,
    MIN_WEIGHT,
    MAX_WEIGHT,
    HORIZON_TRADING_DAYS,
    MIN_OBS,
    TRANSACTION_COST_BPS,
)
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS

# ----
# Variables
# ----

COV_METHOD = "lw"      # "sample" | "ledoit_wolf"   (match MVO)
LINKAGE    = "ward"             # "single" | "ward" | "average"

# ----
# Regime implementation
# ----

regime_def.DETECTOR = "none"     # "none" (baseline) | "changepoint" | "hmm" | "wasserstein"
regime_def.SERIES   = None       # changepoint "vix"|"gspc", hmm "gspc"|"vix", wasserstein "gspc_vix"|"gspc"

MODEL_NAME = f"hrp_t_{TRANSACTION_COST_BPS}_{COV_METHOD}_{LINKAGE}{regime_def.run_tag()}"   # costs live in config.py

FREQUENCIES = [
    "Monthly",
    "Quarterly",
    "Yearly",
]


# ----
# Moments  (identical to mvo.py -- keep the two in sync so HRP and MVO are
# fed the same Sigma^; then any gap between them is the allocation rule alone)
# ----

def _moments(win: pd.DataFrame, w: np.ndarray | None = None):
    """
    (mu, Sigma) from daily log returns, optionally with per-day weights w.
    Rows are centred and scaled by sqrt(w) before Ledoit-Wolf, so the shrinkage
    intensity is estimated on the weighted sample -- it responds to the
    effective rather than the nominal sample size.
    """
    X = win.to_numpy()
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    w = w / w.mean()

    mu = np.average(X, axis=0, weights=w)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    if COV_METHOD == "lw":
        cov = LedoitWolf(assume_centered=True).fit(Xc).covariance_
    else:
        cov = Xc.T @ Xc / (w.sum() - 1.0)
    return mu, cov


# ----
# HRP
# ----

def _quasi_diag(link: np.ndarray) -> list[int]:
    """Dendrogram leaf order (López de Prado, snippet 16.2)."""
    link = link.astype(int)
    n = link[-1, 3]
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= n]
        i = clusters.index
        j = clusters.values - n
        sort_ix[i] = link[j, 0]
        sort_ix = pd.concat([sort_ix, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)

def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            c[k:m]
            for c in clusters
            for k, m in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w.sort_index().to_numpy()

def _hrp_weights(cov: np.ndarray, corr: np.ndarray, method: str) -> np.ndarray:
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    link = linkage(squareform(dist, checks=False), method=method)
    return _recursive_bisection(cov, _quasi_diag(link))

def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


# ----
# Weight box
# ----

def _weight_box(n: int) -> tuple[float, float]:
    lo, hi = MIN_WEIGHT, MAX_WEIGHT
    if n * hi < 1.0:
        hi = 1.0 / n
    if n * lo > 1.0:
        lo = 1.0 / n
    return lo, hi

def _apply_box(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Project long-only weights onto {lo <= w_i <= hi, sum w = 1} by water-filling."""
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    for _ in range(100):
        lo_hit, hi_hit = w < lo - 1e-12, w > hi + 1e-12
        if not lo_hit.any() and not hi_hit.any():
            break
        w[lo_hit], w[hi_hit] = lo, hi
        free = ~(lo_hit | hi_hit)
        slack = 1.0 - w[~free].sum()
        if free.any() and w[free].sum() > 0 and slack > 0:
            w[free] *= slack / w[free].sum()
        else:
            break
    return w / w.sum()


# ----
# Target weights
# ----

def hrp_targets(prices: pd.DataFrame, frequency: str):
    ret = np.log(prices / prices.shift(1))
    cal = prices.loc[START_DATE:END_DATE].index
    month_firsts = cal[~cal.to_period("M").duplicated()]
    reb_dates = month_firsts[month_firsts.month.isin(REBALANCE_MONTHS[frequency])]

    h = HORIZON_TRADING_DAYS[frequency]
    lookback = pd.DateOffset(months=LOOKBACK_MONTHS_HRP)

    rows: dict[pd.Timestamp, pd.Series] = {}
    diag: dict[pd.Timestamp, dict] = {}

    for d in reb_dates:
        uni = [t for t in universe_for(d.year) if t in ret.columns]
        win = ret.loc[d - lookback:d, uni]
        if len(win) < MIN_OBS:
            continue
        keep = win.columns[win.notna().sum() >= MIN_DATA * len(win)]
        win = win[keep].dropna(how="any")
        if len(keep) < 2 or len(win) < MIN_OBS:
            continue

        _, cov_d = _moments(win)
        cov = cov_d * h

        sd = np.sqrt(np.diag(cov))
        if not np.all(sd > 0):
            continue
        corr = cov / np.outer(sd, sd)

        # regime run: inverse-variance when the holding period is classed crisis
        p_now  = regime_def.p_crisis(d)
        crisis = regime_def.in_crisis(d)
        w = _ivp_weights(cov) if crisis else _hrp_weights(cov, corr, LINKAGE)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            continue

        w = _apply_box(w, *_weight_box(len(keep)))
        rows[d] = pd.Series(w, index=keep)
        diag[d] = {"p_crisis": p_now, "crisis": crisis,
                   "vol_ann": float(np.sqrt(np.diag(cov).mean() * 252 / h)),
                   "avg_corr": float(corr[np.triu_indices_from(corr, 1)].mean())}

    return (pd.DataFrame(rows).T, len(reb_dates),
            pd.DataFrame(diag).T.rename_axis("date"))


# ----
# Main Part
# ----

def main() -> None:
    prices = load_prices()
    for frequency in FREQUENCIES:
        targets, n_dates, diag = hrp_targets(prices, frequency)
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"hrp/{MODEL_NAME}_{frequency.lower()}"

        export.build_report(
            name,
            res.log_returns,
            weights=res.weights,
            rebalance_status=res.rebalance_status,
            diagnostics={
                "turnover": res.turnover,
                "transaction_costs": res.transaction_costs,
                "regime": diag,
            },
        )

        n_cr = int(diag["crisis"].sum()) if len(diag) else 0
        print(f"{name:40s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"solved {len(targets)}/{n_dates}  inverse-variance (crisis) {n_cr}")


if __name__ == "__main__":
    main()