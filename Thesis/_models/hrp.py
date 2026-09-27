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

Regime conditioning
-------------------
HRP consumes nothing but a covariance matrix, so the regime enters entirely
through Sigma^.  The moment estimator below is byte-identical to the one in
mvo.py: fed the same COV_METHOD / MOMENT_MODE / lookback, both models receive
the same Sigma^, so any gap between them is the allocation rule alone.
Keep the two copies in sync if either is edited.

  MOMENT_MODE = "pooled" | "weighted" | "mixture"   (see the Moments section below)

  USE_REGIME        shrink toward the inverse-variance portfolio as crisis
                    probability rises:
                        w = (1-a) w_hrp + a w_ivp,   a = p_crisis * A
                    HRP has no expected-return input, so the analogue of MVO's
                    "blend toward minimum variance" is to lean on variances
                    alone and away from the clustering -- which is precisely the
                    structure that degrades when correlations converge in a
                    crisis.  Both portfolios are long-only and sum to 1, and
                    that set is convex, so the blend stays feasible; the box
                    projection is applied afterwards regardless.

With DETECTOR = "none" every path reduces to the pooled baseline.

-> Before Running: 
    Things to adjust
        TRANSACTION_COST_BPS    
            [0 | 10 | 20]
    Do not run each Mode with different Transaction costs. Calculate the Transaction costs impact across a small mode and then further document.
    
    Possible Run Modes:
        DETECTOR = "none" + MOMENT_MODE = "pooled" + USE_REGIME = False
        DETECTOR = "hmm" + MOMENT_MODE = "pooled" + USE_REGIME = True
        DETECTOR = "hmm" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "hmm" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "hmm" + MOMENT_MODE = "mixture" + USE_REGIME = False
        DETECTOR = "wasserstein" + MOMENT_MODE = "pooled" + USE_REGIME = True
        DETECTOR = "wasserstein" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "wasserstein" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "wasserstein" + MOMENT_MODE = "mixture" + USE_REGIME = False
        DETECTOR = "changepoint" + MOMENT_MODE = "pooled" + USE_REGIME = True
        DETECTOR = "changepoint" + MOMENT_MODE = "weighted" + USE_REGIME = False
        DETECTOR = "changepoint" + MOMENT_MODE = "mixture" + USE_REGIME = True
        DETECTOR = "changepoint" + MOMENT_MODE = "mixture" + USE_REGIME = False
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

regime_def.DETECTOR = "none"    # "none" | "changepoint" | "hmm" | "wasserstein"

MOMENT_MODE     = "pooled"      # "pooled" | "weighted" | "mixture"
USE_REGIME_RISK = False         # blend toward inverse-variance in crisis
RISK_MAX_SHRINK = 0.50          # a = p_crisis * RISK_MAX_SHRINK

DETECTOR     = regime_def.DETECTOR
regime_probs = regime_def.regime_probs
WEIGHT_FLOOR = regime_def.REGIME_WEIGHT_FLOOR      # same floor as MVO / the ML models

tag = {"pooled": "p", "weighted": "w", "mixture": "m"}[MOMENT_MODE]
tag += "_R" if USE_REGIME_RISK else ""
det = "" if DETECTOR == "none" else f"{DETECTOR}"

MODEL_NAME = f"mvo_t_{TRANSACTION_COST_BPS}_{COV_METHOD}_{LINKAGE}_{det}_{tag}"          # change per run; costs live in config.py

FREQUENCIES = [
    "Monthly",
    "Quarterly",
    "Yearly",
]


# ----
# Moments  (identical to mvo.py -- keep the two in sync so HRP and MVO can be
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


def _n_eff(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _similarity(p_hist: np.ndarray, p_now: float) -> np.ndarray:
    """Per-day resemblance to the regime expected at d, floored as in regime_def."""
    sim = p_hist * p_now + (1.0 - p_hist) * (1.0 - p_now)
    return WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * sim


def _regime_moments(win: pd.DataFrame, p_hist: np.ndarray, p_now: float):
    """
    Dispatch on MOMENT_MODE; returns (mu, Sigma, n_eff_report).
    A state with too little weight in the window falls back to pooled moments
    for that state, so a crisis-free window cannot produce a degenerate crisis
    covariance.  With p_hist == 0 every mode returns the pooled moments.
    """
    if MOMENT_MODE == "pooled" or not np.any(p_hist):
        mu, cov = _moments(win)
        return mu, cov, float(len(win))

    if MOMENT_MODE == "weighted":
        w = _similarity(p_hist, p_now)
        mu, cov = _moments(win, w)
        return mu, cov, _n_eff(w)

    if MOMENT_MODE != "mixture":
        raise ValueError(f"unknown MOMENT_MODE {MOMENT_MODE!r}")

    pis   = np.array([1.0 - p_now, p_now])
    masks = [1.0 - p_hist, p_hist]
    mu_pool, cov_pool = _moments(win)
    mus, covs, neffs = [], [], []
    for pk in masks:
        if pk.sum() <= 0 or _n_eff(pk) < MIN_OBS:
            mus.append(mu_pool); covs.append(cov_pool); neffs.append(np.nan)
            continue
        m, c = _moments(win, pk)
        mus.append(m); covs.append(c); neffs.append(_n_eff(pk))

    mu = sum(pi * m for pi, m in zip(pis, mus))
    cov = sum(pi * (c + np.outer(m - mu, m - mu))
              for pi, m, c in zip(pis, mus, covs))
    return mu, cov, float(np.nanmax(neffs))


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

        # daily regime probabilities over the estimation window, plus at d
        if DETECTOR == "none":
            p_hist, p_now = np.zeros(len(win)), 0.0
        else:
            p_hist = regime_probs(win.index)["p_crisis"].to_numpy()
            p_now  = float(regime_probs([d])["p_crisis"].iloc[0])

        _, cov_d, ne = _regime_moments(win, p_hist, p_now)
        cov = cov_d * h

        sd = np.sqrt(np.diag(cov))
        if not np.all(sd > 0):
            continue
        corr = cov / np.outer(sd, sd)

        w = _hrp_weights(cov, corr, LINKAGE)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            continue

        # regime-dependent risk: lean on variances, away from the clustering
        a = 0.0
        if USE_REGIME_RISK and DETECTOR != "none" and p_now > 0:
            a = float(p_now * RISK_MAX_SHRINK)
            w = (1.0 - a) * w + a * _ivp_weights(cov)

        w = _apply_box(w, *_weight_box(len(keep)))
        rows[d] = pd.Series(w, index=keep)
        diag[d] = {"p_crisis": p_now, "n_eff": ne, "shrink": a,
                   "vol_ann": float(np.sqrt(np.diag(cov).mean() * 252 / h)),
                   "avg_corr": float(corr[np.triu_indices_from(corr, 1)].mean())}

    return (pd.DataFrame(rows).T, len(reb_dates),
            pd.DataFrame(diag).T.rename_axis("date"))


# ----
# Main Part
# ----

def main() -> None:
    if DETECTOR == "none" and (MOMENT_MODE != "pooled" or USE_REGIME_RISK):
        print("[hrp] NOTE: regime channels set but DETECTOR='none' -> this run "
              "must reproduce the pooled baseline bit-for-bit (regression test).",
              flush=True)

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

        extra = ""
        if DETECTOR != "none" and len(diag):
            extra = (f"  mean p_crisis {diag['p_crisis'].mean():.2f}"
                     f"  min n_eff {diag['n_eff'].min():.0f}")
        print(f"{name:40s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"solved {len(targets)}/{n_dates}{extra}")


if __name__ == "__main__":
    main()