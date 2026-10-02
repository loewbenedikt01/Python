"""
Central regime machinery, shared by every model (xgb / rf / lstm / ...).

A model file sets its knobs on this module right after import
(`regime_def.DETECTOR = "changepoint"`, `regime_def.USE_REGIME_WEIGHTS = True`)
and then calls the functions below, so the mechanism lives here once.

p_crisis semantics
------------------
regime_probs(d) returns the probability that the holding period starting at d
is in the high-volatility state, computed from data through d-1 only.
  * changepoint: current-regime estimate (EWMA vol of the active segment),
    used as the forecast for the holding period -- a persistence assumption.
  * hmm: one-step-ahead forecast xi_t' P from the filtered posterior.
States are always ordered so that 'crisis' is the high-vol state.

With DETECTOR = "none" every channel collapses exactly to the un-regimed model
(weights all 1, no extra columns, theta = 1), which is the regression test.
"""

import sys
from functools import lru_cache
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))


# ----
# Detector
# ----

DETECTOR = "none"       # "none" | "changepoint" | "hmm" | "wasserstein"
REGIME   = None         # None | "calm" | "crisis"

# Every detector writes one daily CSV per input series to
#     _regimes/<detector>/regimes_final/<detector>_<series>.csv
# with at least the columns date, p_calm, p_crisis, already shifted one day
# (value at d uses data through d-1).  Models do not pick the series by hand:
# they loop over detector_series() and call use_series(s), so one run
# produces separate results per series, tagged _vix / _gspc in the output name.
DETECTOR_SERIES = {
    "changepoint": ("vix", "gspc"),     # VIX breaks primary, GSPC robustness
    "hmm":         ("vix", "gspc"),
    "wasserstein": ("vix", "gspc"),
}
SERIES = "vix"                          # set by use_series(), not by hand

# Optional smoothing of the daily signal before it is read on a rebalance or
# training date: exponentially weighted mean of p_crisis over past days, so a
# single-day spike does not decide a whole holding period.  Causal (uses only
# rows <= d, which are already shifted), applied the same way to every
# detector.  None = raw daily value; 5 = half-life of 5 trading days.
SMOOTH_HALFLIFE = None


def detector_series() -> list:
    """Series to run for the current DETECTOR ([None] for 'none')."""
    if DETECTOR == "none":
        return [None]
    if DETECTOR not in DETECTOR_SERIES:
        raise ValueError(f"[regime] unknown DETECTOR {DETECTOR!r}")
    return list(DETECTOR_SERIES[DETECTOR])


def use_series(series) -> str:
    """
    Point regime_probs at `series` and reset the Channel-3 Schmitt-trigger
    state so the next series does not inherit the previous one's crisis flag.
    Returns the tag to append to the output name ("" for DETECTOR 'none').
    """
    global SERIES
    _crisis_state["on"] = False
    if series is None:
        return ""
    if series not in DETECTOR_SERIES.get(DETECTOR, ()):
        raise ValueError(f"[regime] {DETECTOR}: unknown series {series!r}")
    SERIES = series
    return f"_{series}"


def _regime_file(detector: str, series: str) -> Path:
    return _ROOT / "_regimes" / detector / "regimes_final" / f"{detector}_{series}.csv"


@lru_cache(maxsize=None)
def _load(detector: str, series: str) -> pd.DataFrame:
    """Read one detector/series file once per process."""
    path = _regime_file(detector, series)
    if not path.exists():
        raise FileNotFoundError(f"[regime] {detector}/{series}: no regime file at {path}")
    df = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
    missing = {"p_calm", "p_crisis"} - set(df.columns)
    if missing:
        raise KeyError(f"[regime] {path.name} lacks columns {sorted(missing)}")
    df = df[["p_calm", "p_crisis"]].astype(float)
    if not np.allclose(df.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"[regime] {path.name}: p_calm + p_crisis != 1 on some rows")
    return df


@lru_cache(maxsize=None)
def _signal(detector: str, series: str, halflife) -> pd.DataFrame:
    """Daily p_calm/p_crisis, EW-smoothed if a half-life is given."""
    df = _load(detector, series)
    if halflife is None:
        return df
    pc = df["p_crisis"].ewm(halflife=halflife).mean().clip(0.0, 1.0)
    return pd.DataFrame({"p_calm": 1.0 - pc, "p_crisis": pc}, index=df.index)


def regime_probs(dates) -> pd.DataFrame:
    """
    One row per date, columns ['p_calm', 'p_crisis'], summing to 1.
    """
    idx = pd.DatetimeIndex(dates)
    if DETECTOR == "none":
        return pd.DataFrame({"p_calm": 1.0, "p_crisis": 0.0}, index=idx)
    src = _signal(DETECTOR, SERIES, SMOOTH_HALFLIFE)
    # forward-fill to the last regime date <= d, so non-trading days resolve
    out = src.reindex(idx, method="ffill")
    if out.isna().any().any():
        bad = out.index[out.isna().any(axis=1)]
        raise ValueError(f"[regime] {DETECTOR}/{SERIES}: no value for {len(bad)} date(s), "
                         f"first {bad[0].date()} -- a NaN here would silently "
                         f"corrupt sample weights")
    return out


def _regime_label(dates) -> pd.Series:
    """
    Hard {calm, crisis} label from p_crisis -- only for the REGIME hard-split spec.
    """
    p = regime_probs(pd.DatetimeIndex(dates))["p_crisis"]
    return pd.Series(np.where(p.to_numpy() > 0.5, "crisis", "calm"),
                     index=p.index, name="regime")


# ---- Channel 1: sample weights (primary) ------------------------------------
USE_REGIME_WEIGHTS  = False
REGIME_WEIGHT_FLOOR = 0.5       # w = floor + (1-floor) * similarity; keeps n_eff up
REGIME_DECAY_HL     = None      # optional half-life in months

def _obs_weights(tr_months, d):
    """
    Observation weight per training month: resemblance to d's expected regime.
    Near-binary p_crisis makes raw similarity near-binary too, which collapses
    n_eff in crisis periods (~24/60); the floor keeps it near ~56/60.
    """
    if not USE_REGIME_WEIGHTS:
        return None
    P      = regime_probs(tr_months.union(pd.DatetimeIndex([d])))
    p_now  = float(P.loc[d, "p_crisis"])
    p_hist = P.loc[tr_months, "p_crisis"].to_numpy()
    sim = p_hist * p_now + (1.0 - p_hist) * (1.0 - p_now)
    w = REGIME_WEIGHT_FLOOR + (1.0 - REGIME_WEIGHT_FLOOR) * sim
    if REGIME_DECAY_HL:
        age = (d - tr_months).days / 30.44
        w = w * 0.5 ** (age / REGIME_DECAY_HL)
    w = np.clip(w, 1e-6, None)
    return pd.Series(w / w.mean(), index=tr_months)


# ---- Channel 2: interaction features ----------------------------------------
USE_REGIME_FEATURES = False
INTERACT_ON = ["mom_12_1", "beta_12m", "vol_3m",
               "downside_beta_12m", "dollar_vol_level"]

_regfeat_log = {"done": False}

def _add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Append f"{feat}_x_crisis" = (rank - 0.5) * p_crisis.  Ranks are in (0, 1],
    so centring makes the interaction symmetric.  Raw p_crisis is not added for
    tree models: it is constant within a date, so a split on it has zero gain
    (the LSTM adds it separately as a sequence channel).
    Gated on the detector, so a live detector gives every date the same columns.
    """
    if not USE_REGIME_FEATURES:
        return panel
    if DETECTOR == "none":
        if not _regfeat_log["done"] and mp.current_process().name == "MainProcess":
            print("[regime] Channel 2: DETECTOR='none' -> interaction columns not "
                  "appended (stub regression test)", flush=True)
            _regfeat_log["done"] = True
        return panel
    missing = [f for f in INTERACT_ON if f not in panel.columns]
    if missing:
        raise KeyError(f"[regime] INTERACT_ON features not in panel: {missing}")
    dates = panel.index.get_level_values("date")
    pc = regime_probs(dates.unique())["p_crisis"].reindex(dates).to_numpy()
    return panel.assign(**{f"{f}_x_crisis": (panel[f].to_numpy() - 0.5) * pc
                           for f in INTERACT_ON})


# ---- Channel 3: rank sharpness ----------------------------------------------
USE_REGIME_THETA = False
THETA_CALM, THETA_CRISIS = 1.0, 0.5       # 0.0 would make crisis portfolios exactly 1/N
ENTER_CRISIS, EXIT_CRISIS = 0.6, 0.4      # Schmitt trigger

_crisis_state = {"on": False}

def _theta(d) -> float:
    """
    Rank-sharpness exponent for rebalance date d.  Stateful: call once per
    date, in order; models reset _crisis_state at the start of each run.
    """
    if not USE_REGIME_THETA:
        return 1.0
    p = float(regime_probs([d])["p_crisis"].iloc[0])
    _crisis_state["on"] = p > (EXIT_CRISIS if _crisis_state["on"] else ENTER_CRISIS)
    return THETA_CRISIS if _crisis_state["on"] else THETA_CALM