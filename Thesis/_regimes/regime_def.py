
"""
Central regime machinery, shared by every model (xgb / lstm / rf / ...).

A model file picks its own DETECTOR / REGIME / USE_REGIME_* knobs by setting
them on this module (`regime_def.DETECTOR = "changepoint"`) right after
import, then calls the functions below -- the mechanism lives here once
instead of being copy-pasted per model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----
# Regime implementation
# ----

DETECTOR = "none"       # "none" | "hmm" | "wasserstein" | "changepoint"
REGIME   = None         # None | "calm" | "crisis"


def regime_probs(dates) -> pd.DataFrame:
    """
    One row per date, columns ['p_calm', 'p_crisis'], summing to 1.  These are
    p_next (one-step-ahead forecasts), not filtered probabilities.  States are
    sorted by in-state realized volatility at every refit, so 'crisis' is always
    the high-vol state.
    """ 
    idx = pd.DatetimeIndex(dates)
    if DETECTOR == "none":
        return pd.DataFrame({"p_calm": 1.0, "p_crisis": 0.0}, index=idx)
    if DETECTOR == "changepoint":
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from _regimes.changepoint.main import crisis_probs
        return crisis_probs(idx)
    raise NotImplementedError(DETECTOR)

 
def _regime_label(dates) -> pd.Series:
    """Hard {calm, crisis} label from p_crisis -- only for the REGIME hard-split spec."""
    p = regime_probs(pd.DatetimeIndex(dates))["p_crisis"]
    return pd.Series(np.where(p.to_numpy() > 0.5, "crisis", "calm"),
                     index=p.index, name="regime")

# ---- Channel 1: sample weights (primary) --------------------------------------
USE_REGIME_WEIGHTS = False
REGIME_DECAY_HL    = None

def _obs_weights(tr_months, d):
    """Observation weight per training month: resemblance to d's expected regime."""
    if not USE_REGIME_WEIGHTS:
        return None
    P      = regime_probs(tr_months.union(pd.DatetimeIndex([d])))
    p_now  = float(P.loc[d, "p_crisis"])
    p_hist = P.loc[tr_months, "p_crisis"].to_numpy()
    w = p_hist * p_now + (1.0 - p_hist) * (1.0 - p_now)
    if REGIME_DECAY_HL:
        age = (d - tr_months).days / 30.44
        w  = w * 0.5 ** (age / REGIME_DECAY_HL)
    w = np.clip(w, 1e-6, None)
    return pd.Series(w / w.mean(), index=tr_months)


# ---- Channel 2: interaction features ----------------------------------------
USE_REGIME_FEATURES = True
INTERACT_ON = ["mom_12_1", "beta_12m", "vol_3m",
               "downside_beta_12m", "dollar_vol_level"]

_regfeat_log = {"done": False}

def _add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Append f"{feat}_x_crisis" = (rank - 0.5) * p_crisis for each INTERACT_ON
    feature.  Panel features are cross-sectional ranks in (0, 1], so centring at
    0.5 makes the interaction symmetric and the split points interpretable.  Raw
    p_crisis is never added as a column: it is constant within a date, so a split
    on it just partitions by date with zero gain.
    """
    if not USE_REGIME_FEATURES:
        return panel
    dates = panel.index.get_level_values("date")
    pc = regime_probs(dates.unique())["p_crisis"].reindex(dates).to_numpy()
    cols = {f"{f}_x_crisis": (panel[f].to_numpy() - 0.5) * pc for f in INTERACT_ON}
    all_zero = not any(np.any(v) for v in cols.values())
    if all_zero:
        if not _regfeat_log["done"]:
            print(f"[regime] Channel 2: built {len(cols)} _x_crisis cols "
                  f"({', '.join(cols)}), all exactly 0 -> not appended "
                  f"(keeps colsample_bytree subsets identical)", flush=True)
            _regfeat_log["done"] = True
        return panel
    return panel.assign(**cols)


# ---- Channel 3: rank sharpness --------------------------------------------
USE_REGIME_THETA = True
THETA_CALM, THETA_CRISIS = 1.0, 0.0       # exponent on the pct-rank; 0 -> equal weight
ENTER_CRISIS, EXIT_CRISIS = 0.6, 0.4      # Schmitt trigger (hysteresis kills turnover churn)

_crisis_state = {"on": False}

def _theta(d) -> float:
    """Rank-sharpness exponent for rebalance date d.  Stateful: call once per date, in order."""
    if not USE_REGIME_THETA:
        return 1.0
    p = float(regime_probs([d])["p_crisis"].iloc[0])
    if _crisis_state["on"]:
        _crisis_state["on"] = p > EXIT_CRISIS
    else:
        _crisis_state["on"] = p > ENTER_CRISIS
    pc = 1.0 if _crisis_state["on"] else 0.0
    return pc * THETA_CRISIS + (1.0 - pc) * THETA_CALM

