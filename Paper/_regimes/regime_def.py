'''
Central regime machinery, shared by every model (mvo / hrp / xgb / rf / lstm).

Every model has exactly two run options:
    DETECTOR = 'none'                      baseline, no regime information
    DETECTOR = <detector>, SERIES = <s>    regime run with one detector series
A model file picks both by hand right after import
    regime_def.DETECTOR = 'hmm'
    regime_def.SERIES   = 'gspc'
so each detector/series combination is its own run with its own output name
(run_tag()), and runs can be split across machines or sessions.

What the regime run does
------------------------
  * MVO:   max-Sharpe in calm, minimum-variance when in_crisis(d).
  * HRP:   HRP in calm, minimum-variance when in_crisis(d) (CRISIS_PORTFOLIO).
  * MVO / HRP with INTRA_TRIGGER = True: the same switch, but the signal is
    also checked daily and a flip between scheduled dates triggers an extra
    rebalance (trigger_dates).
  * XGB / RF / LSTM: the model learns with the regime as input, through
    the interaction features (rank - 0.5) * p_crisis (add_regime_features);
    the LSTM also gets raw p_crisis as an input channel.

p_crisis semantics
------------------
regime_probs(d) returns the probability that the holding period starting at d
is in the high-volatility state, computed from data through d-1 only.
  * changepoint: current-regime estimate (EWMA vol of the active segment),
    used as the forecast for the holding period -- a persistence assumption.
  * hmm: one-step-ahead forecast alpha_t Gamma_t from the filtered posterior.
  * wasserstein: relative squared W2 distance of the last month's returns to
    the calm and crisis centroids.
States are always ordered so that 'crisis' is the high-vol state.

With DETECTOR = 'none' every model reduces exactly to the baseline.
'''

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:                 # once, at import -- not per call
    sys.path.append(str(_ROOT))


# ----
# Detector and series (set in the model file)
# ----

DETECTOR = 'none'       # 'none' | 'changepoint' | 'hmm' | 'wasserstein'
SERIES   = None         # see DETECTOR_SERIES; ignored for 'none'

# Series each detector can run on.  Every detector writes one daily CSV per
# series to  _regimes/<detector>/regimes_final/<detector>_<series>.csv
# with at least the columns date, p_calm, p_crisis, already shifted one day
# (value at d uses data through d-1).
DETECTOR_SERIES = {
    'changepoint': ('vix', 'gspc'),       # VIX breaks primary, GSPC robustness
    'hmm':         ('gspc', 'vix'),       # S&P 500 returns as in the paper; the VIX-change
                                          # HMM mostly catches short VIX jumps (check_hmm.py)
    'wasserstein': ('gspc_vix', 'gspc'),  # 2-d S&P 500 + VIX main, S&P 500 alone robustness
}

# A rebalance date counts as crisis when p_crisis is above this threshold
# (MVO and HRP switch their allocation rule on it).
CRISIS_THRESHOLD = 0.5

# Optional smoothing of the daily signal before it is read on a rebalance or
# training date: exponentially weighted mean of p_crisis over past days, so a
# single-day spike does not decide a whole holding period.  Causal (uses only
# rows <= d, which are already shifted), applied the same way to every
# detector.  None = raw daily value; 5 = half-life of 5 trading days.
SMOOTH_HALFLIFE = None

# Dates before a detector's first value (e.g. the HMM's 260-day
# initialisation) get p_crisis = 0.5: no regime information.  Only early
# training rows are affected; any request on or after NEUTRAL_BEFORE
# (config.START_DATE, the first rebalance) without a value still raises.
PRE_SAMPLE_NEUTRAL = True
NEUTRAL_BEFORE     = '1998-01-01'


def run_tag() -> str:
    '''
    Check the DETECTOR / SERIES choice and return the suffix for output names:
    '' for the baseline, e.g. '_hmm_gspc' for a regime run.
    '''
    if DETECTOR == 'none':
        return ''
    if DETECTOR not in DETECTOR_SERIES:
        raise ValueError(f'[regime] unknown DETECTOR {DETECTOR!r}; '
                         f'choose "none" or one of {list(DETECTOR_SERIES)}')
    if SERIES not in DETECTOR_SERIES[DETECTOR]:
        raise ValueError(f'[regime] {DETECTOR}: SERIES must be one of '
                         f'{list(DETECTOR_SERIES[DETECTOR])}, got {SERIES!r}')
    return f'_{DETECTOR}_{SERIES}'


# ----
# Reading the signal
# ----

def _regime_file(detector: str, series: str) -> Path:
    return _ROOT / '_regimes' / detector / 'regimes_final' / f'{detector}_{series}.csv'


@lru_cache(maxsize=None)
def _load(detector: str, series: str) -> pd.DataFrame:
    '''Read one detector/series file once per process.'''
    path = _regime_file(detector, series)
    if not path.exists():
        raise FileNotFoundError(f'[regime] {detector}/{series}: no regime file at {path}')
    df = pd.read_csv(path, index_col='date', parse_dates=['date']).sort_index()
    missing = {'p_calm', 'p_crisis'} - set(df.columns)
    if missing:
        raise KeyError(f'[regime] {path.name} lacks columns {sorted(missing)}')
    df = df[['p_calm', 'p_crisis']].astype(float)
    if not np.allclose(df.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f'[regime] {path.name}: p_calm + p_crisis != 1 on some rows')
    return df


@lru_cache(maxsize=None)
def _signal(detector: str, series: str, halflife) -> pd.DataFrame:
    '''Daily p_calm/p_crisis, EW-smoothed if a half-life is given.'''
    df = _load(detector, series)
    if halflife is None:
        return df
    pc = df['p_crisis'].ewm(halflife=halflife).mean().clip(0.0, 1.0)
    return pd.DataFrame({'p_calm': 1.0 - pc, 'p_crisis': pc}, index=df.index)


def regime_probs(dates) -> pd.DataFrame:
    '''
    One row per date, columns ['p_calm', 'p_crisis'], summing to 1.
    '''
    idx = pd.DatetimeIndex(dates)
    if DETECTOR == 'none':
        return pd.DataFrame({'p_calm': 1.0, 'p_crisis': 0.0}, index=idx)
    run_tag()                                   # fails early on an invalid choice
    src = _signal(DETECTOR, SERIES, SMOOTH_HALFLIFE)
    # forward-fill to the last regime date <= d, so non-trading days resolve
    out = src.reindex(idx, method='ffill')
    if PRE_SAMPLE_NEUTRAL:
        early = (out.index < src.index.min()) & (out.index < pd.Timestamp(NEUTRAL_BEFORE))
        out.loc[early, ['p_calm', 'p_crisis']] = 0.5
    if out.isna().any().any():
        bad = out.index[out.isna().any(axis=1)]
        raise ValueError(f'[regime] {DETECTOR}/{SERIES}: no value for {len(bad)} date(s), '
                         f'first {bad[0].date()}')
    return out


def p_crisis(d) -> float:
    '''p_crisis for the holding period starting at d (0 for the baseline).'''
    return float(regime_probs([d])['p_crisis'].iloc[0])


def in_crisis(d) -> bool:
    '''True when the holding period starting at d is classed as crisis.'''
    return DETECTOR != 'none' and p_crisis(d) > CRISIS_THRESHOLD


# ----
# Regime-triggered rebalancing (MVO / HRP, INTRA_TRIGGER = True)
# ----

TRIGGER_CONFIRM_DAYS = 3    # a flip counts once it has held this many trading days in a row


def trigger_dates(calendar, scheduled, confirm_days: int = TRIGGER_CONFIRM_DAYS) -> pd.DatetimeIndex:
    '''
    Off-calendar rebalance dates for a regime run.  Walking the trading days in
    `calendar`, the state held is the crisis flag (p_crisis > CRISIS_THRESHOLD)
    at the last rebalance, scheduled or triggered.  A day t that is not a
    scheduled rebalance becomes a trigger when the flag has differed from the
    held state on each of the last `confirm_days` trading days, t included.
    The flag at t uses data through t-1, so the switch trades at t's close
    without look-ahead.  Empty for the baseline.
    '''
    cal = pd.DatetimeIndex(calendar)
    if DETECTOR == 'none' or len(cal) == 0:
        return pd.DatetimeIndex([])
    sched = set(pd.DatetimeIndex(scheduled))
    flag = (regime_probs(cal)['p_crisis'] > CRISIS_THRESHOLD).to_numpy()
    out, held, streak = [], None, 0
    for t, f in zip(cal, flag):
        if t in sched:
            held, streak = f, 0
            continue
        if held is None:                        # before the first scheduled rebalance
            continue
        streak = streak + 1 if f != held else 0
        if streak >= confirm_days:
            out.append(t)
            held, streak = f, 0
    return pd.DatetimeIndex(out)


# ----
# ML models: interaction features
# ----

INTERACT_ON = ['mom_12_1', 'beta_12m', 'vol_3m',
               'downside_beta_12m', 'dollar_vol_level']


def add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    '''
    Append f'{feat}_x_crisis' = (rank - 0.5) * p_crisis for the regime run.
    Ranks are in (0, 1], so centring makes the interaction symmetric.  Raw
    p_crisis is not added for tree models: it is constant within a date, so a
    split on it has zero gain (the LSTM adds it separately as a sequence
    channel).  The baseline (DETECTOR = 'none') gets no extra columns.
    '''
    if DETECTOR == 'none':
        return panel
    missing = [f for f in INTERACT_ON if f not in panel.columns]
    if missing:
        raise KeyError(f'[regime] INTERACT_ON features not in panel: {missing}')
    dates = panel.index.get_level_values('date')
    pc = regime_probs(dates.unique())['p_crisis'].reindex(dates).to_numpy()
    return panel.assign(**{f'{f}_x_crisis': (panel[f].to_numpy() - 0.5) * pc
                           for f in INTERACT_ON})
