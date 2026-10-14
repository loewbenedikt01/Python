'''
Central regime machinery, shared by every model (mvo / hrp / xgb / rf / lstm).

Run options, set by hand in the model file right after import:
    regime_def.DETECTOR      = 'none'                  baseline, no regime information
    regime_def.DETECTOR      = <detector>              regime run with one detector series
    regime_def.SERIES        = <series>
    regime_def.CRISIS_TARGET = 'cash' | 'hrp' | 'min_var'
Each combination is its own run with its own output name (run_tag()), so runs
can be split across machines or sessions.

What a regime run does
----------------------
The model itself is unchanged: in calm it holds exactly what its baseline
holds.  On every decision date d with in_crisis(d) (p_crisis > 0.5) the whole
portfolio switches once to CRISIS_TARGET:
  * 'cash'     100% in 3-month T-bills (risk_free.rf_daily); every model;
  * 'hrp'      the HRP portfolio of hrp.py at d; hybrid runs of XGB / RF / LSTM;
  * 'min_var'  minimum variance under the weight box; MVO / HRP robustness run.
Decision dates are the scheduled rebalances of the run's frequency.

The switch lives in crisis_overlay(), which every model calls with two
functions: its own weights at d (calm_fn) and the crisis portfolio at d
(crisis_fn, only for 'hrp' and 'min_var').

p_crisis semantics
------------------
regime_probs(d) returns the probability that the holding period starting at d
is in the high-volatility state, computed from data through d-1 only.
  * changepoint: current-regime estimate (EWMA vol of the active segment),
    used as the forecast for the holding period, a persistence assumption.
  * hmm: one-step-ahead forecast alpha_t Gamma_t from the filtered posterior.
  * wasserstein: relative squared W2 distance of the last month's returns to
    the calm and crisis centroids.
States are always ordered so that 'crisis' is the high-vol state.

With DETECTOR = 'none' every model reduces exactly to its baseline.
'''

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:                 # once, at import, not per call
    sys.path.append(str(_ROOT))


# ----
# Run options (set in the model file)
# ----

DETECTOR      = 'none'      # 'none' | 'changepoint' | 'hmm' | 'wasserstein'
SERIES        = None        # see DETECTOR_SERIES; ignored for 'none'
CRISIS_TARGET = 'cash'      # 'cash' | 'hrp' | 'min_var'; what the regime run holds in crisis

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

# Crisis targets each model may use.
CRISIS_TARGETS = {
    'mvo':  ('cash', 'min_var'),
    'hrp':  ('cash', 'min_var'),
    'xgb':  ('cash', 'hrp'),
    'rf':   ('cash', 'hrp'),
    'lstm': ('cash', 'hrp'),
}

# Ticker name of the T-bill position in target weights (portfolio.py prices it
# with risk_free.rf_daily and charges no trading cost on it).
CASH = 'CASH'

# A decision date counts as crisis when p_crisis is above this threshold.
CRISIS_THRESHOLD = 0.5

# Optional smoothing of the daily signal before it is read on a decision
# date: exponentially weighted mean of p_crisis over past days, so a
# single-day spike does not decide a whole holding period.  Causal (uses only
# rows <= d, which are already shifted), applied the same way to every
# detector.  None = raw daily value; 5 = half-life of 5 trading days.
SMOOTH_HALFLIFE = None

# Dates before a detector's first value (e.g. the HMM's 260-day
# initialisation) get p_crisis = 0.5: no regime information.  Any request on
# or after NEUTRAL_BEFORE (config.START_DATE, the first rebalance) without a
# value still raises.
PRE_SAMPLE_NEUTRAL = True
NEUTRAL_BEFORE     = '1998-01-01'


def run_tag() -> str:
    '''
    Check the run options and return the suffix for output names: '' for the
    baseline, e.g. '_hmm_gspc_cash' or '_wasserstein_gspc_vix_hrp'.
    '''
    if DETECTOR == 'none':
        return ''
    if DETECTOR not in DETECTOR_SERIES:
        raise ValueError(f'[regime] unknown DETECTOR {DETECTOR!r}; '
                         f'choose none or one of {list(DETECTOR_SERIES)}')
    if SERIES not in DETECTOR_SERIES[DETECTOR]:
        raise ValueError(f'[regime] {DETECTOR}: SERIES must be one of '
                         f'{list(DETECTOR_SERIES[DETECTOR])}, got {SERIES!r}')
    if CRISIS_TARGET not in ('cash', 'hrp', 'min_var'):
        raise ValueError(f'[regime] unknown CRISIS_TARGET {CRISIS_TARGET!r}')
    return f'_{DETECTOR}_{SERIES}_{CRISIS_TARGET}'


def check_run(model: str) -> str:
    '''run_tag(), plus a check that CRISIS_TARGET is allowed for this model.'''
    tag = run_tag()
    if DETECTOR != 'none' and CRISIS_TARGET not in CRISIS_TARGETS[model]:
        raise ValueError(f'[regime] {model}: CRISIS_TARGET must be one of '
                         f'{list(CRISIS_TARGETS[model])}, got {CRISIS_TARGET!r}')
    return tag


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
# The crisis switch, shared by every model
# ----

def crisis_overlay(scheduled, calm_fn, crisis_fn=None):
    '''
    Target weights of one run.

    scheduled  the frequency's rebalance dates
    calm_fn    d -> pd.Series of the model's own weights at d, or None when the
               model has no weights there (the date is then skipped and
               portfolio.py holds the previous target)
    crisis_fn  d -> pd.Series of the crisis portfolio at d (CRISIS_TARGET
               'hrp' or 'min_var'), or None if it cannot be built

    The run starts at the first scheduled date where calm_fn has weights, so
    a regime run covers exactly the period of its baseline.

    Returns (targets, diag):
      targets   date x ticker; a cash date is one row with CASH = 1
      diag      per rebalance date: p_crisis, crisis, holding
                ('model' | 'cash' | 'hrp' | 'min_var' | 'none')
    '''
    scheduled = pd.DatetimeIndex(scheduled)
    probs = regime_probs(scheduled)['p_crisis']
    target = CRISIS_TARGET if DETECTOR != 'none' else None
    if target in ('hrp', 'min_var') and crisis_fn is None:
        raise ValueError(f'[regime] CRISIS_TARGET {target!r} needs a crisis_fn')

    rows, diag, started = {}, {}, False
    for d in scheduled:
        calm_w = None
        if not started:                         # first date with model weights
            calm_w = calm_fn(d)
            if calm_w is None:
                continue
            started = True
        crisis = target is not None and bool(probs.loc[d] > CRISIS_THRESHOLD)
        if not crisis:
            w, holding = (calm_w if calm_w is not None else calm_fn(d)), 'model'
        elif target == 'cash':
            w, holding = pd.Series({CASH: 1.0}), 'cash'
        else:
            w, holding = crisis_fn(d), target
        if w is None or not len(w):
            holding = 'none'
        else:
            rows[d] = w
        diag[d] = {'p_crisis': float(probs.loc[d]), 'crisis': crisis, 'holding': holding}

    targets = pd.DataFrame(rows).T
    diag = pd.DataFrame(diag).T.rename_axis('date')
    if len(diag):
        diag['p_crisis'] = diag['p_crisis'].astype(float)
        diag['crisis'] = diag['crisis'].astype(bool)
    return targets, diag


# ----
# ML models (XGB / RF / LSTM)
# ----

# A regime run of an ML model holds exactly the baseline's weights in calm
# (the regime is no model input), so it reads the baseline's stored target
# weights instead of training the same ensemble again.  The baseline run (or
# any run that trains) stores them in _output/<model>/_baseline_targets/.
# False = always train.
REUSE_BASELINE = True


def baseline_cache(model: str, name: str, frequency: str) -> Path:
    '''File holding (targets, n_dates, diagnostics) of one ML model and frequency.'''
    from config import OUTPUT_ROOT
    return Path(OUTPUT_ROOT) / model / '_baseline_targets' / f'{name}_{frequency.lower()}.pkl'


def ml_targets(model: str, name: str, frequency: str, train):
    '''
    (targets, n_dates, diagnostics) of an ML model at its scheduled dates.
    name is the MODEL_NAME without regime tag; train() fits the ensemble.
    '''
    path = baseline_cache(model, name, frequency)
    if DETECTOR != 'none' and REUSE_BASELINE and path.exists():
        stamp = pd.Timestamp(path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
        print(f'[{model}] {frequency}: calm weights read from the baseline run '
              f'{path.name} (written {stamp}); REUSE_BASELINE = False to retrain')
        return pd.read_pickle(path)
    out = train()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


def ml_overlay(model_targets: pd.DataFrame, scheduled, crisis_fn=None):
    '''
    crisis_overlay() for XGB / RF / LSTM: calm weights are the model's target
    rows at its scheduled dates.
    '''
    ml = model_targets.sort_index()

    def calm(d):
        return ml.loc[d].dropna() if d in ml.index else None

    return crisis_overlay(scheduled, calm, crisis_fn)
