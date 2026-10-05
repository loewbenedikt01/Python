'''
Hidden Markov model regimes -> the common regime interface.

Two-state Gaussian HMM with time-varying parameters, estimated adaptively as in
Nystrup, Madsen & Lindstrom (2018, Quantitative Finance, section 2 and 4.2),
with the recursive estimator of Nystrup et al. (2016, J. Forecasting, sec. 4).
One model per series:
    vix   VIX log changes
    gspc  S&P 500 log returns      (the paper's setting: index log returns)
Input is the same log-return parquet the changepoint detector uses
(_database/regimes_data_log.parquet, written by changepoint/data_prep.py).

Procedure (see hmm_core.py for the equations):
  1. initialisation: the weighted likelihood is maximised numerically on the
     first INIT_DAYS returns, giving starting parameters and Fisher
     information (paper: 'The first 260 observations were used for
     initialization');
  2. every following day one recursive update with the newest return,
     effective memory N_EFF = 260 days, A = 1/N_EFF (paper sec. 4.2);
  3. after each update the forward filter gives P(S_t | r_1..r_t) (eq. 8) and
     the one-step forecast P(S_{t+1} | r_1..r_t) = alpha_t Gamma_t (eq. 9).

Output, point-in-time: p_crisis(d) is the forecast made at the close of d-1,
so the value at d uses only data through d-1.  This matches the one-day shift
of the changepoint files.  States are ordered by variance: crisis = high vol.
The first value is on observation INIT_DAYS + 1.

Output: regimes_final/hmm_{vix,gspc}.csv, daily, read by regime_def.py.
Columns p_calm, p_crisis, label match the changepoint files; the rest are
HMM diagnostics (parameters in annualised units, forecast mean and vol of the
mixture, eqs. 10-11).
'''

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hmm_engine import adaptive_step, filtered, initialise, to_params

# ----
# Parameters
# ----

SERIES      = {'vix': '^VIX', 'gspc': '^GSPC'}   # output tag -> column in the parquet
N_EFF       = 260           # effective memory in days, 1/(1-lambda) (paper sec. 4.2); 520 as robustness
INIT_DAYS   = 260           # paper: first 260 observations used for initialisation
N_STATES    = 2             # two-state model, as in the paper
SCALE_X     = 100           # returns in percent, keeps the optimiser well-behaved (units only)
N_JOBS      = -1            # the two series run in parallel; each is sequential by construction

WINDOW      = 10 * N_EFF    # observations in the weighted score; older weights < e^-10
                            # (Nystrup et al. 2016, p. 6: 2,500 observations for N_eff = 250)

HERE      = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / '_database' / 'regimes_data_log.parquet'
OUT_DIR   = HERE / 'regimes_final'


# ----
# Data
# ----

def load_returns(path: Path = DATA_PATH) -> pd.DataFrame:
    '''
    Daily log returns, one column per ticker, DatetimeIndex.
    '''
    df = pd.read_parquet(path)
    for c in ('Date', 'date'):
        if c in df.columns:
            df = df.set_index(c)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


# ----
# Adaptive estimation and filtering
# ----

def _run(x: np.ndarray) -> list[tuple]:
    '''
    Walk through the sample once.  Row i holds the forecast for day i, made
    with the parameters and filter at the close of day i-1.
    '''
    if N_STATES != 2:
        raise ValueError('the adaptive estimator is implemented for 2 states, as in the paper')
    if len(x) <= INIT_DAYS:
        raise ValueError(f'need more than INIT_DAYS = {INIT_DAYS} returns, got {len(x)}')

    th, I = initialise(x[:INIT_DAYS], N_EFF)
    loglik = np.nan
    rows = []
    for i in range(INIT_DAYS - 1, len(x)):
        if i >= INIT_DAYS:
            th, I, loglik = adaptive_step(th, I, x[max(0, i + 1 - WINDOW):i + 1], i + 1, N_EFF)
        if i + 1 < len(x):
            p = to_params(th)
            alpha = filtered(th, x[max(0, i + 1 - WINDOW):i + 1])
            rows.append((i + 1, alpha @ p.A, alpha, p, loglik))
    return rows


def build(r: pd.Series, out_path: Path) -> pd.DataFrame:
    r = r.dropna().sort_index()
    idx = r.index
    rows = _run(r.to_numpy(dtype=float) * SCALE_X)
    return _write(idx, rows, out_path)


def _write(idx: pd.DatetimeIndex, rows: list[tuple], out_path: Path) -> pd.DataFrame:
    pos = np.array([q[0] for q in rows])
    fc  = np.array([q[1] for q in rows])
    fil = np.array([q[2] for q in rows])
    mu  = np.array([q[3].mu for q in rows])
    sg  = np.array([q[3].sigma for q in rows])
    ann = np.sqrt(252) / SCALE_X

    m = (fc * mu).sum(axis=1)
    v = (fc * (mu ** 2 + sg ** 2)).sum(axis=1) - m ** 2

    df = pd.DataFrame({
        'p_crisis':          np.clip(fc[:, -1], 0.0, 1.0),
        'p_crisis_filtered': fil[:, -1],
        'sigma_calm_ann':    sg[:, 0] * ann,
        'sigma_crisis_ann':  sg[:, -1] * ann,
        'mu_calm_ann':       mu[:, 0] * 252 / SCALE_X,
        'mu_crisis_ann':     mu[:, -1] * 252 / SCALE_X,
        'p_stay_calm':       [q[3].A[0, 0] for q in rows],
        'p_stay_crisis':     [q[3].A[-1, -1] for q in rows],
        'loglik':            [q[4] for q in rows],
        'n_train':           np.minimum(pos, WINDOW),
        'refit_date':        idx[pos - 1],
        'mu_fc_ann':         m * 252 / SCALE_X,
        'sigma_fc_ann':      np.sqrt(v) * ann,
    }, index=idx[pos])
    df.index.name = 'date'
    df.insert(0, 'p_calm', 1.0 - df['p_crisis'])
    df.insert(2, 'label', np.where(df['p_crisis'] > 0.5, 'crisis', 'calm'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


if __name__ == '__main__':
    rets = load_returns()
    for col in SERIES.values():
        if col not in rets.columns:
            raise KeyError(f'[hmm] column {col!r} not in {DATA_PATH.name}: {list(rets.columns)}')
    outs = Parallel(n_jobs=N_JOBS)(
        delayed(build)(rets[col], OUT_DIR / f'hmm_{tag}.csv') for tag, col in SERIES.items()
    )
    for tag, out in zip(SERIES, outs):
        print(f'[hmm] {tag}: {len(out)} rows, {out.index.min().date()} .. {out.index.max().date()}, '
              f'crisis on {(out['p_crisis'] > 0.5).mean():.1%} of days')
