'''
Compare every run in _output in one Excel workbook.

Reads _output/<model>/<run>/raw/daily_returns.csv for every model folder,
every run and every frequency (_comparison and _baseline_targets are
skipped) and computes all metrics from the daily returns, so the numbers do
not depend on the metrics.csv each run wrote and every run is measured the
same way.

Sheets
  overall            full period (crises.overall)
  one per crisis     crises.main_crises (and crises.sub_crises with
                     SUB_CRISES = True): metrics over peak -> recovery, plus
                     the cumulative return peak -> trough
  windows            the dates of every sheet and the benchmark used
Rows: one per run and frequency.  The run settings (detector, series, crisis
target) are read from the run name.  Columns: one per metric.

Metrics: cumulative and annual return, volatility, Sharpe, Sortino, maximum
drawdown, longest time under water (trading days), Calmar, Ulcer index,
historical VaR / CVaR (daily, TAIL_LEVEL), beta and downside capture against
the benchmark, share of days in the crisis portfolio; on the overall sheet
also annual turnover and the Ledoit-Wolf (2008) Sharpe test against the
benchmark (compare_runs.sharpe_test).  Sharpe and Sortino use the risk-free
rate of risk_free.py, as in metrics.py.

Output: _output/_comparison/compare_all.xlsx (needs openpyxl); without
openpyxl, one CSV per sheet in _output/_comparison/compare_all/.

Usage: set BENCH_MODEL / BENCH_RUN below if needed and run the file.
'''

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / '_metrics'))
from config import OUTPUT_ROOT
from crises import overall, main_crises, sub_crises
from metrics import (
    annualized_return,
    annualized_volatility,
    beta,
    calmar_ratio,
    conditional_value_at_risk,
    cumulative_return,
    downside_capture,
    max_drawdown_duration,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    value_at_risk,
)
from compare_runs import CRISIS_NAMES, sharpe_test

# ----
# Variables
# ----

# Benchmark for beta, downside capture and the Sharpe test: the run in
# _output/<BENCH_MODEL>/ with the same frequency.  BENCH_RUN = the MODEL_NAME
# without frequency; None = the only run in that folder (the first one, with a
# note, if there are several).
BENCH_MODEL = 'market_cap'
BENCH_RUN   = None

SUB_CRISES  = False                  # also one sheet per sub-crisis
TAIL_LEVEL  = 0.95                   # VaR / CVaR level (daily returns)

# Row order of the model folders; folders not listed come after, alphabetically.
MODEL_ORDER = ['sp500', 'market_cap', 'equal_weight', 'mvo', 'hrp', 'xgb', 'rf', 'lstm']
FREQ_ORDER  = ['monthly', 'quarterly', 'yearly']

SKIP_FOLDERS = {'_comparison', '_baseline_targets'}
SAVE_DIR     = Path(OUTPUT_ROOT) / '_comparison'

DETECTORS = ('changepoint', 'hmm', 'wasserstein')
SERIES    = ('gspc_vix', 'gspc', 'vix')
TARGETS   = ('min_var', 'cash', 'hrp')

PCT_COLS = ['cum_return', 'ptt_return', 'ann_return', 'ann_vol', 'max_dd',
            'var_95', 'cvar_95', 'crisis_share']


# ----
# Finding and reading the runs
# ----

def _parse(model: str, folder: str) -> dict | None:
    '''model, run name, frequency and regime settings from a run folder name.
    A folder without frequency (sp500, buy & hold) gets frequency None.'''
    m = re.match(r'^(.*)_(' + '|'.join(FREQ_ORDER) + r')$', folder)
    name, freq = m.groups() if m else (folder, None)
    info = {'model': model, 'run': name, 'frequency': freq,
            'detector': 'none', 'series': '', 'crisis_target': ''}
    pat = (r'_(' + '|'.join(DETECTORS) + r')_(' + '|'.join(SERIES) + r')'
           r'(?:_(' + '|'.join(TARGETS) + r'))?$')
    r = re.search(pat, name)
    if r:
        info['detector'], info['series'] = r.group(1), r.group(2)
        info['crisis_target'] = r.group(3) or 'old run'
    elif name.endswith('_minvar'):
        info['crisis_target'] = 'always min_var'
    return info


def _read(path: Path, col: str | None = None) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df[col] if col else df.iloc[:, 0]
    return s[~s.index.duplicated()].sort_index()


def find_runs() -> list[dict]:
    runs = []
    for mdir in sorted(Path(OUTPUT_ROOT).iterdir()):
        if not mdir.is_dir() or mdir.name in SKIP_FOLDERS or mdir.name.startswith('.'):
            continue
        for rdir in sorted(mdir.iterdir()):
            ret_path = rdir / 'raw' / 'daily_returns.csv'
            info = _parse(mdir.name, rdir.name) if rdir.is_dir() else None
            if not ret_path.exists():
                continue
            info['returns'] = _read(ret_path, 'log_return').dropna()
            info['turnover'] = _read(rdir / 'raw' / 'turnover.csv')
            crisis = None
            reg = rdir / 'raw' / 'regime.csv'
            if reg.exists():
                r = pd.read_csv(reg, index_col=0, parse_dates=True)
                if 'crisis' in r.columns:
                    crisis = r['crisis'].astype(str).str.lower().eq('true').astype(float)
                    crisis = crisis[~crisis.index.duplicated()].sort_index()
            info['crisis'] = crisis
            if info['frequency'] is not None:
                runs.append(info)
            else:                            # no rebalancing: one reference row per frequency
                runs += [{**info, 'frequency': f} for f in FREQ_ORDER]
    return runs


def _benchmarks(runs: list[dict]) -> dict:
    '''frequency -> daily log returns of the benchmark run.'''
    out = {}
    for freq in FREQ_ORDER:
        cand = [r for r in runs if r['model'] == BENCH_MODEL and r['frequency'] == freq
                and (BENCH_RUN is None or r['run'] == BENCH_RUN)]
        if not cand:
            continue
        if len(cand) > 1:
            first = cand[0]['run']
            print(f'[compare_all] {len(cand)} {BENCH_MODEL} runs for {freq}; using '
                  f'{first} (set BENCH_RUN to choose)')
        out[freq] = cand[0]
    return out


# ----
# Metrics for one window
# ----

def _slice(s: pd.Series, start, end) -> pd.Series:
    '''Values in (start, end], as in export.py.'''
    return s.loc[(s.index > pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def _level(r: pd.Series) -> pd.Series:
    return pd.Series(np.exp(np.concatenate([[0.0], r.cumsum().to_numpy()])))


def _crisis_share(crisis: pd.Series | None, idx: pd.DatetimeIndex) -> float:
    '''Share of the window's trading days held in the crisis portfolio.'''
    if crisis is None or not len(idx):
        return np.nan
    return float(crisis.reindex(idx, method='ffill').fillna(0.0).mean())


def window_metrics(run: dict, start, end, bench: pd.Series | None,
                   trough=None, full: bool = False) -> dict:
    r = _slice(run['returns'], start, end)
    out = {'days': len(r)}
    if len(r) < 2:
        return out
    lvl = _level(r)
    if trough is not None:
        out['ptt_return'] = cumulative_return(_slice(run['returns'], start, trough))
    out.update({
        'cum_return':   cumulative_return(r),
        'ann_return':   annualized_return(r),
        'ann_vol':      annualized_volatility(r),
        'sharpe':       sharpe_ratio(r),
        'sortino':      sortino_ratio(r),
        'max_dd':       maximum_drawdown(lvl),
        'dd_days':      max_drawdown_duration(lvl),
        'calmar':       calmar_ratio(r, lvl),
        'ulcer':        ulcer_index(lvl),
        'var_95':       value_at_risk(r, TAIL_LEVEL),
        'cvar_95':      conditional_value_at_risk(r, TAIL_LEVEL),
        'crisis_share': _crisis_share(run['crisis'], r.index),
    })
    if bench is not None:
        b = _slice(bench, start, end)
        out['beta'] = beta(r, b)
        out['down_capture'] = downside_capture(r, b)
    if full:
        to = run['turnover']
        if to is not None and len(to) > 1:
            years = (r.index[-1] - r.index[0]).days / 365.25
            out['turnover_pa'] = float(_slice(to, start, end).iloc[1:].sum() / years)
        if bench is not None and run.get('is_bench') is not True:
            out['d_sharpe_bench'], out['p_bench'] = sharpe_test(r, _slice(bench, start, end))
    return out


# ----
# Tables
# ----

def _sort_key(run: dict):
    m = run['model']
    mi = MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
    return (mi, m, FREQ_ORDER.index(run['frequency']), run['detector'] != 'none',
            run['crisis_target'], run['detector'], run['series'], run['run'])


def build_tables(runs: list[dict]) -> tuple[dict, pd.DataFrame]:
    benches = _benchmarks(runs)
    for run in runs:
        b = benches.get(run['frequency'])
        run['is_bench'] = b is run
    runs = sorted(runs, key=_sort_key)
    keys = ['model', 'frequency', 'detector', 'series', 'crisis_target', 'run']

    windows = [('overall', overall[0], True)]
    crises = list(main_crises) + (list(sub_crises) if SUB_CRISES else [])
    windows += [(CRISIS_NAMES.get(c['label'], c['label']), c, False) for c in crises]

    sheets, info = {}, []
    for title, w, full in windows:
        rows = []
        for run in runs:
            b = benches.get(run['frequency'])
            row = {k: run[k] for k in keys}
            row.update(window_metrics(run, w['peak'], w['even'],
                                      b['returns'] if b is not None else None,
                                      trough=None if full else w['trough'], full=full))
            rows.append(row)
        sheets[title[:31]] = pd.DataFrame(rows)
        info.append({'sheet': title[:31], 'label': w['label'], 'peak / start': w['peak'],
                     'trough': '' if full else w['trough'], 'recovery / end': w['even']})
    bench_names = {f: BENCH_MODEL + '/' + b['run'] for f, b in benches.items()}
    windows_df = pd.DataFrame(info)
    windows_df['note'] = ''
    windows_df.loc[0, 'note'] = ('returns in (start, end]; crisis sheets: metrics over '
                                 'peak -> recovery, ptt_return = peak -> trough')
    bench_df = pd.DataFrame({'sheet': ['benchmark ' + f for f in bench_names],
                             'label': list(bench_names.values())})
    return sheets, pd.concat([windows_df, bench_df], ignore_index=True)


# ----
# Writing
# ----

def _write_xlsx(sheets: dict, windows: pd.DataFrame, path: Path) -> None:
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in {**sheets, 'windows': windows}.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = 'G2' if name != 'windows' else 'A2'
            for j, col in enumerate(df.columns, 1):
                letter = get_column_letter(j)
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) if len(df) else len(str(col))
                ws.column_dimensions[letter].width = min(max(10, width + 2), 45)
                fmt = ('0.00%' if col in PCT_COLS else '0' if col in ('days', 'dd_days')
                       else '0.000' if df[col].dtype.kind == 'f' else None)
                if fmt:
                    for cell in ws[letter][1:]:
                        cell.number_format = fmt


def main() -> None:
    runs = find_runs()
    if not runs:
        print(f'[compare_all] no runs found in {OUTPUT_ROOT}')
        return
    n_models = len({r['model'] for r in runs})
    print(f'[compare_all] {len(runs)} runs in {n_models} model folders')
    sheets, windows = build_tables(runs)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import openpyxl
        path = SAVE_DIR / 'compare_all.xlsx'
        _write_xlsx(sheets, windows, path)
    except ImportError:
        path = SAVE_DIR / 'compare_all'
        path.mkdir(exist_ok=True)
        for name, df in {**sheets, 'windows': windows}.items():
            df.to_csv(path / f'{name}.csv', index=False)
        print('[compare_all] openpyxl not installed (pip install openpyxl): wrote one CSV per sheet')
    print(f'[compare_all] saved to {path}')


if __name__ == '__main__':
    main()
