'''
Crisis performance of the baseline portfolios over the defined crisis windows.

Baseline = every run without a regime detector (no changepoint / hmm /
wasserstein in the name), e.g. market_cap, equal_weight, mvo, hrp,
xgb, rf, lstm; the S&P 500 index is left out (SKIP_MODELS).  Runs are found as in compare_all.py
(_output/<model>/<run>_<frequency>/raw/daily_returns.csv).

One workbook per frequency: _output/_comparison/crisis_analysis_<frequency>.xlsx
  overall      crises.overall: full-window metrics plus CAGR, Sharpe,
               Sortino, Calmar and turnover
  C1 ... C6    crises.main_crises: full window (peak -> recovery), PTT
               (peak -> trough) and TTP (trough -> recovery)
Rows: one per portfolio (model, run, frequency).  Every window is (start, end]
and the portfolio value is 1 at its start.

Excess return = relative: (1 + r) / (1 + r_bench) - 1, the end value against
the benchmark's.  Unlike r - r_bench it stays comparable over long windows
with large compounded returns (e.g. dot-com, peak -> recovery).

Columns (bench = BENCHMARK run of the same frequency, same window)
  full_cum_return       cumulative return
  full_excess_return    relative excess return
  full_max_dd           maximum drawdown
  full_ulcer            Ulcer index
  full_ann_vol          annualised volatility
  full_underwater_share share of days below the running peak
  full_recovered        True if the value at the end >= the value at the
                        crisis start (full window only: in PTT / TTP the value
                        is reset to 1 at the phase start, so >= 1 means nothing)
  full_cagr, full_sharpe, full_sortino, full_calmar, full_turnover_pa
                        overall sheet only (metrics.py; turnover per year
                        without the initial build, as in compare_all.py)
  hit_rate_full, hit_rate_ptt, hit_rate_ttp
                        overall sheet only: share of the crises C1 ... C6 in
                        which the excess return of that phase is > 0
                        (empty for the benchmark)
  hit_count_full, hit_count_ptt, hit_count_ttp
                        the same as hits / crises, e.g. 4/6
  ptt_cum_return        cumulative return peak -> trough
  ptt_excess_return     relative excess return peak -> trough
  ptt_down_capture      ptt_cum_return / bench ptt_cum_return (< 1 = lost less)
  ptt_max_dd            maximum drawdown peak -> trough
  ttp_cum_return        cumulative return trough -> recovery
  ttp_excess_return     relative excess return trough -> recovery
  ttp_recovery_capture  ttp_cum_return / bench ttp_cum_return (> 1 = recovered more)
  ttp_max_dd            maximum drawdown trough -> recovery

Usage: run the file.
'''

import numpy as np
import pandas as pd

import compare_all as ca
from crises import overall, main_crises
from metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_return,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
)


# ----
# Variables
# ----

BENCHMARK   = 'market_cap'               # model folder of the benchmark run
SKIP_MODELS = {'sp500'}                  # model folders left out (S&P 500 index)
MODEL_ORDER = ['market_cap', 'equal_weight', 'mvo', 'hrp', 'xgb', 'rf', 'lstm']
FREQ_ORDER  = ['monthly', 'quarterly', 'yearly']

SAVE_DIR    = ca.SAVE_DIR

KEY_COLS = ['model', 'run', 'frequency']
PCT_COLS = ['full_cum_return', 'full_excess_return', 'full_max_dd', 'full_ulcer', 'full_ann_vol',
            'full_underwater_share', 'full_cagr', 'full_turnover_pa',
            'hit_rate_full', 'hit_rate_ptt', 'hit_rate_ttp',
            'ptt_cum_return', 'ptt_excess_return', 'ptt_max_dd',
            'ttp_cum_return', 'ttp_excess_return', 'ttp_max_dd']


# ----
# Metrics
# ----

def _ratio(a: float, b: float) -> float:
    return a / b if b != 0 and np.isfinite(b) else np.nan


def _excess(cum: float, b_cum: float) -> float:
    '''Relative excess return: end value against the benchmark's end value.'''
    return (1 + cum) / (1 + b_cum) - 1


def _turnover_pa(run: dict, r: pd.Series, start, end) -> float:
    '''Turnover per year in (start, end], without the initial build (as compare_all).'''
    to = run['turnover']
    if to is None or len(to) < 2:
        return np.nan
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float(ca._slice(to, start, end).iloc[1:].sum() / years)


def full_metrics(r: pd.Series, b: pd.Series) -> dict:
    '''Metrics of the whole window; r, b: daily log returns of run and benchmark.'''
    if len(r) < 2:
        return {}
    lvl = ca._level(r)
    cum, b_cum = cumulative_return(r), cumulative_return(b) if len(b) else np.nan
    return {
        'full_cum_return':       cum,
        'full_excess_return':    _excess(cum, b_cum),
        'full_max_dd':           maximum_drawdown(lvl),
        'full_ulcer':            ulcer_index(lvl),
        'full_ann_vol':          annualized_volatility(r),
        'full_underwater_share': float((lvl < lvl.cummax()).iloc[1:].mean()),
        'full_recovered':        bool(lvl.iloc[-1] >= lvl.iloc[0]),
    }


def overall_metrics(run: dict, r: pd.Series, start, end) -> dict:
    '''Extra full-period metrics of the overall sheet.'''
    if len(r) < 2:
        return {}
    return {
        'full_cagr':        annualized_return(r),
        'full_sharpe':      sharpe_ratio(r),
        'full_sortino':     sortino_ratio(r),
        'full_calmar':      calmar_ratio(r, ca._level(r)),
        'full_turnover_pa': _turnover_pa(run, r, start, end),
    }


def phase_metrics(r: pd.Series, b: pd.Series, phase: str) -> dict:
    '''PTT or TTP metrics; capture = run cum return / benchmark cum return.'''
    if len(r) == 0:
        return {}
    cum, b_cum = cumulative_return(r), cumulative_return(b) if len(b) else np.nan
    capture = 'down_capture' if phase == 'ptt' else 'recovery_capture'
    return {
        f'{phase}_cum_return':    cum,
        f'{phase}_excess_return': _excess(cum, b_cum),
        f'{phase}_{capture}':     _ratio(cum, b_cum),
        f'{phase}_max_dd':        maximum_drawdown(ca._level(r)),
    }


def window_row(run: dict, bench: pd.Series, w: dict, phases: bool) -> dict:
    row = {k: run[k] for k in KEY_COLS}
    s = lambda x, start, end: ca._slice(x, start, end)
    r = s(run['returns'], w['peak'], w['even'])
    row.update(full_metrics(r, s(bench, w['peak'], w['even'])))
    if not phases:                                    # overall sheet
        row.update(overall_metrics(run, r, w['peak'], w['even']))
    else:
        row.update(phase_metrics(s(run['returns'], w['peak'], w['trough']),
                                 s(bench, w['peak'], w['trough']), 'ptt'))
        row.update(phase_metrics(s(run['returns'], w['trough'], w['even']),
                                 s(bench, w['trough'], w['even']), 'ttp'))
    return row


# ----
# Main
# ----

def add_hit_rates(sheets: dict, tests: dict) -> None:
    '''
    hit_rate_<phase> on the overall sheet: share of the crisis sheets in which
    the run beats the benchmark, test(row, bench_row) -> True / False, or NaN
    when not comparable (left out of the share); hit_count_<phase> = the same
    as "hits/comparable crises", e.g. 2/5.  Empty for the benchmark.
    '''
    crises = [df for name, df in sheets.items() if name != 'overall']
    ov = sheets['overall']
    is_bench = ov['model'] == BENCHMARK
    for phase, test in tests.items():
        hits = []
        for df in crises:
            b = df[df['model'] == BENCHMARK].iloc[0]
            h = df.apply(lambda row: test(row, b), axis=1)
            hits.append(pd.to_numeric(h, errors='coerce').set_axis(df['run']))
        h = pd.concat(hits, axis=1)
        count = h.sum(axis=1).astype(int).astype(str) + '/' + h.notna().sum(axis=1).astype(str)
        ov[f'hit_rate_{phase}'] = ov['run'].map(h.mean(axis=1)).where(~is_bench)
        ov[f'hit_count_{phase}'] = ov['run'].map(count).where(~is_bench)


# Beats the benchmark: higher return than the benchmark over the phase
HIT_TESTS = {
    'full': lambda r, b: r['full_excess_return'] > 0,
    'ptt':  lambda r, b: r['ptt_excess_return'] > 0,
    'ttp':  lambda r, b: r['ttp_excess_return'] > 0,
}


def _sort_key(run: dict):
    m = run['model']
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m, run['run'])


def _write(sheets: dict, path) -> None:
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = 'D2'
            for j, col in enumerate(df.columns, 1):
                letter = get_column_letter(j)
                ws.column_dimensions[letter].width = min(max(10, len(str(col)) + 2), 32)
                fmt = '0.00%' if col in PCT_COLS else '0.000' if df[col].dtype.kind == 'f' else None
                if fmt:
                    for cell in ws[letter][1:]:
                        cell.number_format = fmt


def main() -> None:
    runs = [r for r in ca.find_runs() if r['detector'] == 'none' and r['model'] not in SKIP_MODELS]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for freq in FREQ_ORDER:
        group = sorted((r for r in runs if r['frequency'] == freq), key=_sort_key)
        bench = next((r for r in group if r['model'] == BENCHMARK), None)
        if bench is None:
            print(f'[crisis_analysis] no {BENCHMARK} run for {freq}: skipped')
            continue
        windows = [('overall', overall[0], False)]
        windows += [(f'C{i}', c, True) for i, c in enumerate(main_crises, 1)]
        sheets = {name: pd.DataFrame([window_row(r, bench['returns'], w, phases) for r in group])
                  for name, w, phases in windows}
        add_hit_rates(sheets, HIT_TESTS)
        path = SAVE_DIR / f'crisis_analysis_{freq}.xlsx'
        _write(sheets, path)
        print(f'[crisis_analysis] {freq}: {len(group)} portfolios, benchmark '
              f'{bench["run"]}; saved to {path}')


if __name__ == '__main__':
    main()
