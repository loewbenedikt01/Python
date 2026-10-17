'''
Crisis performance of the baseline portfolios with portfolio-specific windows.

Same portfolios, files and sheets as crisis_analysis_defined.py (baseline runs
= no changepoint / hmm / wasserstein in the name; one workbook per frequency;
sheets overall, C1 ... C6 from crises.main_crises), but only the start of each
crisis is taken from crises.py.  From there every portfolio gets its own
episode:
  start     the crisis peak date of crises.py; portfolio value = 1 (anchor)
  trough    the portfolio's lowest value in (start, search end], search end =
            defined recovery date + TROUGH_BUFFER_MONTHS (at most the next
            crisis peak).  A trough on the last day of the search window is
            flagged (trough_at_boundary): read its depth as "at least".
  recovery  the first day after the trough with value >= 1 (back at the
            anchor), searched up to the next crisis peak (last crisis: end of
            the data); none = not recovered
  own window  start -> recovery (-> recovery search end if not recovered)
A portfolio whose lowest value stays above 1 - NO_DD_TOL has no drawdown: no
trough or recovery date, ptt_days = ttp_days = 0, own window = defined window.
Not recovered = censored: full_days, full_days_underwater and ttp_days are then
counted up to the recovery search end and mean "at least this many days".
Sheet overall: anchoring at the 1998 start would only catch an early dip, so
the start is the portfolio's own high-water mark before its deepest drawdown
of the full period, searched to the end of the data (ptt_days_vs_defined left
empty).

Output: _output/_comparison/crisis_analysis_variable_<frequency>.xlsx
Rows: one per portfolio (model, run, frequency).  Days = trading days.

Columns (bench = BENCHMARK run of the same frequency)
  peak_date               start (the defined crisis peak; overall: own peak)
  trough_date, recovery_date   the portfolio's own dates
  full_days               length of the own window
  full_anchored_dd        trough value / anchor - 1
  full_hwm_dd             max drawdown from the running high-water mark
                          (can be deeper than anchored if the portfolio rose first)
  full_days_underwater    days below the anchor (start value) in the own window
  full_ulcer              Ulcer index over the own window, drawdowns from the
                          running high (standard definition, metrics.ulcer_index)
  full_no_drawdown        True if the lowest value stays above 1 - NO_DD_TOL
  full_not_recovered      True if not back at the anchor by the recovery
                          search end (day counts censored)
  ptt_days                start -> own trough
  ptt_days_vs_defined     ptt_days - days from start to the defined trough
                          (negative = bottomed earlier)
  ptt_depth               anchored drawdown at the trough (= full_anchored_dd)
  ptt_max_dd              max drawdown (high-water mark) start -> trough
  ptt_trough_at_boundary  True if the trough is the last day of the search window
  ttp_days                own trough -> own recovery (censored if not recovered)
  ttp_days_vs_bench       ttp_days - bench ttp_days (negative = recovered
                          faster); empty if either has not recovered
  ttp_vs_bench_censored   True if ttp_days_vs_bench is empty for that reason
  ttp_return_to_even      return from the own trough back to the anchor,
                          1 / trough value - 1
  hit_rate_full, hit_rate_ptt, hit_rate_ttp
                          overall sheet only: share of the crises C1 ... C6 in
                          which the portfolio beats the benchmark (empty for
                          the benchmark).  full: back at the start value in
                          fewer days; ptt: shallower ptt_depth; ttp: fewer
                          ttp_days.  Recovered beats not recovered; both not
                          recovered = crisis left out of the share.

Usage: run the file.
'''

import numpy as np
import pandas as pd

import compare_all as ca
from crises import overall, main_crises
from metrics import maximum_drawdown, ulcer_index
from crisis_analysis_defined import (
    BENCHMARK, FREQ_ORDER, SAVE_DIR, SKIP_MODELS, KEY_COLS, _sort_key, add_hit_rates,
)


# ----
# Variables
# ----

TROUGH_BUFFER_MONTHS = 6       # trough search past the defined recovery date
NO_DD_TOL            = 0.01    # lowest value above 1 - NO_DD_TOL = no drawdown

PCT_COLS  = ['full_anchored_dd', 'full_hwm_dd', 'full_ulcer', 'ptt_depth', 'ptt_max_dd',
             'ttp_return_to_even', 'hit_rate_full', 'hit_rate_ptt', 'hit_rate_ttp']
DATE_COLS = ['peak_date', 'trough_date', 'recovery_date']


# ----
# Episode
# ----

def _with_anchor(lvl: pd.Series) -> pd.Series:
    '''Level series with the anchor value 1 prepended (for drawdowns from the start).'''
    return pd.concat([pd.Series([1.0]), lvl.reset_index(drop=True)], ignore_index=True)


def episode(returns: pd.Series, c: dict, rec_end=None, own_peak: bool = False) -> dict:
    '''
    The portfolio's own crisis episode from the defined start c['peak'].
    rec_end: last day of the recovery search (the next crisis peak; None = end
    of the data).  own_peak (overall sheet): start at the high-water mark
    before the portfolio's deepest drawdown in the window instead.
    '''
    r = returns[returns.index > pd.Timestamp(c['peak'])]
    lvl = np.exp(r.cumsum())
    if rec_end is not None:
        lvl = lvl[lvl.index <= pd.Timestamp(rec_end)]
    peak_date = pd.Timestamp(c['peak'])
    even = pd.Timestamp(c['even'])
    if own_peak:
        win = lvl[lvl.index <= even]
        t = (win / win.cummax()).idxmin()
        peak_date = win.loc[:t].idxmax()
        lvl = lvl[lvl.index > peak_date] / lvl.loc[peak_date]
        trough_end = even
    else:
        trough_end = even + pd.DateOffset(months=TROUGH_BUFFER_MONTHS)
    win = lvl[lvl.index <= trough_end]
    if len(win) < 2:
        return {}
    defined_ptt = np.nan if own_peak else int((lvl.index <= pd.Timestamp(c['trough'])).sum())
    t_date, t_val = win.idxmin(), float(win.min())

    if t_val >= 1.0 - NO_DD_TOL:                      # no (meaningful) drawdown
        own = lvl[lvl.index <= even]
        ptt_days, ttp_days, t_date, rec_date = 0, 0, None, None
        no_dd, not_rec, at_boundary = True, False, False
    else:
        at_boundary = bool(t_date == win.index[-1])
        ptt_days = lvl.index.get_loc(t_date) + 1
        after = lvl[lvl.index > t_date]
        back = after[after >= 1.0]
        no_dd, not_rec = False, back.empty
        rec_date = None if not_rec else back.index[0]
        own = lvl if not_rec else lvl[lvl.index <= rec_date]
        ttp_days = len(own) - ptt_days                # lower bound if not recovered

    own_lvl = _with_anchor(own)
    return {
        'peak_date':              peak_date,
        'trough_date':            t_date,
        'recovery_date':          rec_date,
        'full_days':              len(own),
        'full_anchored_dd':       t_val - 1.0,
        'full_hwm_dd':            maximum_drawdown(own_lvl),
        'full_days_underwater':   int((own < 1.0).sum()),
        'full_ulcer':             ulcer_index(own_lvl),
        'full_no_drawdown':       no_dd,
        'full_not_recovered':     not_rec,
        'ptt_days':               ptt_days,
        'ptt_days_vs_defined':    ptt_days - defined_ptt,
        'ptt_depth':              t_val - 1.0,
        'ptt_max_dd':             maximum_drawdown(_with_anchor(lvl.iloc[:ptt_days])),
        'ptt_trough_at_boundary': at_boundary,
        'ttp_days':               ttp_days,
        'ttp_return_to_even':     0.0 if no_dd else 1.0 / t_val - 1.0,
    }


def _faster(r, b, days: str) -> float:
    '''
    Back at the start value sooner than the benchmark (censored = not
    recovered by the search end).  Recovered vs not recovered decides; both
    not recovered: not comparable (NaN).
    '''
    r_cens, b_cens = bool(r['full_not_recovered']), bool(b['full_not_recovered'])
    if r_cens and b_cens:
        return np.nan
    if r_cens != b_cens:
        return float(b_cens)
    return float(r[days] < b[days])


# Beats the benchmark: full = back at the start value in fewer days,
# ptt = shallower drop to the own trough, ttp = faster recovery from the trough
HIT_TESTS = {
    'full': lambda r, b: _faster(r, b, 'full_days'),
    'ptt':  lambda r, b: float(r['ptt_depth'] > b['ptt_depth']),
    'ttp':  lambda r, b: _faster(r, b, 'ttp_days'),
}


def vs_bench(row: dict, b: dict) -> None:
    '''ttp_days_vs_bench, empty (and flagged) if either has not recovered.'''
    if 'ttp_days' not in row or 'ttp_days' not in b:
        return
    censored = row['full_not_recovered'] or b['full_not_recovered']
    row['ttp_days_vs_bench'] = np.nan if censored else row['ttp_days'] - b['ttp_days']
    row['ttp_vs_bench_censored'] = censored


# ----
# Main
# ----

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
                fmt = ('0.00%' if col in PCT_COLS else 'yyyy-mm-dd' if col in DATE_COLS
                       else '0.000' if df[col].dtype.kind == 'f' and 'days' not in col else None)
                if fmt:
                    for cell in ws[letter][1:]:
                        cell.number_format = fmt


def main() -> None:
    runs = [r for r in ca.find_runs() if r['detector'] == 'none' and r['model'] not in SKIP_MODELS]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # (sheet, crisis, recovery search end, own peak)
    windows = [('overall', overall[0], None, True)]
    for i, c in enumerate(main_crises):
        nxt = main_crises[i + 1]['peak'] if i + 1 < len(main_crises) else None
        windows.append((f'C{i + 1}', c, nxt, False))
    for freq in FREQ_ORDER:
        group = sorted((r for r in runs if r['frequency'] == freq), key=_sort_key)
        bench = next((r for r in group if r['model'] == BENCHMARK), None)
        if bench is None:
            print(f'[crisis_analysis_variable] no {BENCHMARK} run for {freq}: skipped')
            continue
        sheets = {}
        for name, c, rec_end, own_peak in windows:
            b = episode(bench['returns'], c, rec_end, own_peak)
            rows = []
            for run in group:
                row = {**{k: run[k] for k in KEY_COLS}, **episode(run['returns'], c, rec_end, own_peak)}
                vs_bench(row, b)
                rows.append(row)
            df = pd.DataFrame(rows)
            for col in ('ttp_vs_bench_censored', 'ttp_days_vs_bench'):   # next to ttp_days
                if col in df:
                    df.insert(df.columns.get_loc('ttp_days') + 1, col, df.pop(col))
            sheets[name] = df
        add_hit_rates(sheets, HIT_TESTS)
        path = SAVE_DIR / f'crisis_analysis_variable_{freq}.xlsx'
        _write(sheets, path)
        print(f'[crisis_analysis_variable] {freq}: {len(group)} portfolios, benchmark '
              f'{bench["run"]}; saved to {path}')


if __name__ == '__main__':
    main()
