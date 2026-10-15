'''
Z-scores and Borda count of every run in _output, one sheet per metric.

Columns: one per run (model, run, frequency), as found by compare_all.py.
Rows (the same in every block):
  Overall                          the full period (crises.overall)
  C1 <name> peak -> trough         crises.main_crises, (peak, trough]
  C1 <name> trough -> recovery     (trough, recovery]
  C2 ...                           and so on for every main crisis

Three blocks per sheet, split by a thick line:
  value    the metric as a plain decimal (max_dd -0.49, not -49%).  calmar
           of a run without any drawdown in the period (max_dd = 0, e.g.
           all in cash) and a positive return is 'inf': it ranks best
  z        z-score within the row: (value - row mean) / row std, over all
           runs with a finite value in that period
  borda    Borda points within the row: a run gets one point for every run
           it beats on that metric (best = n - 1, worst = 0, ties share the
           average).  Ranked in the metric's better direction (METRICS
           below), so for ann_vol or ulcer the lowest value wins; the same
           order as the z-scores (z is monotone within a row), but 'inf'
           calmar values are ranked too.  A period where the metric is
           undefined for every run (down_capture when the benchmark has no
           down month) stays empty.
  rank     below a thick line: 'Overall rank', the rank of every run in the
           Overall row (1 = best; the order of the Overall z-scores)
  sens     sensitivity analysis: the crisis rows of the Borda block combined
           into one composite crisis score per weighting scheme (SEVERITY
           below) and its rank.  Score of crisis c = mean of its peak ->
           trough and trough -> recovery Borda points (only one of the two
           if the other is empty); composite = sum_c w_c * score_c,
           w_c = s_c / sum_j s_j (s_c = severity of c), with the weights
           rescaled to sum to 1 over the crises that have a score.
           Three variants, one after the other, each with a score and a
           rank row per scheme:
             '<scheme> score'      peak -> trough and trough -> recovery
             'PTT <scheme> score'  peak -> trough rows only
             'TTP <scheme> score'  trough -> recovery rows only
Sheet borda_all: the Overall-period Borda points of every metric (rows) per
run, their sum over all metrics ('Overall total') and its rank ('Overall
rank', 1 = best); below it the composite crisis scores of all three variants
summed over all metrics, per weighting scheme, and their ranks.
Sheet weights: the severity metrics and the resulting crisis weights.

Metrics are computed with compare_all.window_metrics, so they match
compare_all.xlsx; beta and down_capture are against compare_all.BENCH_MODEL.

Output (needs openpyxl), in _output/_comparison/ (WORKBOOKS below):
  zscores.xlsx             all runs ranked together
  zscores_monthly.xlsx     only the monthly runs ranked against each other,
  zscores_quarterly.xlsx   likewise quarterly and yearly, so every run is
  zscores_yearly.xlsx      compared within its rebalancing frequency
  z_scores_monthly_base.xlsx    only the base runs (no regime switch: the
  z_scores_quarterly_base.xlsx  models' baselines and the benchmarks) of
  z_scores_yearly_base.xlsx     one frequency against each other
Runs without a rebalancing frequency (sp500) are in every workbook.

Usage: set the variables below and run the file.
'''

import numpy as np
import pandas as pd

import compare_all as ca
from compare_runs import CRISIS_NAMES
from crises import overall, main_crises

# ----
# Variables
# ----

# metric -> better direction: +1 higher is better, -1 lower is better,
# 0 = values and z-scores only, no Borda count
METRICS = {
    'sharpe':       +1,
    'sortino':      +1,
    'calmar':       +1,
    'ann_return':   +1,
    'cum_return':   +1,
    'ann_vol':      -1,
    'max_dd':       +1,      # negative numbers: -0.20 beats -0.50
    'dd_days':      -1,
    'ulcer':        -1,
    'var_95':       +1,      # negative numbers: a smaller loss is better
    'cvar_95':      +1,
    'beta':         -1,      # less market exposure counts as better
    'down_capture': -1,
}

INCLUDE_MODELS = None        # e.g. ['mvo', 'hrp', 'market_cap']; None = all model folders

# Sensitivity analysis: severity s_c of each main crisis, C1 ... C6 in the
# order of crises.main_crises; weights w_c = s_c / sum_j s_j
SEVERITY = {
    'Equal':    [1, 1, 1, 1, 1, 1],
    'Drawdown': [49.14, 56.78, 19.78, 33.92, 25.43, 18.90],     # |max drawdown| in %
    'Duration': [930, 517, 94, 33, 282, 48],                    # calendar days peak -> trough
    'VIX':      [25.28, 32.68, 19.78, 46.42, 26.03, 23.54],     # mean VIX during the crisis
}

# file name -> (frequencies, group) ranked together in it.
# frequencies: list, None = all.  group: None = all runs; 'base' = runs without
# a regime switch (model baselines and benchmarks; not the always-min-var
# control); 'cash' / 'hrp' / 'min_var' = the regime runs with that crisis target
WORKBOOKS = {
    'zscores.xlsx':                 (None, None),
    'zscores_monthly.xlsx':         (['monthly'], None),
    'zscores_quarterly.xlsx':       (['quarterly'], None),
    'zscores_yearly.xlsx':          (['yearly'], None),
    'z_scores_monthly_base.xlsx':   (['monthly'], 'base'),
    'z_scores_quarterly_base.xlsx': (['quarterly'], 'base'),
    'z_scores_yearly_base.xlsx':    (['yearly'], 'base'),
}

FMT = {'value': '0.0000', 'z': '0.00', 'borda': '0.0', 'rank': '0', 'sens': '0.00'}


# ----
# Data
# ----

def periods() -> list[tuple[str, str, str]]:
    '''(row label, start, end) of every row, in order; returns in (start, end].'''
    rows = [('Overall', overall[0]['peak'], overall[0]['even'])]
    for i, c in enumerate(main_crises, 1):
        name = f'C{i} ' + CRISIS_NAMES.get(c['label'], c['label'])
        rows.append((name + ' peak -> trough', c['peak'], c['trough']))
        rows.append((name + ' trough -> recovery', c['trough'], c['even']))
    return rows


def select_runs() -> tuple[list[dict], dict]:
    '''
    (runs, benchmarks).  Runs are those of compare_all, filtered and sorted,
    each with a column name.  A folder without frequency (sp500) is listed by
    compare_all once per frequency; here it is kept once, named without
    frequency and marked no_freq.
    '''
    all_runs = ca.find_runs()
    benches = ca._benchmarks(all_runs)
    n_shared = pd.Series([id(r['returns']) for r in all_runs]).value_counts()
    runs, seen = [], set()
    for r in sorted(all_runs, key=ca._sort_key):
        if INCLUDE_MODELS is not None and r['model'] not in INCLUDE_MODELS:
            continue
        if id(r['returns']) in seen:
            continue
        seen.add(id(r['returns']))
        r['no_freq'] = bool(n_shared[id(r['returns'])] > 1)
        r['column'] = r['run'] if r['no_freq'] else r['run'] + '_' + r['frequency']
        runs.append(r)
    return runs, benches


def value_tables(runs: list[dict], benches: dict) -> dict[str, pd.DataFrame]:
    '''metric -> period x run table of values.'''
    rows = {m: [] for m in METRICS}
    for label, start, end in periods():
        vals = {}
        for r in runs:
            b = benches.get(r['frequency'])
            v = ca.window_metrics(r, start, end, b['returns'] if b is not None else None)
            if v.get('max_dd') == 0 and v.get('ann_return', 0) > 0:
                v['calmar'] = np.inf                      # no drawdown at all
            vals[r['column']] = v
        for m in METRICS:
            rows[m].append(pd.Series({c: v.get(m, np.nan) for c, v in vals.items()}, name=label))
    return {m: pd.DataFrame(r).astype(float) for m, r in rows.items()}


def zscores(values: pd.DataFrame) -> pd.DataFrame:
    '''z-score within every row over the runs with a value.'''
    values = values.replace([np.inf, -np.inf], np.nan)
    mu = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    return values.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def borda(values: pd.DataFrame, direction: int) -> pd.DataFrame:
    '''
    Borda points within every row: number of runs beaten, ties averaged
    (rank - 1 with average ranks, best = n - 1).
    '''
    return (direction * values).rank(axis=1, method='average') - 1.0


def weights() -> pd.DataFrame:
    '''scheme x crisis weights, w_c = s_c / sum_j s_j (rows sum to 1).'''
    n = len(main_crises)
    cols = [f'C{i}' for i in range(1, n + 1)]
    for name, s in SEVERITY.items():
        if len(s) != n:
            raise ValueError(f'[zscores] SEVERITY {name!r} has {len(s)} values, '
                             f'crises.main_crises has {n}')
    sev = pd.DataFrame(SEVERITY, index=cols).T.astype(float).abs()
    return sev.div(sev.sum(axis=1), axis=0)


# sensitivity variants: row-label prefix -> which crisis rows (offsets 1 =
# peak -> trough, 2 = trough -> recovery in the period list of each crisis)
VARIANTS = {'': (1, 2), 'PTT ': (1,), 'TTP ': (2,)}


def sensitivity(pts: pd.DataFrame) -> pd.DataFrame:
    '''
    Composite crisis score per variant and weighting scheme from the Borda
    points of the crisis rows (pts: period x run): per variant first the
    score rows of all schemes, then their rank rows.
    '''
    labels = [p[0] for p in periods()]
    w = weights()
    scores = {}
    for prefix, offsets in VARIANTS.items():
        crisis = pd.DataFrame([pts.loc[[labels[2 * i + k] for k in offsets]].mean(axis=0)
                               for i in range(len(main_crises))], index=w.columns)
        has = crisis.notna()
        for name, wc in w.iterrows():
            wsum = has.mul(wc, axis=0).sum(axis=0)              # weight of the crises with a score
            score = crisis.mul(wc, axis=0).sum(axis=0, min_count=1) / wsum
            scores[prefix + name + ' score'] = score.where(wsum > 0)
    return _ranks(pd.DataFrame(scores).T)


def _ranks(scores: pd.DataFrame) -> pd.DataFrame:
    '''
    Rank row (1 = best) for every score row; per variant the score rows of
    all schemes first, then their rank rows.
    '''
    rows = []
    for prefix in VARIANTS:
        sc = [l for l in scores.index if l.startswith(prefix) and
              not any(p and l.startswith(p) for p in VARIANTS if p != prefix)]
        rows += [scores.loc[l] for l in sc]
        rows += [scores.loc[l].rank(ascending=False, method='min').rename(l.replace(' score', ' rank'))
                 for l in sc]
    return pd.DataFrame(rows)


# ----
# Writing
# ----

def _write_block(ws, df: pd.DataFrame, block: str, row0: int, thick_top: bool) -> int:
    '''Write one block from row row0 on; returns the next free row.'''
    from openpyxl.styles import Border, Font, Side
    thick = Side(style='thick')
    for i, (label, vals) in enumerate(df.iterrows()):
        r = row0 + i
        cells = [ws.cell(row=r, column=1, value=block), ws.cell(row=r, column=2, value=label)]
        for j, v in enumerate(vals.to_numpy(), 3):
            v = None if pd.isna(v) else 'inf' if np.isinf(v) else float(v)
            c = ws.cell(row=r, column=j, value=v)
            c.number_format = '0' if label.endswith('rank') else FMT[block]
            cells.append(c)
        if label.endswith(('rank', 'total')):
            for c in cells:
                c.font = Font(bold=True)
        if thick_top and i == 0:
            for c in cells:
                c.border = Border(top=thick)
    return row0 + len(df)


def write(tables: dict, summary: pd.DataFrame, sens_all: pd.DataFrame | None, path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    cols = list(next(iter(tables.values()))['value'].columns)
    for name in [*tables, 'borda_all']:
        if name == 'borda_all' and summary.empty:
            continue
        ws = wb.create_sheet(name[:31])
        ws.cell(row=1, column=1, value='block').font = Font(bold=True)
        ws.cell(row=1, column=2, value='period' if name != 'borda_all' else 'metric').font = Font(bold=True)
        for j, col in enumerate(cols, 3):
            c = ws.cell(row=1, column=j, value=col)
            c.font = Font(bold=True)
            c.alignment = Alignment(wrap_text=True, vertical='top')
        ws.row_dimensions[1].height = 75
        ws.column_dimensions['A'].width = 8
        ws.column_dimensions['B'].width = 30
        for j in range(3, len(cols) + 3):
            ws.column_dimensions[get_column_letter(j)].width = 14
        ws.freeze_panes = 'C2'
        if name == 'borda_all':
            row = _write_block(ws, summary, 'borda', 2, False)
            if sens_all is not None:
                _write_block(ws, sens_all, 'sens', row, True)
            continue
        row = 2
        for k, block in enumerate(('value', 'z', 'borda', 'rank', 'sens')):
            df = tables[name].get(block)
            if df is None:
                continue
            row = _write_block(ws, df, block, row, thick_top=k > 0)

    ws = wb.create_sheet('weights')
    sev = pd.DataFrame(SEVERITY, index=weights().columns).T
    table = pd.concat([sev.rename(index=lambda s: s + ' severity'),
                       weights().rename(index=lambda s: s + ' weight')])
    ws.cell(row=1, column=1, value='scheme').font = Font(bold=True)
    for j, col in enumerate(table.columns, 2):
        ws.cell(row=1, column=j, value=col).font = Font(bold=True)
    for i, (label, vals) in enumerate(table.iterrows(), 2):
        ws.cell(row=i, column=1, value=label)
        for j, v in enumerate(vals.to_numpy(), 2):
            ws.cell(row=i, column=j, value=float(v)).number_format = (
                '0.00%' if label.endswith('weight') else '0.00')
    ws.column_dimensions['A'].width = 22
    wb.save(path)


def _in_group(run: dict, group: str | None) -> bool:
    if group is None:
        return True
    if group == 'base':
        return run['detector'] == 'none' and run['crisis_target'] == ''
    return run['detector'] != 'none' and run['crisis_target'] == group


def build(values: dict[str, pd.DataFrame], path) -> None:
    '''z-scores, Borda count and sensitivity analysis of the runs in values
    (the columns), written to path.'''
    tables, totals, sens_sum = {}, {}, None
    for m, direction in METRICS.items():
        z = zscores(values[m])
        tables[m] = {'value': values[m], 'z': z}
        if direction:
            b = borda(values[m], direction)
            sens = sensitivity(b)
            tables[m]['borda'] = b
            tables[m]['rank'] = b.loc[['Overall']].rank(axis=1, ascending=False, method='min') \
                .rename(index={'Overall': 'Overall rank'})
            tables[m]['sens'] = sens
            totals[m] = b.loc['Overall']
            scores = sens.loc[[i for i in sens.index if i.endswith('score')]]
            sens_sum = scores if sens_sum is None else sens_sum.add(scores)
    summary = pd.DataFrame(totals).T
    total = summary.sum(axis=0, min_count=1)
    summary.loc['Overall total'] = total
    summary.loc['Overall rank'] = total.rank(ascending=False, method='min')
    write(tables, summary, _ranks(sens_sum) if sens_sum is not None else None, path)
    best = summary.loc['Overall rank'].sort_values().head(3)
    print(f'[zscores] {path.name}: {summary.shape[1]} runs, top 3: '
          + ', '.join(f'{int(v)}. {k}' for k, v in best.items()))


def main() -> None:
    runs, benches = select_runs()
    if not runs:
        print('[zscores] no runs found')
        return
    print(f'[zscores] {len(runs)} runs, {len(periods())} periods, {len(METRICS)} metrics')
    values = value_tables(runs, benches)          # every run is measured once
    ca.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for name, (freqs, group) in WORKBOOKS.items():
        cols = [r['column'] for r in runs
                if (freqs is None or r['no_freq'] or r['frequency'] in freqs)
                and _in_group(r, group)]
        if len(cols) < 2:
            print(f'[zscores] {name}: fewer than 2 runs, skipped')
            continue
        build({m: v[cols] for m, v in values.items()}, ca.SAVE_DIR / name)
    print(f'[zscores] saved to {ca.SAVE_DIR}')

if __name__ == '__main__':
    main()
