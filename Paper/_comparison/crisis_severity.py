'''
Severity of every main crisis (crises.main_crises), the input of the crisis
weights in z_scores.py (w_c = s_c / sum_j s_j).

Per crisis, from the adjusted closes of ^GSPC and ^VIX (portfolio.load_prices):
  Drawdown   |max drawdown| of the S&P 500 from the peak, in %
  Duration   calendar days peak -> trough
  VIX        mean VIX close from the peak to the trough (both included)

Usage: run the file to print the values and weights; z_scores.py calls
severity() directly, so the weights follow crises.py.
'''

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / '_metrics'))
from crises import main_crises
from portfolio import load_prices


@lru_cache(maxsize=None)
def severity_table() -> pd.DataFrame:
    '''crisis (C1 ...) x severity metric.'''
    px = load_prices()
    gspc, vix = px['^GSPC'].dropna(), px['^VIX'].dropna()
    rows = {}
    for i, c in enumerate(main_crises, 1):
        peak, trough = pd.Timestamp(c['peak']), pd.Timestamp(c['trough'])
        sp = gspc.loc[peak:trough]
        rows[f'C{i}'] = {
            'label':    c['label'],
            'Drawdown': round(100 * (1 - sp.min() / sp.iloc[0]), 2),
            'Duration': (trough - peak).days,
            'VIX':      round(vix.loc[peak:trough].mean(), 2),
        }
    return pd.DataFrame(rows).T


def severity() -> dict[str, list[float]]:
    '''scheme -> severity per crisis in the order of crises.main_crises.'''
    t = severity_table()
    return {'Equal': [1.0] * len(t),
            **{m: t[m].astype(float).tolist() for m in ('Drawdown', 'Duration', 'VIX')}}


if __name__ == '__main__':
    print(severity_table().to_string())
    sev = pd.DataFrame(severity(), index=severity_table().index).T
    print('\nweights w_c = s_c / sum_j s_j')
    print(sev.div(sev.sum(axis=1), axis=0).round(4).to_string())
