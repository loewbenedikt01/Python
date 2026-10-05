'''
S&P 500 standard benchmark.

Pure buy & hold of ^GSPC (adjusted close) from START_DATE to END_DATE — no
universe, no rebalancing, one position held the whole way through.  Reference
line only; not one of the compared strategies.
'''

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / '_metrics'))

import export
from config import START_DATE, END_DATE
from portfolio import load_prices

# ---- 
# Variables
# ----

MODEL_NAME = 's&p_500'
TICKER     = '^GSPC'

# ---- 
# Main Part
# ----

def main() -> None:
    px = load_prices()[TICKER].dropna().loc[START_DATE:END_DATE]
    log_returns = np.log(px / px.shift(1)).dropna().rename('portfolio')

    weights  = pd.DataFrame(1.0, index=log_returns.index, columns=[TICKER])
    turnover = pd.Series({log_returns.index[0]: 1.0}, name='turnover')

    export.build_report(
        f'sp500/{MODEL_NAME}',
        log_returns,
        weights=weights,
        diagnostics={
            'turnover': turnover,
            'transaction_costs': turnover * 0.0,
        },
    )

    print(f'{MODEL_NAME}: {len(log_returns)} days '
          f'{log_returns.index[0].date()}..{log_returns.index[-1].date()}  '
          f'cum {np.expm1(log_returns.sum()):.1%}')


if __name__ == '__main__':
    main()
