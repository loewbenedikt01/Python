"""Copy the defensive holding periods of the LSTM regime runs for the paper NAV charts.

raw/regime.csv has one row per decision date; the holding set there is kept
until the next decision date.  Output: date, defensive (1 = crisis target held),
plus a closing row at the last month end so the final holding month is drawn.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUTPUT = HERE.parent / "_output" / "lstm"
RUNS = [
    "lstm_t_10_holdout_rank_changepoint_vix_cash_monthly",
    "lstm_t_10_holdout_rank_hmm_gspc_cash_monthly",
    "lstm_t_10_holdout_rank_wasserstein_gspc_vix_cash_monthly",
]

for run in RUNS:
    df = pd.read_csv(OUTPUT / run / "raw" / "regime.csv", parse_dates=["date"])
    out = pd.DataFrame({"date": df["date"], "defensive": (df["holding"] != "model").astype(int)})
    end = out["date"].iloc[-1] + pd.offsets.MonthEnd(0)
    out = pd.concat([out, pd.DataFrame({"date": [end], "defensive": [out["defensive"].iloc[-1]]})])
    path = HERE / f"{run}_defensive.csv"
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"{path.name}: {len(out)} rows, defensive in {out['defensive'].iloc[:-1].sum()} months")
