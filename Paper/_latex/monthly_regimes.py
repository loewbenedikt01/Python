"""Reduce daily regime CSVs to the first trading day of each month for the paper charts."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FILES = ["regime_hmm_gspc.csv", "regime_wasserstein_gspc_vix.csv"]

for name in FILES:
    df = pd.read_csv(HERE / name, parse_dates=["date"])
    monthly = df.groupby(df["date"].dt.to_period("M")).head(1)
    out = HERE / name.replace(".csv", "_monthly.csv")
    monthly[["date", "p_crisis"]].to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"{out.name}: {len(monthly)} rows")
