from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

for t in ("gspc_vix", "gspc"):
    a = pd.read_csv(HERE / "regimes_final_5" / f"wasserstein_{t}.csv", index_col="date")["p_crisis"]   # step 5
    b = pd.read_csv(HERE / "regimes_final" / f"wasserstein_{t}.csv", index_col="date")["p_crisis"]     # step 1
    print(t, "label agreement", ((a > .5) == (b > .5)).mean().round(3), "corr", a.corr(b).round(3))
