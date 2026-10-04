"""
Sanity check of the HMM regime files: crisis share in each crisis window from
_metrics/crises.py (peak -> trough) and in calm benchmark years, parameter
medians, and how often the label flips.  Run after main_hmm.py.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1] / "_metrics"))
from crises import main_crises, sub_crises

CALM = [("2004-01-01", "2006-12-31"), ("2013-01-01", "2014-12-31"), ("2017-01-01", "2017-12-31")]

for tag in ("vix", "gspc"):
    df = pd.read_csv(HERE / "regimes_final" / f"hmm_{tag}.csv", index_col="date", parse_dates=["date"])
    crisis = df["p_crisis"] > 0.5
    print(f"\n=== {tag}: crisis on {crisis.mean():.1%} of days, "
          f"{(crisis != crisis.shift()).sum() / (len(df) / 252):.1f} label flips per year")
    for c in main_crises + sub_crises:
        s = crisis.loc[c["peak"]:c["trough"]]
        print(f"  {c['label']:13s} {c['peak']} .. {c['trough']}  crisis {s.mean():6.1%}")
    for a, b in CALM:
        print(f"  calm          {a} .. {b}  crisis {crisis.loc[a:b].mean():6.1%}")
    cols = ["sigma_calm_ann", "sigma_crisis_ann", "mu_calm_ann", "mu_crisis_ann",
            "p_stay_calm", "p_stay_crisis"]
    print("  medians: " + ", ".join(f"{k} {df[k].median():.3f}" for k in cols))
