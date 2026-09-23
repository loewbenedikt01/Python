"""
Turn the R/cpm changepoint CSV into the common regime interface.

The CSV gives, per detection:
    changepoint_date   tau-hat   -- where the break actually happened
    detection_date     t         -- the day the alarm fired (t > tau-hat)

At t you know tau-hat, so re-seeding the volatility estimate from tau-hat..t is
point-in-time valid; acting before t is not.  Breaks are detected on VIX
log-changes; the volatility that feeds the 20% threshold is estimated on equity
returns (a 20% annualised target is an equity number, not a VIX-change number).

Output: regimes/changepoint.csv, daily, shifted one day so the value at date
d uses only data through d-1.
"""

import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

# ---- knobs -------------------------------------------------------------------
LAMBDA      = 0.95      # EWMA forgetting factor (~20-day memory), from the paper
VOL_TARGET  = 0.20      # annualised vol at which p_crisis = 0.5, fixed a priori
SCALE       = 0.05      # transition width; sweep 0.03 / 0.05 / 0.08 as robustness
SERIES      = "vix"     # "vix" (primary, median delay 24d) | "gspc" (robustness)


def build(cp_csv: str, mkt_logret: pd.Series, out_path: str) -> pd.DataFrame:
    cp = pd.read_csv(cp_csv, parse_dates=["changepoint_date", "detection_date"])

    # map each detection day -> its estimated changepoint.  Snap both to actual
    # trading days so the lookups below cannot silently miss.
    idx = mkt_logret.index
    det = {}
    for _, row in cp.iterrows():
        t   = idx[idx.searchsorted(row.detection_date)]
        tau = idx[idx.searchsorted(row.changepoint_date)]
        det[t] = min(tau, t)

    sig2, rows = np.nan, []
    for t, r in mkt_logret.items():
        if t in det:
            # alarm fired: discard pre-break history, re-seed from tau..t
            seg = mkt_logret.loc[det[t]:t]
            s = float(seg.iloc[0] ** 2)
            for x in seg.iloc[1:]:
                s = LAMBDA * s + (1 - LAMBDA) * x ** 2
            sig2 = s
        elif np.isnan(sig2):
            sig2 = float(r ** 2)
        else:
            sig2 = LAMBDA * sig2 + (1 - LAMBDA) * float(r) ** 2
        ann = float(np.sqrt(sig2 * 252))
        rows.append((t, ann, float(norm.cdf((ann - VOL_TARGET) / SCALE))))

    df = pd.DataFrame(rows, columns=["date", "sigma_ann", "p_crisis"]).set_index("date")
    df["p_calm"] = 1.0 - df.p_crisis
    df["label"]  = np.where(df.p_crisis > 0.5, "crisis", "calm")

    last_det = pd.Series(list(det.keys()), index=list(det.keys())).reindex(df.index).ffill()
    df["days_since_break"] = (df.index.to_series() - last_det).dt.days
    df["refit_date"] = last_det

    # one-day execution delay: value at d uses only data through d-1
    out = df.shift(1)
    out["label"] = df["label"].shift(1)
    out = out.dropna(subset=["p_crisis"])
    out = out[["p_calm", "p_crisis", "label", "sigma_ann", "days_since_break", "refit_date"]]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    return out


def validate(df: pd.DataFrame) -> None:
    print(f"rows {len(df)}   {df.index.min().date()} .. {df.index.max().date()}")
    print(f"p_crisis > 0.5 on {(df.p_crisis > 0.5).mean():.1%} of days "
          f"(want roughly 10-40%)")
    print(f"mean p_crisis {df.p_crisis.mean():.3f}   mean sigma_ann {df.sigma_ann.mean():.3f}\n")
    print("crisis checkpoints (all should be elevated; 2020-03 near 1.0):")
    for d in ["2000-03-31", "2002-09-30", "2008-10-31", "2011-08-31",
              "2020-03-31", "2022-06-30"]:
        if pd.Timestamp(d) >= df.index.min():
            print(f"  {d}  p_crisis={df.p_crisis.asof(d):.3f}  vol={df.sigma_ann.asof(d):.3f}")
    print("\ncalm checkpoints (all should be low):")
    for d in ["2005-06-30", "2013-06-28", "2017-06-30", "2019-09-30"]:
        if pd.Timestamp(d) >= df.index.min():
            print(f"  {d}  p_crisis={df.p_crisis.asof(d):.3f}  vol={df.sigma_ann.asof(d):.3f}")
    print("\nshare of days in crisis, by year:")
    yr = (df.p_crisis > 0.5).groupby(df.index.year).mean()
    print("  " + "  ".join(f"{y}:{v:.0%}" for y, v in yr.items()))


_CRISIS_CACHE: dict[str, pd.DataFrame] = {}


def crisis_probs(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Point-in-time p_calm/p_crisis for arbitrary dates, read from the precomputed
    regimes/{changepoint,changepoint_gspc}.csv written by build() (SERIES picks
    which).  Values are forward-filled to the nearest prior date already in that
    file, so a request for a non-trading day still resolves.
    """
    if SERIES not in _CRISIS_CACHE:
        fname = "changepoint_vix.csv" if SERIES == "vix" else "changepoint_gspc.csv"
        path = Path(__file__).resolve().parent / "regimes" / fname
        df = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
        _CRISIS_CACHE[SERIES] = df[["p_calm", "p_crisis"]]
    src = _CRISIS_CACHE[SERIES]

    idx = pd.DatetimeIndex(idx)
    out = src.reindex(idx, method="ffill")
    missing = out.index[out["p_crisis"].isna()]
    if len(missing):
        raise ValueError(
            f"crisis_probs: {len(missing)} requested date(s) before the first "
            f"available regime date {src.index.min().date()} (earliest requested "
            f"{missing.min().date()}) -- rerun build() over a wider range."
        )
    return out


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    sys.path.append(str(HERE.parents[1] / "_metrics"))
    from features import load_db, MKT_TICKER
    db  = load_db()
    mkt = np.log(db[("Adj Close", MKT_TICKER)]).diff().dropna()

    print("=== VIX (primary) ===")
    out = build(str(HERE / "vix_changepoint.csv"), mkt, str(HERE / "regimes/changepoint_vix.csv"))
    validate(out)

    print("\n=== GSPC (robustness) ===")
    out_gspc = build(str(HERE / "gspc_changepoint.csv"), mkt, str(HERE / "regimes/changepoint_gspc.csv"))
    validate(out_gspc)