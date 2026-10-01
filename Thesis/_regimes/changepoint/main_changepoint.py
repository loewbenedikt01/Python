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

# ----
# Parameters
# ----

LAMBDA      = 0.95      # EWMA forgetting factor (~20-day memory), from the paper
VOL_TARGET  = 0.20      # annualised vol at which p_crisis = 0.5
SCALE       = 0.05
SERIES      = ('vix', 'gspc')
STARTUP     = 21
OBSERVATION = 5


# ----
# Build
# ----

def segment_volatility(seg:pd.Series) -> float:
    r2 = seg.to_numpy(dtype=float) ** 2
    n0 = min(OBSERVATION, len(r2))
    s = float(r2[:n0].mean())
    for x in r2[n0:]:
        s = LAMBDA * s + (1.0 - LAMBDA) * x
    return float(np.sqrt(s * 252))


def build(cp_csv: str, market_logret: pd.Series, out_path: str) -> pd.DataFrame:
    cp = pd.read_csv(cp_csv, parse_dates=["changepoint_date", "detection_date"])
    market_logret = market_logret.dropna().sort_index()
    idx = market_logret.index
 
    days  = idx.to_numpy(dtype="datetime64[ns]")
    pos_t = np.searchsorted(days, pd.to_datetime(cp["detection_date"]).to_numpy(dtype="datetime64[ns]"))
    pos_c = np.searchsorted(days, pd.to_datetime(cp["changepoint_date"]).to_numpy(dtype="datetime64[ns]"))
    det: dict[pd.Timestamp, pd.Timestamp] = {}
    for it, itau in zip(pos_t.tolist(), pos_c.tolist()):
        if it >= len(idx):
            continue
        t, tau = idx[it], idx[min(itau, it)]
        det[t] = min(tau, det.get(t, tau))

    first_t = idx[STARTUP - 1]
    sigma = segment_volatility(market_logret.iloc[:STARTUP])
    seg_start, regime_id = idx[0], 0
 
    rows = []
    for t in idx[STARTUP - 1:]:
        detected = t in det and t > first_t
        if detected:
            seg_start = det[t]
            sigma = segment_volatility(market_logret.loc[seg_start:t])
            regime_id += 1
        rows.append((t, sigma, regime_id, seg_start, detected))
 
    df = pd.DataFrame(rows, columns=["date", "sigma_ann", "regime_id",
                                     "tau_hat", "detected"]).set_index("date")
    df["p_crisis"] = norm.cdf((df["sigma_ann"] - VOL_TARGET) / SCALE)
    df["p_calm"]   = 1.0 - df["p_crisis"]
    df["label"]    = np.where(df["p_crisis"] > 0.5, "crisis", "calm")
 
    pos = pd.Series(np.arange(len(idx)), index=idx)
    df["segment_age"]     = pos.reindex(df.index).to_numpy() - pos.reindex(df["tau_hat"]).to_numpy()
    df["detection_delay"] = df["segment_age"].where(df["detected"])
 
    cols = ["p_calm", "p_crisis", "label", "sigma_ann", "regime_id",
            "tau_hat", "segment_age", "detected", "detection_delay"]
    out = df[cols].shift(1).dropna(subset=["p_crisis"])
    out["regime_id"] = out["regime_id"].astype(int)
    out["detected"]  = out["detected"].astype(bool)
 
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    return out


CRISIS_CACHE: dict[str, pd.DataFrame] = {}


def crisis_probs(idx: pd.DatetimeIndex, series: str = 'vix') -> pd.DataFrame:
    if series not in SERIES:
        raise ValueError(f"crisis_probs: series must be one of {SERIES}, got {series!r}")
    if series not in CRISIS_CACHE:
        path = Path(__file__).resolve().parent / "regimes_final" / f"changepoint_{series}.csv"
        df = pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()
        CRISIS_CACHE[series] = df[["p_calm", "p_crisis"]]
    src = CRISIS_CACHE[series]

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

    for series in SERIES:
        out = build(str(HERE / "regimes_r" / f"{series}_changepoint.csv"), mkt,
                    str(HERE / "regimes_final" / f"changepoint_{series}.csv"))
        print(f"[changepoint] {series}: {len(out)} rows, {int(out['detected'].sum())} breaks, "
              f"{out.index.min().date()} .. {out.index.max().date()}")