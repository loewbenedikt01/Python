"""
Wasserstein k-means regimes -> the common regime interface.

2-d WK-means of Mc Greevy et al. (2024, SSRN 4758243) on the joint daily
returns of the S&P 500 and the VIX, one market-wide regime per day:
    gspc_vix  2-d: S&P 500 log returns and VIX log changes (main)
    gspc      uni-d: S&P 500 log returns only (robustness; same algorithm)
Input is the same log-return parquet the other detectors use
(_database/changepoint.parquet, written by changepoint/data_prep.py).

Procedure:
  1. lift: every window of H1 = 21 daily returns (one month) is one empirical
     measure, a cloud of 21 points in R^d.  Each series is divided by its
     expanding standard deviation up to the day before the window starts, so
     VIX changes (about 4x as volatile) do not dominate the distance and the
     scale uses past data only;
  2. at the first trading day of each month, WK-means with K = 2 is fitted on
     all windows that end before that day (expanding; every TRAIN_STEP-th
     window, to keep the distance matrix tractable).  The crisis cluster is
     the one whose windows have the higher average S&P 500 variance;
  3. every day d, the window ending on d-1 is compared with both centroids,
     in squared W2, the loss k-means minimises:
        p_crisis(d) = W2(w, calm)^2 / (W2(w, calm)^2 + W2(w, crisis)^2),
     so p_crisis > 0.5 exactly when the window is closer to the crisis
     centroid.  The value at d uses only data through d-1.

The paper's trading test assumes the cluster of each new window is known in
advance; here clusters are fitted on past windows only and new windows are
assigned out of sample, so there is no look-ahead.

Output: regimes_final/wasserstein_{gspc_vix,gspc}.csv, daily, read by
regime_def.py.  Columns p_calm, p_crisis, label match the other detectors;
the rest are diagnostics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from wasserstein_engine import distance_matrix, w2, wk_means

# ----
# Parameters
# ----

SERIES      = {"gspc_vix": ("^GSPC", "^VIX"),    # output tag -> parquet columns
               "gspc":     ("^GSPC",)}           # first column decides which cluster is crisis
H1          = 21        # window length in days (one month)
H2          = 20        # overlap: scored windows slide by H1 - H2 = 1 day
TRAIN_STEP  = 5         # training uses every 5th window (lift (21, 16)); the pairwise
                        # distance matrix grows with the square of the window count
K           = 2         # calm / crisis
N_INIT      = 10        # k-means++ starts per refit, lowest total distance kept
SCALE_MIN   = 126       # days of history needed for the expanding standard deviation
MIN_TRAIN   = 252       # days of windows needed before the first fit
N_JOBS      = -1

HERE      = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "_database" / "regimes_data_log.parquet"
OUT_DIR   = HERE / "regimes_final"


# ----
# Data
# ----

def load_returns(path: Path = DATA_PATH) -> pd.DataFrame:
    """Daily log returns, one column per ticker, DatetimeIndex."""
    df = pd.read_parquet(path)
    for c in ("Date", "date"):
        if c in df.columns:
            df = df.set_index(c)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _expanding_std(r: np.ndarray) -> np.ndarray:
    """sd[t] = standard deviation of r[0:t] per column (data strictly before t)."""
    n = np.arange(len(r) + 1, dtype=float)[:, None]
    s1 = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    s2 = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r ** 2, axis=0)])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 ** 2 / n) / (n - 1)
    return np.sqrt(var)


def _window(r: np.ndarray, sd: np.ndarray, e: int) -> np.ndarray:
    """Scaled window of the H1 returns ending at position e, shape (H1, d)."""
    s = e - H1 + 1
    return r[s:e + 1] / sd[s]


# ----
# Fit and score
# ----

def _fit(D: np.ndarray, m: int, raw_var: np.ndarray, seed: int) -> dict:
    """WK-means on the first m training windows; identify the crisis cluster."""
    labels, cent, _ = wk_means(D[:m, :m], k=K, n_init=N_INIT, seed=seed)
    v = np.array([raw_var[:m][labels == j].mean() for j in range(K)])
    crisis = int(np.argmax(v))
    return {"cent": cent, "crisis": crisis, "calm": 1 - crisis,
            "var": v, "share_crisis": float((labels == crisis).mean())}


def build(r: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    r = r.dropna().sort_index()
    idx = r.index
    x = r.to_numpy(dtype=float)
    sd = _expanding_std(x)
    T = len(x)

    first_end = SCALE_MIN + H1 - 1                    # first window with a valid scale
    train_ends = np.arange(first_end, T, TRAIN_STEP)
    W_train = np.stack([_window(x, sd, e) for e in train_ends])
    raw_var = np.array([x[e - H1 + 1:e + 1, 0].var() for e in train_ends])   # S&P 500
    D = distance_matrix(W_train, n_jobs=N_JOBS)

    # refit at the first trading day of each month, once MIN_TRAIN days of windows exist
    month_first = np.flatnonzero(~idx.to_period("M").duplicated())
    refits = [t for t in month_first if t - 1 - first_end >= MIN_TRAIN]
    m_of = [int(np.searchsorted(train_ends, t - 1, side="right")) for t in refits]   # ends <= t-1
    fits = Parallel(n_jobs=N_JOBS)(
        delayed(_fit)(D, m, raw_var, seed=k) for k, m in enumerate(m_of)
    )

    rows = []
    bounds = refits[1:] + [T]
    for t0, t1, m, f in zip(refits, bounds, m_of, fits):
        calm_w = W_train[f["cent"][f["calm"]]]
        cris_w = W_train[f["cent"][f["crisis"]]]
        for d in range(t0, t1):
            w = _window(x, sd, d - 1)                 # returns through d-1
            dc, dk = w2(w, calm_w), w2(w, cris_w)
            p = dc ** 2 / (dc ** 2 + dk ** 2) if dc + dk > 0 else 0.5
            rows.append((idx[d], p, dc, dk, idx[t0], m, f["share_crisis"],
                         np.sqrt(f["var"][f["calm"]] * 252), np.sqrt(f["var"][f["crisis"]] * 252)))

    df = pd.DataFrame(rows, columns=["date", "p_crisis", "d_calm", "d_crisis", "refit_date",
                                     "n_train", "crisis_share_train",
                                     "sigma_calm_ann", "sigma_crisis_ann"]).set_index("date")
    df.insert(0, "p_calm", 1.0 - df["p_crisis"])
    df.insert(2, "label", np.where(df["p_crisis"] > 0.5, "crisis", "calm"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


if __name__ == "__main__":
    rets = load_returns()
    for tag, cols in SERIES.items():
        missing = [c for c in cols if c not in rets.columns]
        if missing:
            raise KeyError(f"[wasserstein] columns {missing} not in {DATA_PATH.name}: {list(rets.columns)}")
        out = build(rets[list(cols)], OUT_DIR / f"wasserstein_{tag}.csv")
        print(f"[wasserstein] {tag}: {len(out)} rows, {out.index.min().date()} .. {out.index.max().date()}, "
              f"crisis on {(out['p_crisis'] > 0.5).mean():.1%} of days")
