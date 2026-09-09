"""
main.py -- self-contained VIX / S&P 500 change-point regime detector.

Nonparametric, distribution-free sequential change-point detection
(Nystrup, Hansen, Madsen & Lindstrom, 2016, "Regime-Based Versus Static
Asset Allocation"). No parametric regime model: no fixed state count, no
distributional assumption, no EM. A rank-based Mood two-sample statistic
flags shifts in SCALE (volatility) -- the paper's key empirical finding is
that scale shifts carry the profitable signal while location (mean) shifts
only add detection delay.

This module deliberately reuses NOTHING from the sibling files in this
folder (detector.py, mood.py, thresholds.py, state.py, integrate.py,
start.py). The whole pipeline is inlined:

    1. load_index_close / log_returns
         VIX and S&P 500 daily closes -> daily log-returns.
    2. mood_max
         standardised |Mood statistic|, maximised over every split point.
    3. load_or_calibrate_thresholds
         sequential threshold h_t via Monte-Carlo simulation, with the
         CONDITIONAL false-alarm rate pinned at 1 / ARL0 (ARL0 = 10,000,
         i.e. one false alarm per 10,000 stable observations). Cached.
    4. detect
         strictly-forward sequential test on VIX log-returns with
         restart-after-detection. The state for date t is a function of
         returns[:t] only; detection lag (tau_hat < t) is kept, never
         back-dated (that would be look-ahead bias).
    5. compute_state
         EWMA(lambda = 0.95, ~20-day memory) volatility on S&P 500
         returns, re-seeded from the retained post-change observations at
         each detection, frozen between detections (paper-faithful).
         sigma_ann >= 20%  ->  crisis (100% cash);  else calm (100% index).
    6. build_regime_state / crisis_probs
         full daily regime frame, plus a causal as-of resampler returning
         p_calm / p_crisis for an arbitrary set of dates with a one-day
         implementation delay -- the shape Thesis/_models/xgb.py
         :regime_probs() consumes.

Run end-to-end (the first run calibrates thresholds -- a few minutes --
then caches them):

    python Thesis/_regimes/changepoint/main.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ----
# Configuration
# ----

_HERE           = Path(__file__).resolve().parent
PROJECT_ROOT    = _HERE.parents[2]
DATABASE        = PROJECT_ROOT / "Database" / "database.parquet"
CACHE_DIR       = _HERE / "cache"
RESULTS_DIR     = _HERE / "results"

VIX_TICKER = "^VIX"
SPX_TICKER = "^GSPC"

MIN_SEG        = 20      # minimum size of both Mood sub-samples
STARTUP        = 21      # obs seeding the first segment before monitoring (1 month)
ARL0           = 10_000  # target average run length between false alarms
LAMBDA         = 0.95    # EWMA decay -> ~20 trading-day memory
TRADING_DAYS   = 252
VOL_THRESHOLD  = 0.20    # annualised-vol switch point (a long-run-average
                         # assumption -- NOT to be tuned by backtest P&L)
IMPL_DELAY     = 1       # act on a signal one trading day after it is known
RESEED_MIN_OBS = 5       # min post-change obs to re-seed the EWMA directly

# Monte-Carlo threshold-calibration defaults (paper-scale).
MC_N_PATHS   = 20_000
MC_T_MAX     = 1_500     # thresholds held flat beyond this (documented conservatism)
MC_SEED      = 0
MC_MIN_ALIVE = 500       # stop calibrating once fewer paths survive


def label_crisis(date: pd.Timestamp) -> str:
    for name, c_start, c_end in CRISIS_PERIODS:
        if pd.Timestamp(c_start) <= date <= pd.Timestamp(c_end):
            return name
    return ""


# ----
# 1. Data
# ----

def load_index_close(ticker: str, parquet_path: Path = INDICES_PARQUET) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    if ticker not in df.index.get_level_values(-1):
        raise KeyError(f"{ticker!r} not in {parquet_path.name}")
    s = df.xs(ticker, level=-1)["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[s > 0.0].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(ticker)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns (VIX 'daily change' = log change, matching the repo)."""
    return np.log(prices / prices.shift(1)).dropna()


# ----
# 2. Mood test  (rank-based, distribution-free scale-shift statistic)
# ----
#
# At time t we hold x_1..x_t in time order. For a candidate split k, sample
# A = {x_1..x_{k-1}} (n_A = k-1) and B = {x_k..x_t} (n_B = t-k+1), n = t.
# Rank the pooled sample 1..n; r(x_i) is the rank of x_i.
#
#   M'_{k,t} = sum_{x_i in A} ( r(x_i) - (n+1)/2 )^2
#   mu       = n_A (n^2 - 1) / 12
#   var      = n_A n_B (n+1)(n^2 - 4) / 180
#   M_{k,t}  = | (M'_{k,t} - mu) / sqrt(var) |          (both-sided: up or down)
#
# M depends only on ranks, so under H0 its null law is independent of F --
# which is exactly what keeps the calibrated thresholds valid on fat-tailed
# return data.
#
# tau_hat convention: 0-indexed position of the FIRST observation of segment
# B, i.e. tau_hat == n_A at the maximising split.

def _mood_M_from_ranks(ranks: np.ndarray, n: int, min_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardised |Mood| for every split n_A = j in [min_seg, n - min_seg]."""
    c = (ranks - (n + 1) / 2.0) ** 2
    prefix = np.cumsum(c)                        # prefix[j-1] = M' for n_A = j
    j = np.arange(min_seg, n - min_seg + 1)
    n_A = j.astype(float)
    n_B = n - n_A
    mu = n_A * (n ** 2 - 1) / 12.0
    var = n_A * n_B * (n + 1) * (n ** 2 - 4) / 180.0
    M = np.abs((prefix[j - 1] - mu) / np.sqrt(var))
    return M, j


def mood_max(x: np.ndarray, min_seg: int = MIN_SEG) -> tuple[float, int]:
    """
    max standardised |Mood statistic| over all valid splits of x (time order).
    Returns (D_max, tau_hat); (0.0, -1) if len(x) < 2 * min_seg. Uses
    tie-averaged ranks -- correct for real return data, which has ties.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_seg:
        return 0.0, -1
    M, j = _mood_M_from_ranks(rankdata(x, method="average"), n, min_seg)
    best = int(np.argmax(M))
    return float(M[best]), int(j[best])


def _mood_max_batch(X: np.ndarray, min_seg: int = MIN_SEG) -> np.ndarray:
    """
    Vectorised D_max over many i.i.d. simulated paths -- calibration only.
    X: (n_paths, t), time order along axis 1. Ranks via argsort (no tie
    averaging), valid for the continuous standard-normal innovations used
    to calibrate. Returns D_max, shape (n_paths,).
    """
    X = np.asarray(X, dtype=float)
    n_paths, n = X.shape
    if n < 2 * min_seg:
        return np.zeros(n_paths)
    order = np.argsort(X, axis=1)
    ranks = np.empty_like(order, dtype=float)
    fill = np.broadcast_to(np.arange(1, n + 1, dtype=float), (n_paths, n))
    np.put_along_axis(ranks, order, fill, axis=1)

    c = (ranks - (n + 1) / 2.0) ** 2
    prefix = np.cumsum(c, axis=1)
    j = np.arange(min_seg, n - min_seg + 1)
    n_A = j[None, :].astype(float)
    n_B = n - n_A
    mu = n_A * (n ** 2 - 1) / 12.0
    var = n_A * n_B * (n + 1) * (n ** 2 - 4) / 180.0
    M = np.abs((prefix[:, j - 1] - mu) / np.sqrt(var))
    return M.max(axis=1)


# ----
# 3. Sequential threshold calibration
# ----
#
# The detector fires when D_max,t > h_t. The sequence h_2, h_3, ... is
# calibrated so that
#
#   P( D_max,t > h_t | D_max,i <= h_i for all i < t ) = 1 / ARL0
#
# h_t has no closed form (it depends on the whole preceding sequence) and
# is simulated once via Monte Carlo, then cached. Distribution-free: the
# paths are standard normal but the ranks make h_t valid for any continuous
# F, fat tails included.

def calibrate_thresholds(
    n_paths: int = MC_N_PATHS,
    t_max: int = MC_T_MAX,
    min_seg: int = MIN_SEG,
    arl0: int = ARL0,
    seed: int = MC_SEED,
    min_alive: int = MC_MIN_ALIVE,
    verbose: bool = True,
) -> np.ndarray:
    """h[t] = threshold to apply when the monitored segment holds exactly t
    observations. h[0 .. 2*min_seg - 1] = inf (no valid split yet)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_paths, t_max))       # ~ n_paths * t_max * 8 bytes
    alive = np.ones(n_paths, dtype=bool)
    alpha = 1.0 / arl0

    h = np.full(t_max + 1, np.inf)
    last_h = np.inf
    for t in range(2 * min_seg, t_max + 1):
        n_alive = int(alive.sum())
        if n_alive < min_alive:
            if verbose:
                print(f"  [thresholds] stop at t={t}: {n_alive} paths alive")
            h[t:] = last_h
            break
        D = _mood_max_batch(X[:, :t], min_seg=min_seg)
        h_t = float(np.quantile(D[alive], 1.0 - alpha))
        h[t] = last_h = h_t
        alive &= D <= h_t
        if verbose and (t % 200 == 0 or t == t_max):
            print(f"  [thresholds] t={t:5d}  h_t={h_t:.4f}  alive={int(alive.sum())}")
    return h


def load_or_calibrate_thresholds(
    n_paths: int = MC_N_PATHS,
    t_max: int = MC_T_MAX,
    min_seg: int = MIN_SEG,
    arl0: int = ARL0,
    seed: int = MC_SEED,
    force: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Cached wrapper around `calibrate_thresholds` (cache/thresholds_*.npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"thresholds_arl{arl0}_seg{min_seg}.npz"
    meta = dict(n_paths=n_paths, t_max=t_max, min_seg=min_seg, arl0=arl0, seed=seed)

    if cache_file.exists() and not force:
        data = np.load(cache_file)
        if all(int(data[k]) == v for k, v in meta.items()):
            if verbose:
                print(f"  [thresholds] loaded cache {cache_file.name}")
            return data["h"]
        if verbose:
            print("  [thresholds] cache params differ -- regenerating")

    if verbose:
        print(f"  [thresholds] calibrating {meta} (a few minutes)...")
    h = calibrate_thresholds(n_paths, t_max, min_seg, arl0, seed, verbose=verbose)
    np.savez(cache_file, h=h, **meta)
    if verbose:
        print(f"  [thresholds] cached to {cache_file.name}")
    return h


# ----
# 4. Sequential detector with restart
# ----

def detect(
    returns: pd.Series,
    thresholds: np.ndarray,
    min_seg: int = MIN_SEG,
    startup: int = STARTUP,
) -> pd.DataFrame:
    """
    Strictly-forward Mood-test change-point detection with
    restart-after-detection.

    Returns a frame indexed by returns.index[startup:] with columns:
        detected         bool     -- change point flagged on this day
        tau_hat          datetime -- estimated change-point date (NaT otherwise)
        detection_delay  float    -- trading days between tau_hat and detection
        segment_age      int      -- days since the current segment's start
                                     (on a detection row: age of the NEW segment)
    """
    if returns.isna().any():
        bad = returns[returns.isna()].index[:5].tolist()
        raise ValueError(f"returns has NaNs, e.g. at {bad} -- fill or drop first")

    values = returns.to_numpy(dtype=float)
    dates = returns.index
    n = len(values)
    if n <= startup:
        raise ValueError(f"series len={n} must exceed startup={startup}")
    max_h_idx = len(thresholds) - 1

    records = []
    seg_start = 0                                   # index of the segment's first obs
    for t in range(startup, n):
        segment = values[seg_start : t + 1]         # causal: data up to & incl. t
        seg_len = len(segment)

        detected, tau_hat_date, delay = False, pd.NaT, np.nan
        if seg_len >= 2 * min_seg:
            h_t = thresholds[min(seg_len, max_h_idx)]
            D, tau_local = mood_max(segment, min_seg=min_seg)
            if D > h_t:
                detected = True
                tau_pos = seg_start + tau_local
                tau_hat_date = dates[tau_pos]
                delay = float(t - tau_pos)
                seg_start = tau_pos                 # restart, keep post-change obs

        records.append({
            "date": dates[t],
            "detected": detected,
            "tau_hat": tau_hat_date,
            "detection_delay": delay,
            "segment_age": t - seg_start,
        })

    out = pd.DataFrame.from_records(records).set_index("date")
    out["detected"] = out["detected"].astype(bool)
    out["segment_age"] = out["segment_age"].astype(int)
    return out


# ----
# 5. EWMA regime volatility + switching state
# ----

def compute_state(
    vol_returns: pd.Series,
    detection: pd.DataFrame,
    lam: float = LAMBDA,
    mode: str = "frozen",
    vol_threshold: float = VOL_THRESHOLD,
    reseed_min_obs: int = RESEED_MIN_OBS,
) -> pd.DataFrame:
    """
    Detector output -> daily annualised vol, crisis state and equity exposure.

        EWMA_t    = lam * EWMA_{t-1} + (1 - lam) * r_t^2
        sigma_ann = sqrt(252 * EWMA_t)

    On a detection at t with change point tau_hat, the EWMA is re-seeded from
    mean(r[tau_hat:t+1]^2) when at least `reseed_min_obs` post-change obs are
    available, else the previous regime's value is kept and left to decay.
      - mode 'frozen'  (primary): sigma held at its detection-time value until
        the next detection.
      - mode 'rolling' (robustness): EWMA keeps updating every day.

    Columns: sigma_ann, p_crisis, exposure, regime_id, segment_age, detected.
    """
    if mode not in ("frozen", "rolling"):
        raise ValueError("mode must be 'frozen' or 'rolling'")

    idx = detection.index
    r = vol_returns.reindex(idx)
    if r.isna().any():
        bad = r[r.isna()].index[:5].tolist()
        raise ValueError(f"vol_returns missing inside monitored window, e.g. {bad}")

    r2 = r.to_numpy(dtype=float) ** 2
    seg_age = detection["segment_age"].to_numpy()
    detected = detection["detected"].to_numpy()
    n = len(idx)

    ewma_roll = np.empty(n)
    ewma_frozen = np.empty(n)
    regime_id = np.empty(n, dtype=int)

    running = float(r2[0])
    frozen = running
    rid = 0
    for i in range(n):
        if detected[i]:
            rid += 1
            local_start = i - int(seg_age[i])       # position of tau_hat
            window = r2[local_start : i + 1]
            if len(window) >= reseed_min_obs:
                running = float(window.mean())
            frozen = running
        else:
            running = lam * running + (1.0 - lam) * r2[i]
        ewma_roll[i] = running
        ewma_frozen[i] = frozen
        regime_id[i] = rid

    ewma = ewma_frozen if mode == "frozen" else ewma_roll
    sigma_ann = np.sqrt(TRADING_DAYS * ewma)
    p_crisis = (sigma_ann >= vol_threshold).astype(float)

    return pd.DataFrame({
        "sigma_ann": sigma_ann,
        "p_crisis": p_crisis,
        "exposure": 1.0 - p_crisis,                 # long-only switching
        "regime_id": regime_id,
        "segment_age": seg_age,
        "detected": detected,
    }, index=idx)


# ----
# 6. Full daily regime state + causal resampler
# ----

_STATE_CACHE: dict[tuple, pd.DataFrame] = {}


def build_regime_state(
    start: str | None = None,
    end: str | None = None,
    mode: str = "frozen",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the whole pipeline and return the daily regime frame.

    Change points are detected on VIX log-returns; regime volatility is
    estimated on S&P 500 log-returns -- two series, two roles, never
    conflated. The window is clipped to where both are defined.
    """
    vix_lr = log_returns(load_index_close(VIX_TICKER))
    spx_lr = log_returns(load_index_close(SPX_TICKER))
    lo = max(vix_lr.index.min(), spx_lr.index.min())
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    hi = min(vix_lr.index.max(), spx_lr.index.max())
    if end is not None:
        hi = min(hi, pd.Timestamp(end))
    vix_lr = vix_lr.loc[lo:hi]
    spx_lr = spx_lr.loc[lo:hi]
    if verbose:
        print(f"  [data] window {lo.date()} -> {hi.date()}  "
              f"({len(vix_lr)} VIX obs, {len(spx_lr)} SPX obs)")

    h = load_or_calibrate_thresholds(verbose=verbose)
    det = detect(vix_lr, h)
    spx_aligned = spx_lr.reindex(det.index).ffill().bfill()
    st = compute_state(spx_aligned, det, mode=mode)
    if verbose:
        print(f"  [detect] {int(st['detected'].sum())} change points over "
              f"{len(st)} monitored days; {int(st['regime_id'].iloc[-1]) + 1} regimes")
    return st


def crisis_probs(
    dates,
    mode: str = "frozen",
    implementation_delay_days: int = IMPL_DELAY,
) -> pd.DataFrame:
    """
    Causal one-step-ahead crisis probabilities for an arbitrary date set --
    the shape Thesis/_models/xgb.py:regime_probs() consumes.

    For each requested date D, take the regime state known at the close of
    the last trading day <= D minus `implementation_delay_days` (a state
    computed at close t is only actionable at t + delay). Dates before the
    detector starts are 'calm'. Output columns p_calm, p_crisis sum to 1;
    p_crisis is in {0, 1} (the paper switches hard).
    """
    key = (mode,)
    st = _STATE_CACHE.get(key)
    if st is None:
        st = _STATE_CACHE[key] = build_regime_state(mode=mode, verbose=False)

    idx = pd.DatetimeIndex(dates)
    asof = idx - pd.Timedelta(days=implementation_delay_days) if implementation_delay_days else idx
    pc = st["p_crisis"].astype(float).asof(asof).to_numpy()
    pc = np.nan_to_num(pc, nan=0.0)
    return pd.DataFrame({"p_calm": 1.0 - pc, "p_crisis": pc}, index=idx)


# ----
# 7. CLI
# ----

def main() -> None:
    print("=== VIX / S&P 500 change-point regime detector "
          "(Nystrup et al. 2016, self-contained) ===\n")
    t0 = time.time()
    st = build_regime_state(verbose=True)

    hits = st[st["detected"]].copy()
    hits["tau_hat"] = hits.index - pd.to_timedelta(hits["segment_age"], unit="D")
    frac_crisis = float(st["p_crisis"].mean())
    mean_expo = float(st["exposure"].mean())

    print(f"\n{'detected':<12} {'~tau_hat':<12} {'delay(d)':>8}  crisis window")
    print("-" * 60)
    for date, row in hits.iterrows():
        print(f"{str(date.date()):<12} {str(pd.Timestamp(row['tau_hat']).date()):<12} "
              f"{int(row['segment_age']):>8}  {label_crisis(date)}")

    print(f"\nregimes            : {int(st['regime_id'].iloc[-1]) + 1}")
    print(f"days in crisis      : {frac_crisis:.1%}")
    print(f"mean equity exposure: {mean_expo:.1%}  (switching @ {VOL_THRESHOLD:.0%} ann. vol)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    st.to_csv(RESULTS_DIR / "regime_state.csv")
    st.to_parquet(RESULTS_DIR / "regime_state.parquet")
    hits[["tau_hat", "segment_age"]].to_csv(RESULTS_DIR / "detections.csv")
    print(f"\nsaved  {RESULTS_DIR / 'regime_state.csv'}")
    print(f"saved  {RESULTS_DIR / 'regime_state.parquet'}")
    print(f"saved  {RESULTS_DIR / 'detections.csv'}")

    print(f"\ndone in {time.time() - t0:.1f}s")
    print("\nwire into xgb.py:  DETECTOR = 'changepoint', then in regime_probs():")
    print("    from _regimes.changepoint.main import crisis_probs")
    print("    return crisis_probs(dates)")


if __name__ == "__main__":
    main()
