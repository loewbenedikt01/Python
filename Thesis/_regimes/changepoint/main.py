"""
main.py -- self-contained VIX / S&P 500 change-point regime detector.

Nonparametric, distribution-free sequential change-point detection
(Nystrup, Hansen, Madsen & Lindstrom, 2016, "Regime-Based Versus Static
Asset Allocation"), adapted from an allocation switch to a *probability*
signal for the common regime interface used by Thesis/_models/*.

Adaptations vs. the paper
-------------------------
* ARL0 = 500 (~2 trading years between false alarms), not 10,000. The paper
  needs risk-on/risk-off with rare, expensive false alarms; we need enough
  regime resolution to weight training samples. ARL0 is a swept parameter --
  run `python main.py --arl0 {500,1000,5000}` and report the sensitivity row.
* Output is a probability, not an allocation. The EWMA annualised vol of the
  active segment is mapped through a probit:
        p_crisis(t) = Phi( (sigma_ann(t) - VOL_THRESHOLD) / PROBIT_SCALE )
  VOL_THRESHOLD (0.20) and PROBIT_SCALE (0.05) are FIXED PRIORS -- the
  paper's long-run equity-vol assumption -- not fitted to this sample, so
  they introduce no look-ahead. `label` keeps the hard 20% switch for the
  robustness spec.

Which series feeds which step
-----------------------------
* detector  : VIX daily LOG-DIFFERENCES (the paper's most profitable input;
              scale shifts in d log VIX = vol-of-vol regime changes).
* EWMA vol  : realised S&P 500 log-returns over the CURRENT segment
              [tau_hat, t]. Two series, two roles -- never conflated.

Pipeline (all inlined; nothing imported from the sibling files)
--------------------------------------------------------------
  1. load_index_close / log_returns
  2. mood_max               rank-based scale-shift statistic, argmax over splits
  3. load_or_calibrate_thresholds
                            sequential h_t via Monte-Carlo, conditional
                            false-alarm rate pinned at 1/ARL0. Cached.
  4. detect                 strictly-forward test on d log VIX, restart after
                            each detection. State at t uses returns[:t] only;
                            detection lag (tau_hat < t) is kept, never back-dated.
                            Emits daily d_max and n_active as continuous extras.
  5. compute_state          EWMA(lambda=0.95) vol on SPX returns, re-seeded from
                            the retained post-change observations, frozen between
                            detections (paper-faithful).
  6. build_regime_state     assembles the daily frame; probit -> p_calm/p_crisis.
     crisis_probs           causal resampler: for a rebalance date D, reads the
                            state of the PRIOR trading day (= the paper's one-day
                            execution delay at month-start frequency).
     write_regime_parquet   -> _database/_regimes/vix_cp.parquet

Run end-to-end (first run calibrates thresholds, then caches):
    python Thesis/_regimes/changepoint/main.py [--arl0 500]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

# ----
# Configuration
# ----

_HERE           = Path(__file__).resolve().parent
THESIS_ROOT     = _HERE.parents[1]                      # .../Python/Thesis
PROJECT_ROOT    = _HERE.parents[2]                      # .../Python
INDICES_PARQUET = PROJECT_ROOT / "Database" / "indices.parquet"
CACHE_DIR       = _HERE / "cache"
RESULTS_DIR     = _HERE / "results"
OUTPUT_PARQUET  = THESIS_ROOT / "_database" / "_regimes" / "vix_cp.parquet"

VIX_TICKER = "^VIX"      # change points detected on d log VIX (paper: most profitable)
SPX_TICKER = "^GSPC"     # regime volatility estimated on realised SPX returns

MIN_SEG        = 10      # >= 10 obs on EACH side (Mood variance formula unreliable below)
STARTUP        = 21      # obs seeding the first segment before monitoring (1 month)
ARL0           = 500     # target average run length between false alarms (~2y). Swept.
LAMBDA         = 0.95    # EWMA decay -> ~20 trading-day memory
TRADING_DAYS   = 252
VOL_THRESHOLD  = 0.20    # probit centre = long-run equity-vol prior (FIXED, not fitted)
PROBIT_SCALE   = 0.05    # probit scale s (FIXED, not fitted)
RESEED_MIN_OBS = 5       # min post-change obs to re-seed the EWMA directly

# Split-scan coarsening: once the active window exceeds GRID_AFTER, test only
# every GRID_STRIDE-th split point. Bounds tau_hat resolution to +/-GRID_STRIDE
# for old segments (days_since_break is coarse for old breaks anyway) and keeps
# the daily scan O(t / stride). The MC calibration uses the SAME rule so h_t
# matches the coarsened statistic.
GRID_AFTER  = 200
GRID_STRIDE = 5

# Monte-Carlo threshold-calibration defaults.
MC_N_PATHS   = 20_000
MC_T_MAX     = 1_000     # thresholds held flat beyond this (windows rarely get here
                         # once ARL0 is 500 -- resets are frequent)
MC_SEED      = 0
MC_MIN_ALIVE = 400       # stop calibrating once fewer paths survive

# Well-known crisis windows -- annotation only, never fed to the detector.
CRISIS_PERIODS = [
    ("Dot-com crash",    "2000-03-24", "2002-10-09"),
    ("GFC",              "2007-10-09", "2009-03-09"),
    ("Euro sovereign",   "2011-05-02", "2011-10-04"),
    ("China / oil",      "2015-08-01", "2016-02-11"),
    ("Volmageddon",      "2018-01-26", "2018-02-14"),
    ("Q4-2018 selloff",  "2018-10-01", "2018-12-24"),
    ("COVID-19 crash",   "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
]


def label_crisis(date: pd.Timestamp) -> str:
    for name, c_start, c_end in CRISIS_PERIODS:
        if pd.Timestamp(c_start) <= date <= pd.Timestamp(c_end):
            return name
    return ""


# ----
# 1. Data
# ----

def load_index_close(ticker: str, parquet_path: Path = INDICES_PARQUET) -> pd.Series:
    """Daily close for one ticker from the (Date, ticker) MultiIndex parquet."""
    df = pd.read_parquet(parquet_path)
    if ticker not in df.index.get_level_values(-1):
        raise KeyError(f"{ticker!r} not in {parquet_path.name}")
    s = df.xs(ticker, level=-1)["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[s > 0.0].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(ticker)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-differences: d log P_t. For VIX this is the 'daily change'
    series the paper tests; for SPX it is the realised return."""
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
# which is what keeps the calibrated thresholds valid on fat-tailed returns.
#
# tau_hat convention: 0-indexed position of the FIRST observation of segment
# B, i.e. tau_hat == n_A at the maximising split.

def _mood_M_from_ranks(
    ranks: np.ndarray, n: int, min_seg: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised |Mood| for split points n_A = j in [min_seg, n - min_seg]
    (every `stride`-th j)."""
    c = (ranks - (n + 1) / 2.0) ** 2
    prefix = np.cumsum(c)                        # prefix[j-1] = M' for n_A = j
    j = np.arange(min_seg, n - min_seg + 1, stride)
    n_A = j.astype(float)
    n_B = n - n_A
    mu = n_A * (n ** 2 - 1) / 12.0
    var = n_A * n_B * (n + 1) * (n ** 2 - 4) / 180.0
    M = np.abs((prefix[j - 1] - mu) / np.sqrt(var))
    return M, j


def mood_max(x: np.ndarray, min_seg: int = MIN_SEG, stride: int = 1) -> tuple[float, int]:
    """
    max standardised |Mood statistic| over the scanned splits of x (time order).
    Returns (D_max, tau_hat); (0.0, -1) if len(x) < 2 * min_seg. Tie-averaged
    ranks -- correct for real return data, which has ties.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_seg:
        return 0.0, -1
    M, j = _mood_M_from_ranks(rankdata(x, method="average"), n, min_seg, stride)
    best = int(np.argmax(M))
    return float(M[best]), int(j[best])


def _scan_stride(seg_len: int) -> int:
    return GRID_STRIDE if seg_len > GRID_AFTER else 1


def _mood_max_batch(X: np.ndarray, min_seg: int, stride: int) -> np.ndarray:
    """
    Vectorised D_max over many i.i.d. simulated paths -- calibration only.
    X: (n_paths, t), time order along axis 1. Ranks via argsort (no tie
    averaging), valid for the continuous standard-normal calibration paths.
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
    j = np.arange(min_seg, n - min_seg + 1, stride)
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
# The detector fires when D_max,t > h_t. h_2, h_3, ... is calibrated so that
#
#   P( D_max,t > h_t | D_max,i <= h_i for all i < t ) = 1 / ARL0
#
# h_t has no closed form and is simulated once via Monte Carlo, then cached.
# The paths are standard normal but the rank statistic makes h_t valid for
# any continuous F (fat tails included) -- and it never touches the real
# data, so calibration introduces no look-ahead.

def calibrate_thresholds(
    n_paths: int = MC_N_PATHS,
    t_max: int = MC_T_MAX,
    min_seg: int = MIN_SEG,
    arl0: int = ARL0,
    seed: int = MC_SEED,
    min_alive: int = MC_MIN_ALIVE,
    verbose: bool = True,
) -> np.ndarray:
    """h[t] = threshold when the monitored segment holds exactly t obs.
    h[0 .. 2*min_seg - 1] = inf (no valid split yet)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_paths, t_max))
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
        D = _mood_max_batch(X[:, :t], min_seg, _scan_stride(t))
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
    """Cached wrapper around `calibrate_thresholds`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"arl{arl0}_seg{min_seg}_g{GRID_AFTER}x{GRID_STRIDE}"
    cache_file = CACHE_DIR / f"thresholds_{tag}.npz"
    meta = dict(n_paths=n_paths, t_max=t_max, min_seg=min_seg, arl0=arl0, seed=seed,
                grid_after=GRID_AFTER, grid_stride=GRID_STRIDE)

    if cache_file.exists() and not force:
        data = np.load(cache_file)
        if all(int(data[k]) == v for k, v in meta.items()):
            if verbose:
                print(f"  [thresholds] loaded cache {cache_file.name}")
            return data["h"]
        if verbose:
            print("  [thresholds] cache params differ -- regenerating")

    if verbose:
        print(f"  [thresholds] calibrating {meta} ...")
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
    Strictly-forward Mood-test change-point detection with restart-after-
    detection, run on `returns` (d log VIX).

    Returns a frame indexed by returns.index[startup:] with columns:
        detected         bool     -- change point flagged on this day
        break_date       datetime -- tau_hat currently in effect (segment start;
                                     = series start before the first detection)
        refit_date       datetime -- date the current segment was OPENED, i.e.
                                     the last detection date (NaT before the first)
        days_since_break int      -- t - tau_hat (trading days)
        n_active         int      -- obs in the active monitored window (resets
                                     at each break; = days_since_break + 1)
        d_max            float    -- today's standardised Mood statistic
        detection_delay  float    -- t - tau_hat on detection rows, else NaN
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
    seg_start = 0                                    # index of the segment's first obs
    last_refit = pd.NaT
    for t in range(startup, n):
        seg_len = t + 1 - seg_start                  # causal: obs up to & incl. t
        d_max = 0.0
        detected, delay = False, np.nan
        if seg_len >= 2 * min_seg:
            segment = values[seg_start : t + 1]
            h_t = thresholds[min(seg_len, max_h_idx)]
            d_max, tau_local = mood_max(segment, min_seg, _scan_stride(seg_len))
            if d_max > h_t:
                detected = True
                tau_pos = seg_start + tau_local
                delay = float(t - tau_pos)
                seg_start = tau_pos                  # restart, keep post-change obs
                last_refit = dates[t]

        records.append({
            "date": dates[t],
            "detected": detected,
            "break_date": dates[seg_start],
            "refit_date": last_refit,
            "days_since_break": t - seg_start,
            "n_active": t + 1 - seg_start,
            "d_max": d_max,
            "detection_delay": delay,
        })

    out = pd.DataFrame.from_records(records).set_index("date")
    out["detected"] = out["detected"].astype(bool)
    out["days_since_break"] = out["days_since_break"].astype(int)
    out["n_active"] = out["n_active"].astype(int)
    return out


# ----
# 5. EWMA regime volatility of the active segment
# ----

def compute_state(
    vol_returns: pd.Series,
    detection: pd.DataFrame,
    lam: float = LAMBDA,
    mode: str = "frozen",
    reseed_min_obs: int = RESEED_MIN_OBS,
) -> pd.Series:
    """
    Detector output + realised SPX returns -> daily annualised vol of the
    CURRENT segment.

        EWMA_t    = lam * EWMA_{t-1} + (1 - lam) * r_t^2
        sigma_ann = sqrt(252 * EWMA_t)

    On a detection at t (change point tau_hat) the EWMA is re-seeded from
    mean(r[tau_hat:t+1]^2) when >= reseed_min_obs post-change obs exist, else
    the previous regime's value is kept and decays via the ordinary recursion.
      - mode 'frozen'  (primary): held at the detection-time value until the
        next detection.
      - mode 'rolling' (robustness): updates every day.

    Returns a Series `sigma_ann` indexed like `detection`.
    """
    if mode not in ("frozen", "rolling"):
        raise ValueError("mode must be 'frozen' or 'rolling'")

    idx = detection.index
    r = vol_returns.reindex(idx)
    if r.isna().any():
        bad = r[r.isna()].index[:5].tolist()
        raise ValueError(f"vol_returns missing inside monitored window, e.g. {bad}")

    r2 = r.to_numpy(dtype=float) ** 2
    since_break = detection["days_since_break"].to_numpy()
    detected = detection["detected"].to_numpy()
    n = len(idx)

    ewma = np.empty(n)
    running = float(r2[0])
    frozen = running
    for i in range(n):
        if detected[i]:
            local_start = max(0, i - int(since_break[i]))   # tau_hat position (clamped:
                                                            # tau_hat can precede startup)
            window = r2[local_start : i + 1]
            if len(window) >= reseed_min_obs:
                running = float(window.mean())
            frozen = running
        else:
            running = lam * running + (1.0 - lam) * r2[i]
        ewma[i] = frozen if mode == "frozen" else running

    return pd.Series(np.sqrt(TRADING_DAYS * ewma), index=idx, name="sigma_ann")


# ----
# 6. Full daily regime frame + causal resampler
# ----

def _map_probs(sigma_ann: np.ndarray, vol_threshold: float, probit_scale: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Probit map: p_crisis = Phi((sigma_ann - centre) / scale); hard label."""
    p_crisis = norm.cdf((sigma_ann - vol_threshold) / probit_scale)
    label = (sigma_ann >= vol_threshold).astype("int8")
    return p_crisis, label


def build_regime_state(
    start: str | None = None,
    end: str | None = None,
    arl0: int = ARL0,
    mode: str = "frozen",
    vol_threshold: float = VOL_THRESHOLD,
    probit_scale: float = PROBIT_SCALE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the whole pipeline and return the daily regime frame.

    Detector runs on d log VIX; segment volatility on realised SPX returns.
    Window clipped to where both series are defined (and to [start, end]).

    Columns: p_calm, p_crisis, label, sigma_ann, d_max, days_since_break,
    n_active, refit_date  (+ detected, break_date, regime_id for inspection).
    """
    vix_chg = log_returns(load_index_close(VIX_TICKER))     # detector input
    spx_ret = log_returns(load_index_close(SPX_TICKER))     # EWMA-vol input
    lo = max(vix_chg.index.min(), spx_ret.index.min())
    hi = min(vix_chg.index.max(), spx_ret.index.max())
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    if end is not None:
        hi = min(hi, pd.Timestamp(end))
    vix_chg = vix_chg.loc[lo:hi]
    spx_ret = spx_ret.loc[lo:hi]
    if verbose:
        print(f"  [data] window {lo.date()} -> {hi.date()}  "
              f"({len(vix_chg)} VIX obs, {len(spx_ret)} SPX obs)")

    h = load_or_calibrate_thresholds(arl0=arl0, verbose=verbose)
    det = detect(vix_chg, h)
    spx_aligned = spx_ret.reindex(det.index).ffill().bfill()
    sigma_ann = compute_state(spx_aligned, det, mode=mode)

    p_crisis, label = _map_probs(sigma_ann.to_numpy(), vol_threshold, probit_scale)
    out = pd.DataFrame({
        "p_calm": 1.0 - p_crisis,
        "p_crisis": p_crisis,
        "label": label,
        "sigma_ann": sigma_ann.to_numpy(),
        "d_max": det["d_max"].to_numpy(),
        "days_since_break": det["days_since_break"].to_numpy(),
        "n_active": det["n_active"].to_numpy(),
        "refit_date": det["refit_date"].to_numpy(),
        "detected": det["detected"].to_numpy(),
        "break_date": det["break_date"].to_numpy(),
        "regime_id": det["detected"].cumsum().to_numpy().astype(int),
    }, index=det.index)
    if verbose:
        n_det = int(out["detected"].sum())
        print(f"  [detect] arl0={arl0}: {n_det} change points over {len(out)} days; "
              f"{out['regime_id'].iloc[-1] + 1} regimes; "
              f"mean p_crisis {out['p_crisis'].mean():.3f}; "
              f"label crisis {out['label'].mean():.1%} of days")
    return out


_STATE_CACHE: dict[tuple, pd.DataFrame] = {}


def _cached_state(arl0: int, mode: str) -> pd.DataFrame:
    key = (arl0, mode)
    st = _STATE_CACHE.get(key)
    if st is None:
        st = _STATE_CACHE[key] = build_regime_state(arl0=arl0, mode=mode, verbose=False)
    return st


def crisis_probs(
    dates,
    arl0: int = ARL0,
    mode: str = "frozen",
    lag_trading_days: int = 1,
) -> pd.DataFrame:
    """
    Causal crisis probabilities for an arbitrary date set -- the shape
    Thesis/_models/xgb.py:regime_probs() consumes.

    For each requested date D, return the regime state of the trading day
    `lag_trading_days` before the last state row <= D. At month-start
    rebalances this reproduces the paper's one-day execution delay: you act
    on what was known at the prior close. Dates before the detector starts
    are 'calm' (p_crisis = 0). Columns p_calm, p_crisis sum to 1; p_crisis
    is the smooth probit value in [0, 1].
    """
    st = _cached_state(arl0, mode)
    s = st["p_crisis"].to_numpy(dtype=float)
    state_idx = st.index.values

    q = np.asarray(pd.DatetimeIndex(dates).values, dtype="datetime64[ns]")
    pos = np.searchsorted(state_idx, q, side="left") - lag_trading_days  # prior trading day
    pc = np.where(pos >= 0, s[np.clip(pos, 0, len(s) - 1)], 0.0)
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"p_calm": 1.0 - pc, "p_crisis": pc}, index=idx)


def write_regime_parquet(st: pd.DataFrame, path: Path = OUTPUT_PARQUET) -> Path:
    """Persist the 8-column regime signal for downstream models."""
    cols = ["p_calm", "p_crisis", "label", "sigma_ann",
            "d_max", "days_since_break", "n_active", "refit_date"]
    path.parent.mkdir(parents=True, exist_ok=True)
    st[cols].to_parquet(path)
    return path


# ----
# 7. CLI
# ----

def main() -> None:
    arl0 = ARL0
    if "--arl0" in sys.argv:
        arl0 = int(sys.argv[sys.argv.index("--arl0") + 1])

    print(f"=== VIX / S&P 500 change-point regime detector "
          f"(Nystrup et al. 2016, adapted; ARL0={arl0}) ===\n")
    t0 = time.time()
    st = build_regime_state(arl0=arl0, verbose=True)

    hits = st[st["detected"]].copy()
    hits["tau_hat"] = hits["break_date"]

    print(f"\n{'detected':<12} {'~tau_hat':<12} {'delay':>6}  {'sigma_ann':>9}  crisis window")
    print("-" * 66)
    for date, row in hits.iterrows():
        print(f"{str(date.date()):<12} {str(pd.Timestamp(row['tau_hat']).date()):<12} "
              f"{int(row['days_since_break']):>6}  {row['sigma_ann']:>9.1%}  {label_crisis(date)}")

    print(f"\nregimes             : {int(st['regime_id'].iloc[-1]) + 1}")
    print(f"days label=crisis    : {st['label'].mean():.1%}")
    print(f"mean p_crisis        : {st['p_crisis'].mean():.3f}")
    print(f"p_crisis > 0.5       : {(st['p_crisis'] > 0.5).mean():.1%} of days")

    out_path = write_regime_parquet(st)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    st.to_csv(RESULTS_DIR / f"regime_state_arl{arl0}.csv")
    print(f"\nsaved  {out_path}")
    print(f"saved  {RESULTS_DIR / f'regime_state_arl{arl0}.csv'}")

    print(f"\ndone in {time.time() - t0:.1f}s")
    print("\nwire into xgb.py:  DETECTOR = 'changepoint', then in regime_probs():")
    print("    from _regimes.changepoint.main import crisis_probs")
    print("    return crisis_probs(dates)")
    print("\nsensitivity:  for a in 500 1000 5000; do python main.py --arl0 $a; done")


if __name__ == "__main__":
    main()
