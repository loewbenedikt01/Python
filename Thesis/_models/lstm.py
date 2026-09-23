"""
Long Short Term Memory (LSTM) — cross-sectional return forecast -> portfolio weights.

Walk-forward, refit at every rebalance date `d` on point-in-time data:
  * fixed rolling window (no expansion): TRAINING_MONTHS_RF months of training,
    an EMBARGO_MONTHS_RF gap, then a VALIDATION_MONTHS_RF block ending at `d`;
  * the feature panel is rebuilt per `d` with the cross-section pinned to
    `universe_for(d.year)` — the same ~20 names for every training row;
  * every (grid point, seed) in RF_GRID x BASE_SEED is fit — no search, no
    selection — and the raw predictions are averaged over the whole ensemble,
    then ranked cross-sectionally and mapped to weights via the config weight
    box.  The validation block only reports ensemble val R^2 / rank IC.

Reported specification count is 1 (the ensemble).  Output tree: _output/rf/.
"""

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from pathlib import Path
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # before importing tensorflow
 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras import layers, models, regularizers
 
sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))
sys.path.append(str(Path(__file__).resolve().parents[1]))
 
from _regimes import regime_def
import export
from config import (
    START_DATE,
    END_DATE,
    MIN_WEIGHT,
    MAX_WEIGHT,
    HORIZON_TRADING_DAYS,
    TRAINING_MONTHS_LSTM,
    EMBARGO_MONTHS_LSTM,
    VALIDATION_MONTHS_LSTM,
    BASE_SEED_LSTM,
    LSTM_FIXED,
    LSTM_GRID,
)
from features import load_db, features_panel, feature_cache_stats
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS
 
 

# ----
# Variables
# ----

MODEL_NAME   = "LSTM_no_t_h"        # change per run
WINDOW_MODE  = "holdout"            # "holdout" | "latest"
SEQ_LEN     = int(LSTM_FIXED["seq_len"])
SEEDS       = list(BASE_SEED_LSTM)   # LSTM seed variance is large
N_WORKERS   = max(1, (os.cpu_count() or 2) // 2)   # parallel rebalance dates
TF_THREADS  = 2                                     # per worker; keep fixed (determinism)
 
FREQUENCIES = [
    "Monthly",
    #"Quarterly",
    #"Yearly",
]
 
 
# ----
# Regime implementation
# ----
 
regime_def.DETECTOR            = "none"   # "none" | "hmm" | "wasserstein" | "changepoint"
regime_def.REGIME              = None     # None | "calm" | "crisis"
regime_def.USE_REGIME_WEIGHTS  = False    # Channel 1: sample weights
regime_def.USE_REGIME_FEATURES = True     # Channel 2: interaction features
regime_def.USE_REGIME_THETA    = True     # Channel 3: rank sharpness
 
# LSTM-only extra: raw p_crisis as an input channel per timestep.  A neural net
# can learn interactions itself and p_crisis varies along the sequence.  Only
# active with a real detector: even an all-zero column would change the weight
# initialisation draws and break the stub regression test.
USE_PCRISIS_INPUT = True
 
DETECTOR             = regime_def.DETECTOR
REGIME               = regime_def.REGIME
USE_REGIME_WEIGHTS   = regime_def.USE_REGIME_WEIGHTS
USE_REGIME_FEATURES  = regime_def.USE_REGIME_FEATURES
USE_REGIME_THETA     = regime_def.USE_REGIME_THETA
 
regime_probs         = regime_def.regime_probs
_regime_label        = regime_def._regime_label
_obs_weights         = regime_def._obs_weights
_add_regime_features = regime_def._add_regime_features
_theta               = regime_def._theta
_crisis_state        = regime_def._crisis_state
_regfeat_log         = regime_def._regfeat_log
 
 
def _add_pcrisis(panel: pd.DataFrame) -> pd.DataFrame:
    if not USE_PCRISIS_INPUT or DETECTOR == "none":
        return panel
    dates = panel.index.get_level_values("date")
    pc = regime_probs(dates.unique())["p_crisis"].reindex(dates).to_numpy()
    return panel.assign(p_crisis=pc)
 
 
# ----
# Weight box  (water-filling projection onto {lo <= w_i <= hi, sum w = 1})
# ----
 
def _weight_box(n: int) -> tuple[float, float]:
    lo, hi = MIN_WEIGHT, MAX_WEIGHT
    if n * hi < 1.0:
        hi = 1.0 / n
    if n * lo > 1.0:
        lo = 1.0 / n
    return lo, hi
 
def _apply_box(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    for _ in range(100):
        lo_hit, hi_hit = w < lo - 1e-12, w > hi + 1e-12
        if not lo_hit.any() and not hi_hit.any():
            break
        w[lo_hit], w[hi_hit] = lo, hi
        free = ~(lo_hit | hi_hit)
        slack = 1.0 - w[~free].sum()
        if free.any() and w[free].sum() > 0 and slack > 0:
            w[free] *= slack / w[free].sum()
        else:
            break
    return w / w.sum()
 
 
# ----
# Labels / calendar
# ----
 
def _forward_returns(prices: pd.DataFrame, h: int) -> pd.DataFrame:
    return prices.shift(-h) / prices - 1.0
 
def _month_firsts(prices: pd.DataFrame, start: str) -> pd.DatetimeIndex:
    cal = prices.loc[start:END_DATE].index
    return cal[~cal.to_period("M").duplicated()]
 
def _resolve_dates(prices: pd.DataFrame, month_firsts: pd.DatetimeIndex, h: int) -> pd.Series:
    idx = prices.index
    pos = idx.get_indexer(month_firsts)
    out = pd.Series(pd.NaT, index=month_firsts, dtype="datetime64[ns]")
    ok = (pos >= 0) & (pos + h < len(idx))
    out.loc[month_firsts[ok]] = idx[pos[ok] + h]
    return out
 
def train_start(frequency: str, mode: str = "holdout", seq_len: int = 1,
                buffer: int = 2) -> str:
    """Earliest month the pool must reach so the first rebalance at START_DATE
    has a full window plus seq_len-1 months of sequence history."""
    label_m = max(1, HORIZON_TRADING_DAYS[frequency] // 21)
    if mode == "latest":
        back = TRAINING_MONTHS_LSTM + label_m
    else:
        back = TRAINING_MONTHS_LSTM + EMBARGO_MONTHS_LSTM[frequency] + VALIDATION_MONTHS_LSTM
    back += (seq_len - 1) + buffer
    return (pd.Timestamp(START_DATE) - pd.DateOffset(months=back)).strftime("%Y-%m-%d")
 
 
# ----
# Model
# ----
 
def _grid() -> list[dict]:
    keys = list(LSTM_GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*LSTM_GRID.values())]
 
 
GRID          = _grid()
FITS_PER_DATE = len(GRID) * len(SEEDS)
 
 
def _fmt(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
 
 
def _make_model(units: int, dropout: float, n_features: int) -> tf.keras.Model:
    """LSTM(units) -> Dropout -> Dense(units//2, relu) -> Dense(1).
    No BatchNorm: batches mix stocks and dates and samples are few."""
    reg = regularizers.l2(float(LSTM_FIXED["l2"]))
    m = models.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(int(units), kernel_regularizer=reg, recurrent_regularizer=reg),
        layers.Dropout(float(dropout)),
        layers.Dense(max(int(units) // 2, 4), activation="relu", kernel_regularizer=reg),
        layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(LSTM_FIXED["lr"])),
              loss=tf.keras.losses.Huber(delta=1.0))   # delta = 1 sd, target is scaled
    return m
 
 
def _rank_ic(y_true, y_pred, dates) -> float:
    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)},
                      index=pd.Index(np.asarray(dates), name="d"))
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby(level="d") if len(g) > 2]
    return float(np.nanmean(ics)) if ics else np.nan
 
 
def _by_date(panel: pd.DataFrame, cols: list) -> dict:
    """{month-start: ticker x cols frame} -- built once per rebalance date."""
    return {d: g.droplevel("date")[cols] for d, g in panel.groupby(level="date")}
 
 
def _build_sequences(by_date: dict, target_months, cols: list):
    """
    One sample per (t, ticker): the SEQ_LEN consecutive month-starts ending at t.
    Rank features are centred to (x - 0.5); interaction columns are already
    centred.  A window with a missing month is skipped, never mis-spaced.
    Returns X (n, SEQ_LEN, F) float32 and a (date, ticker) MultiIndex.
    """
    dates = pd.DatetimeIndex(sorted(by_date))
    pos = {d: i for i, d in enumerate(dates)}
    month_no = dates.year * 12 + dates.month
    center = np.array([0.0 if c.endswith("_x_crisis") else 0.5 for c in cols],
                      dtype="float32")
    X, keys = [], []
    for t in pd.DatetimeIndex(target_months):
        i = pos.get(t)
        if i is None or i < SEQ_LEN - 1:
            continue
        if (np.diff(month_no[i - SEQ_LEN + 1 : i + 1]) != 1).any():
            continue
        hist = dates[i - SEQ_LEN + 1 : i + 1]
        frames = [by_date[h] for h in hist]
        tick = sorted(set.intersection(*(set(f.index) for f in frames)))
        if not tick:
            continue
        X.append(np.stack([f.loc[tick].to_numpy(dtype="float32") - center
                           for f in frames], axis=1))
        keys += [(t, k) for k in tick]
    if not X:
        return (np.empty((0, SEQ_LEN, len(cols)), dtype="float32"),
                pd.MultiIndex.from_tuples([], names=["date", "ticker"]))
    return (np.concatenate(X).astype("float32"),
            pd.MultiIndex.from_tuples(keys, names=["date", "ticker"]))
 
 
def _fit_predict(X_tr, y_tr, sw, X_va, X_d):
    """
    Fit every (grid point, seed), predict validation and d immediately, discard
    the net.  Target is scaled by its training sd (ranks are invariant to that)
    so Huber(delta=1) means one standard deviation.  Returns ensemble-mean
    validation predictions, d predictions, and gradient attribution |dy/dx|
    averaged over stocks and timesteps (replaces feature_importances_).
    """
    y_sd = float(np.std(y_tr)) or 1.0
    y_fit = (y_tr / y_sd).astype("float32")
    pv, pdd, imp = [], [], []
    for seed in SEEDS:
        for p in GRID:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(int(seed))
            m = _make_model(p["units"], p["dropout"], X_tr.shape[2])
            m.fit(X_tr, y_fit, sample_weight=sw,
                  epochs=int(LSTM_FIXED["epochs"]),
                  batch_size=int(LSTM_FIXED["batch_size"]),
                  shuffle=True, verbose=0)
            if len(X_va):
                pv.append(m(tf.constant(X_va), training=False).numpy().ravel())
            xd = tf.constant(X_d)
            with tf.GradientTape() as tape:
                tape.watch(xd)
                out = m(xd, training=False)
            pdd.append(out.numpy().ravel())
            imp.append(np.abs(tape.gradient(out, xd).numpy()).mean(axis=(0, 1)))
            del m
    val_pred = np.mean(pv, axis=0) * y_sd if pv else np.empty(0)
    return val_pred, np.mean(pdd, axis=0) * y_sd, np.mean(imp, axis=0)
 
 
# ----
# ----
# Worker: one rebalance date, fitted in a fresh process
# ----
# TensorFlow accumulates state across the hundreds of nets built in one
# process, so per-date fit time grows linearly through a run.  Each date is
# therefore fitted in its own short-lived process (max_tasks_per_child=1):
# nothing can accumulate, and independent dates run in parallel.  Only theta
# depends on date order; it is applied afterwards in the main process.
 
_STATE: dict = {}
 
def _worker_init() -> None:
    """Runs first in every worker process, before any TF op."""
    tf.config.threading.set_intra_op_parallelism_threads(TF_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
 
 
def _state(frequency: str):
    """Process-local data: loaded once per worker process."""
    if frequency not in _STATE:
        prices = load_prices()
        db = load_db()
        h = HORIZON_TRADING_DAYS[frequency]
        fs = _forward_returns(prices, h).stack()
        fs.index = fs.index.set_names(["date", "ticker"])
        fs = fs - fs.groupby(level="date").transform("mean")
        train_firsts = _month_firsts(prices, train_start(frequency, WINDOW_MODE, SEQ_LEN))
        _STATE[frequency] = (db, fs, train_firsts)
    return _STATE[frequency]
 
 
def _date_job(frequency: str, d, tr_months, va_months) -> dict:
    """Everything for one rebalance date up to the raw prediction.
    Returns a plain dict (picklable); weights are built in the main process."""
    t0 = time.time()
    db, fwd_stack, train_firsts = _state(frequency)
    tf_months = train_firsts.to_period("M")
    uni_year = universe_for(d.year)
 
    # SEQ_LEN-1 months of feature history before the first training month,
    # through d.  Features only -- no labels -- so this is leak-free.
    lo   = tr_months.min().to_period("M") - (SEQ_LEN - 1)
    want = train_firsts[(tf_months >= lo) & (train_firsts <= d)]
    try:
        panel = features_panel(db, want, universe=uni_year)
    except ValueError:
        return {"d": d, "reject": "no feature rows in window"}
    panel = _add_pcrisis(_add_regime_features(panel))
    cols = list(panel.columns)
    if d not in panel.index.get_level_values("date"):
        return {"d": d, "reject": "no feature panel row at d"}
 
    by_date = _by_date(panel, cols)
    X_tr, k_tr = _build_sequences(by_date, tr_months, cols)
    y_tr = fwd_stack.reindex(k_tr).to_numpy()
    ok = np.isfinite(y_tr)
    X_tr, k_tr, y_tr = X_tr[ok], k_tr[ok], y_tr[ok]
 
    X_va, k_va = _build_sequences(by_date, va_months, cols)
    y_va = fwd_stack.reindex(k_va)
    ok = y_va.notna().to_numpy()
    X_va, k_va, y_va = X_va[ok], k_va[ok], y_va[ok]
 
    X_d, k_d = _build_sequences(by_date, pd.DatetimeIndex([d]), cols)
 
    if len(y_tr) < 100 or (WINDOW_MODE == "holdout" and len(y_va) < 30):
        return {"d": d, "reject": f"seqs train {len(y_tr)} / val {len(y_va)}"}
    if len(k_d) == 0:
        return {"d": d, "reject": "no prediction sequence at d"}
    uni = [t for t in uni_year if t in panel.loc[d].index]
    if len(uni) < 2:
        return {"d": d, "reject": f"universe {len(uni)} < 2"}
 
    month_w = _obs_weights(tr_months, d)
    sw = None
    if month_w is not None and not np.allclose(month_w.to_numpy(), 1.0):
        sw = month_w.reindex(k_tr.get_level_values("date")).to_numpy()
    n_eff = (float(month_w.sum() ** 2 / (month_w ** 2).sum())
             if month_w is not None else np.nan)
 
    val_pred, d_pred, imp = _fit_predict(X_tr, y_tr, sw, X_va, X_d)
 
    if len(y_va):
        yv = y_va.to_numpy()
        sst = float(np.sum((yv - yv.mean()) ** 2)) or np.nan
        val_r2 = 1.0 - float(np.sum((yv - val_pred) ** 2)) / sst
        val_ic = _rank_ic(yv, val_pred, k_va.get_level_values("date"))
    else:
        val_r2 = val_ic = np.nan
 
    # names without a full sequence get the median prediction (middle rank)
    pred = pd.Series(d_pred, index=k_d.get_level_values("ticker")).reindex(uni)
    n_filled = int(pred.isna().sum())
    pred = pred.fillna(pred.median())
 
    return {"d": d, "pred": pred, "imp": pd.Series(imp, index=cols),
            "val_r2": val_r2, "val_ic": val_ic, "n_eff": n_eff, "n_filled": n_filled,
            "n_tr": len(tr_months), "n_va": len(va_months), "n_seq": len(y_tr),
            "dt": time.time() - t0}
 
 
# ----
# Target weights
# ----
 
def lstm_targets(db: pd.DataFrame, prices: pd.DataFrame, frequency: str):
    h = HORIZON_TRADING_DAYS[frequency]
    train_firsts = _month_firsts(prices, train_start(frequency, WINDOW_MODE, SEQ_LEN))
    reb_firsts   = _month_firsts(prices, START_DATE)
    reb_dates    = reb_firsts[reb_firsts.month.isin(REBALANCE_MONTHS[frequency])]
 
    fwd = _forward_returns(prices, h)
    resolve = _resolve_dates(prices, train_firsts, h)
    regime = _regime_label(train_firsts) if REGIME is not None else None
    min_tr = 24 if REGIME is not None else TRAINING_MONTHS_LSTM
    _crisis_state["on"] = False
    _regfeat_log["done"] = False
 
    rows:  dict[pd.Timestamp, pd.Series] = {}
    preds: dict[pd.Timestamp, pd.Series] = {}
    imp_rows: dict[pd.Timestamp, pd.Series] = {}
    r2_sel:   dict[pd.Timestamp, float] = {}
    ic_val:   dict[pd.Timestamp, float] = {}
    n_filled: dict[pd.Timestamp, int] = {}
    win_log:  list[tuple] = []
    reject:   dict[pd.Timestamp, str] = {}
 
    # ---- pass 1: which rebalance dates will train?
    todo: list[tuple] = []
    for d in reb_dates:
        keep = ((train_firsts < d)
                & resolve.reindex(train_firsts).lt(d).to_numpy())
        if REGIME is not None:
            keep = keep & regime.reindex(train_firsts).eq(REGIME).to_numpy()
        pit = train_firsts[keep]
 
        if WINDOW_MODE == "latest":
            tr_months = pit[-TRAINING_MONTHS_LSTM:]
            va_months = pit[:0]
        else:
            pm       = pit.to_period("M")
            val_lo   = d.to_period("M") - VALIDATION_MONTHS_LSTM
            train_hi = val_lo - EMBARGO_MONTHS_LSTM[frequency]
            train_lo = train_hi - TRAINING_MONTHS_LSTM
            tr_months = pit[(pm >= train_lo) & (pm < train_hi)]
            va_months = pit[pm >= val_lo]
 
        win_log.append((d, len(tr_months), len(va_months)))
        if len(tr_months) < min_tr:
            reject[d] = f"train months {len(tr_months)} < {min_tr}"; continue
        if WINDOW_MODE == "holdout" and len(va_months) < 6:
            reject[d] = f"val months {len(va_months)} < 6"; continue
        todo.append((d, tr_months, va_months))
 
    print(f"[lstm] {frequency}: {len(todo)}/{len(reb_dates)} rebalance dates to fit  "
          f"| {len(GRID)} configs x {len(SEEDS)} seeds = {FITS_PER_DATE} nets/date "
          f"| {N_WORKERS} worker processes x {TF_THREADS} threads, one fresh process per date",
          flush=True)
 
    # ---- pass 2a: fit every date in its own fresh process (parallel)
    results: dict = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1,
                             initializer=_worker_init) as ex:
        futs = {ex.submit(_date_job, frequency, d, tr, va): d for d, tr, va in todo}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results[r["d"]] = r
            elapsed = time.time() - t_start
            eta = elapsed / k * (len(todo) - k)
            if "reject" in r:
                msg = f"REJECTED ({r['reject']})"
            else:
                msg = (f"train={r['n_tr']}mo/{r['n_seq']}seq val={r['n_va']}mo  "
                       f"{_fmt(r['dt'])}  valIC={r['val_ic']:+.3f}"
                       + (f" n_eff={r['n_eff']:.0f}" if np.isfinite(r["n_eff"]) else "")
                       + (f" filled={r['n_filled']}" if r["n_filled"] else ""))
            print(f"[lstm] {frequency} {r['d'].date()}  {k:>3}/{len(todo)}  {msg}  "
                  f"elapsed {_fmt(elapsed)} / ETA {_fmt(eta)}", flush=True)
 
    # ---- pass 2b: weights in date order (theta is stateful)
    for d in sorted(results):
        r = results[d]
        if "reject" in r:
            reject[d] = r["reject"]; continue
        pred = r["pred"]
        th = _theta(d)
        rk = pred.rank(pct=True).to_numpy()
        w = _apply_box(rk if th == 1.0 else rk ** th, *_weight_box(len(pred)))
        rows[d] = pd.Series(w, index=pred.index)
        preds[d] = pred
        imp_rows[d] = r["imp"]
        r2_sel[d] = r["val_r2"]
        ic_val[d] = r["val_ic"]
        n_filled[d] = r["n_filled"]
 
    targets = pd.DataFrame(rows).T
 
    first_solved = min(rows) if rows else None
    print(f"[lstm] {frequency}: first solved {first_solved.date() if first_solved else None}  "
          f"solved {len(rows)}/{len(reb_dates)}")
    if reject:
        r = pd.Series(reject).sort_index()
        print(f"[lstm] {frequency}: {len(r)} rejected, e.g. "
              f"{r.index[0].date()} -> {r.iloc[0]}")
    if win_log:
        nv = np.array([v for _, _, v in win_log])
        nt = np.array([t for _, t, _ in win_log])
        ics = np.array(list(ic_val.values()), dtype=float)
        print(f"[lstm] {frequency}: train months {nt.min()}-{nt.max()}, "
              f"val months min/med/max {nv.min()}/{int(np.median(nv))}/{nv.max()}  "
              f"(label overlap ~{max(1, h // 21) - 1}/{max(1, h // 21)})")
        if len(ics) and np.isfinite(ics).any():
            print(f"[lstm] {frequency}: ensemble validation IC min/med/max "
                  f"{np.nanmin(ics):+.3f}/{np.nanmedian(ics):+.3f}/{np.nanmax(ics):+.3f}")
    if n_filled:
        nf = pd.Series(n_filled)
        print(f"[lstm] {frequency}: names median-filled on {int((nf > 0).sum())} dates "
              f"({int(nf.sum())} name-dates)")
 
    # ── out-of-sample skill (realised demeaned within date, matching the target)
    r2_oos, sp_rho, sp_p, dir_acc = {}, {}, {}, {}
    for d, pred in preds.items():
        realized = fwd.loc[d].reindex(pred.index).dropna()
        if len(realized) < 4:
            continue
        realized = realized - realized.mean()
        p, r = pred.reindex(realized.index), realized
        r2_oos[d] = 1.0 - float(np.sum((r - p) ** 2) / np.sum(r ** 2))
        rho, pv = spearmanr(p, r)
        sp_rho[d], sp_p[d] = float(rho), float(pv)
        dir_acc[d] = float(np.mean(np.sign(p) == np.sign(r)))
 
    hp_spec = pd.DataFrame(GRID).assign(**LSTM_FIXED, seeds=", ".join(map(str, SEEDS)))
    hp_spec.index.name = "config"
 
    diagnostics = {
        "hyperparameters":       hp_spec,
        "feature_importance":    pd.DataFrame(imp_rows).T.rename_axis("date"),
        "r2_selected":           pd.Series(r2_sel, name="r2_selected").rename_axis("date"),
        "val_ic":                pd.Series(ic_val, name="val_ic").rename_axis("date"),
        "r2_raw_vs_zero":        pd.Series(r2_oos, name="r2_raw_vs_zero").rename_axis("date"),
        "spearman_p":            pd.DataFrame({"rho": sp_rho, "p": sp_p}).rename_axis("date"),
        "directional_accuracy":  pd.Series(dir_acc, name="directional_accuracy").rename_axis("date"),
    }
    return targets, len(reb_dates), diagnostics
 
 
# ----
# Main Part
# ----
 
def main() -> None:
    stub = (DETECTOR == "none" and
            float(regime_probs(pd.DatetimeIndex([pd.Timestamp(START_DATE)]))["p_crisis"].iloc[0]) == 0.0)
    if REGIME is not None and stub:
        raise NotImplementedError(
            f"REGIME = {REGIME!r} (hard-split spec) but DETECTOR = 'none'. "
            "Pick a real DETECTOR before running a hard split.")
    if stub and (USE_REGIME_WEIGHTS or USE_REGIME_FEATURES or USE_REGIME_THETA):
        print("[lstm] NOTE: regime channels enabled but DETECTOR='none' -> stub; "
              "this run must reproduce the un-regimed result bit-for-bit.", flush=True)
 
    prices = load_prices()
    db = load_db()
 
    print(f"[lstm] ensemble (no selection): {len(GRID)} configs x {len(SEEDS)} seeds "
          f"= {FITS_PER_DATE} nets/date; window={WINDOW_MODE}; seq_len={SEQ_LEN}; "
          f"{N_WORKERS} workers", flush=True)
 
    for frequency in FREQUENCIES:
        targets, n_dates, diagnostics = lstm_targets(db, prices, frequency)
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"lstm/{MODEL_NAME}_{frequency.lower()}"
 
        export.build_report(
            name,
            res.log_returns,
            weights=res.weights,
            rebalance_status=res.rebalance_status,
            ml=True,
            diagnostics={
                "turnover": res.turnover,
                "transaction_costs": res.transaction_costs,
                **diagnostics,
            },
        )
 
        ic = np.nanmean(list(diagnostics["spearman_p"]["rho"])) if len(diagnostics["spearman_p"]) else np.nan
        print(f"{name:40s} {len(res.log_returns):5d} days  "
              f"cum {np.expm1(res.log_returns.sum()):8.1%}  "
              f"avg turnover {res.turnover.iloc[1:].mean():.3f}  "
              f"solved {len(targets)}/{n_dates}  mean IC {ic:+.3f}")
 
 
if __name__ == "__main__":
    main()