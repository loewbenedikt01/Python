"""
Extreme Gradient Boosting — cross-sectional return forecast -> portfolio weights.

Walk-forward, refit at every rebalance date `d` on point-in-time data:
  * fixed rolling window (no expansion): TRAINING_MONTHS_XGB months of training,
    an EMBARGO_MONTHS_XGB gap, then a VALIDATION_MONTHS_XGB block ending at `d`;
  * the feature panel is rebuilt per `d` with the cross-section pinned to
    `universe_for(d.year)` — the same ~20 names for every training row;
  * every (grid point, seed) in XGB_GRID x BASE_SEED is fit — no search, no
    selection — and the raw predictions are averaged over the whole ensemble,
    then ranked cross-sectionally and mapped to weights via the config weight
    box.  The validation block only reports ensemble val R^2 / rank IC.

Reported specification count is 1 (the ensemble).  Output tree: _output/xgb/.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import export
from _regimes import regime_def
from config import (
    START_DATE,
    END_DATE,
    MIN_WEIGHT,
    MAX_WEIGHT,
    HORIZON_TRADING_DAYS,
    TRAINING_MONTHS_XGB,
    EMBARGO_MONTHS_XGB,
    VALIDATION_MONTHS_XGB,
    BASE_SEED,
    XGB_FIXED,
    XGB_GRID,
)
from features import load_db, features_panel, feature_cache_stats
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS

# ----
# Variables
# ----

MODEL_NAME   = "xgb_no_t_h_test_cp_w"        # change per run
WINDOW_MODE  = "holdout"         # "holdout" | "latest"

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
regime_def.USE_REGIME_FEATURES = False    # Channel 2: interaction features
regime_def.USE_REGIME_THETA    = False    # Channel 3: rank sharpness

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


# ----
# Weight box  (water-filling projection onto {lo <= w_i <= hi, sum w = 1})
# ----

def _weight_box(n: int) -> tuple[float, float]:
    """
    [MIN_WEIGHT, MAX_WEIGHT] per name, widened only if that box cannot sum to 1
    for n names (n < 10 lifts the ceiling, n > 100 drops the floor).
    """
    lo, hi = MIN_WEIGHT, MAX_WEIGHT
    if n * hi < 1.0:
        hi = 1.0 / n
    if n * lo > 1.0:
        lo = 1.0 / n
    return lo, hi

def _apply_box(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Project long-only weights onto {lo <= w_i <= hi, sum w = 1} by water-filling.
    """
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
    """
    Simple return from each date to h trading days later, aligned at the start date.
    """
    return prices.shift(-h) / prices - 1.0

def _month_firsts(prices: pd.DataFrame, start: str) -> pd.DatetimeIndex:
    cal = prices.loc[start:END_DATE].index
    return cal[~cal.to_period("M").duplicated()]

def _resolve_dates(prices: pd.DataFrame, month_firsts: pd.DatetimeIndex, h: int) -> pd.Series:
    """
    The trading day h steps after each month-start (NaT if it runs off the end).
    """
    idx = prices.index
    pos = idx.get_indexer(month_firsts)
    out = pd.Series(pd.NaT, index=month_firsts, dtype="datetime64[ns]")
    ok = (pos >= 0) & (pos + h < len(idx))
    out.loc[month_firsts[ok]] = idx[pos[ok] + h]
    return out


# ----
# Model
# ----

def _grid() -> list[dict]:
    """
    Every combination in XGB_GRID (full Cartesian product).
    """
    keys = list(XGB_GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*XGB_GRID.values())]


GRID          = _grid()
FITS_PER_DATE = len(GRID) * len(BASE_SEED)


def _fmt(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _make_model(params: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        **XGB_FIXED,
        **params,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )


def _rank_ic(y_true, y_pred, dates) -> float:
    """
    Spearman(pred, realised) per date, averaged.  Dates with <3 names are skipped.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)},
                      index=pd.Index(np.asarray(dates), name="d"))
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby(level="d") if len(g) > 2]
    return float(np.nanmean(ics)) if ics else np.nan


def _fit_one(X_tr, y_tr, params: dict, seed: int, sw=None) -> XGBRegressor:
    m = _make_model(params, seed)
    m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    return m


def _train(X_tr, y_tr, X_val, y_val, month_w=None):
    """
    Fit every (grid point, seed) in GRID x BASE_SEED and keep them all — no
    search, no selection.  Inference averages the raw predictions across the
    whole ensemble (`_predict`); the validation block is used only to report
    ensemble val R^2 / rank IC, never to pick anything.  Every model runs the
    full XGB_FIXED['n_estimators'] rounds (no early stopping), so there is no
    per-model tree count to record.  Returns (models, ensemble val_r2, ensemble
    val_ic, mean feature importance).

    `month_w` (Channel 1): per training-month observation weights, expanded to
    rows and passed through as sample_weight.  All-ones -> passed as None so the
    fit is bit-identical to unweighted training.
    """
    sw = None
    if month_w is not None and not np.allclose(month_w.to_numpy(), 1.0):
        sw = month_w.reindex(X_tr.index.get_level_values("date")).to_numpy()

    models = Parallel(n_jobs=-1, backend="threading")(
        delayed(_fit_one)(X_tr, y_tr, p, seed, sw)
        for seed in BASE_SEED for p in GRID
    )

    if len(y_val):
        y_val_np = y_val.to_numpy()
        sst = float(np.sum((y_val_np - y_val_np.mean()) ** 2)) or np.nan
        val_pred = np.mean([m.predict(X_val) for m in models], axis=0)
        val_r2 = 1.0 - float(np.sum((y_val_np - val_pred) ** 2)) / sst
        val_ic = _rank_ic(y_val_np, val_pred, y_val.index.get_level_values("date"))
    else:
        val_r2 = val_ic = np.nan
    importance = pd.Series(
        np.mean([m.feature_importances_ for m in models], axis=0), index=X_tr.columns
    )
    return models, val_r2, val_ic, importance


def _predict(models: list[XGBRegressor], X) -> np.ndarray:
    """
    Mean of the raw predictions over the whole ensemble.
    """
    return np.mean([m.predict(X) for m in models], axis=0)


def train_start(frequency: str, mode: str = "holdout", seq_len: int = 1,
                buffer: int = 2) -> str:
    """
    Earliest month the training pool must reach so the first rebalance at
    START_DATE has a full window.  buffer absorbs calendar edge effects.
    """
    label_m = max(1, HORIZON_TRADING_DAYS[frequency] // 21)
    if mode == "latest":
        back = TRAINING_MONTHS_XGB + label_m
    else:
        back = TRAINING_MONTHS_XGB + EMBARGO_MONTHS_XGB[frequency] + VALIDATION_MONTHS_XGB
    back += (seq_len - 1) + buffer
    return (pd.Timestamp(START_DATE) - pd.DateOffset(months=back)).strftime("%Y-%m-%d")


# ----
# Target weights
# ----

def xgb_targets(db: pd.DataFrame, prices: pd.DataFrame, frequency: str):
    """
    One XGB target-weight row per rebalance date for `frequency`, plus a dict of
    per-date diagnostics frames.  The feature panel is rebuilt per rebalance
    date `d` with the cross-section pinned to `universe_for(d.year)` — the same
    ~20 names for every training row — so ranks match the set actually traded.
    """
    h = HORIZON_TRADING_DAYS[frequency]
    train_firsts = _month_firsts(prices, train_start(frequency, WINDOW_MODE))                 # training pool
    reb_firsts   = _month_firsts(prices, START_DATE)                  # investment horizon
    reb_dates = reb_firsts[reb_firsts.month.isin(REBALANCE_MONTHS[frequency])]

    fwd = _forward_returns(prices, h)
    fwd_stack = fwd.stack()
    fwd_stack.index = fwd_stack.index.set_names(["date", "ticker"])
    fwd_stack = fwd_stack - fwd_stack.groupby(level="date").transform("mean")
    resolve = _resolve_dates(prices, train_firsts, h)
    regime = _regime_label(train_firsts) if REGIME is not None else None
    min_tr = 24 if REGIME is not None else TRAINING_MONTHS_XGB
    _crisis_state["on"] = False
    _regfeat_log["done"] = False

    def _slice(panel, panel_dates, months):
        sub = panel[panel_dates.isin(months)]
        y = fwd_stack.reindex(sub.index)
        return sub[y.notna()], y.dropna()

    rows:  dict[pd.Timestamp, pd.Series] = {}
    preds: dict[pd.Timestamp, pd.Series] = {}
    imp_rows: dict[pd.Timestamp, pd.Series] = {}
    r2_sel:   dict[pd.Timestamp, float] = {}
    ic_val:   dict[pd.Timestamp, float] = {}
    win_log:  list[tuple] = []
    reject:   dict[pd.Timestamp, str] = {}

    # ---- pass 1: which rebalance dates will actually train?
    todo: list[tuple] = []
    for d in reb_dates:
        keep = ((train_firsts < d)
                & resolve.reindex(train_firsts).lt(d).to_numpy())
        if REGIME is not None:
            keep = keep & regime.reindex(train_firsts).eq(REGIME).to_numpy()
        pit = train_firsts[keep]

        if WINDOW_MODE == "latest":
            tr_months = pit[-TRAINING_MONTHS_XGB:]
            va_months = pit[:0]
        else:
            pm       = pit.to_period("M")
            val_lo   = d.to_period("M") - VALIDATION_MONTHS_XGB
            train_hi = val_lo - EMBARGO_MONTHS_XGB[frequency]
            train_lo = train_hi - TRAINING_MONTHS_XGB
            tr_months = pit[(pm >= train_lo) & (pm < train_hi)]
            va_months = pit[pm >= val_lo]

        win_log.append((d, len(tr_months), len(va_months)))
        if len(tr_months) < min_tr:
            reject[d] = f"train months {len(tr_months)} < {min_tr}"; continue
        if WINDOW_MODE == "holdout" and len(va_months) < 6:
            reject[d] = f"val months {len(va_months)} < 6"; continue
        todo.append((d, tr_months, va_months))

    print(f"[xgb] {frequency}: {len(todo)}/{len(reb_dates)} rebalance dates to fit  "
          f"| {len(GRID)} configs x {len(BASE_SEED)} seeds = {FITS_PER_DATE} models/date "
          f"(all averaged, no selection)  | {len(todo) * FITS_PER_DATE:,} fits total", flush=True)

    # ---- pass 2: fit + select per rebalance date, with live progress / ETA
    t_start = time.time()
    for i, (d, tr_months, va_months) in enumerate(todo, 1):
        uni_year = universe_for(d.year)
        want = tr_months.union(va_months).union(pd.DatetimeIndex([d]))
        try:
            panel = features_panel(db, want, universe=uni_year)
        except ValueError:
            reject[d] = "no feature rows in window"
            continue
        panel = _add_regime_features(panel)
        feat_cols = list(panel.columns)
        panel_dates = panel.index.get_level_values("date")
        if d not in panel_dates:
            reject[d] = "no feature panel row at d"
            continue

        X_tr, y_tr = _slice(panel, panel_dates, tr_months)
        X_va, y_va = _slice(panel, panel_dates, va_months)
        if len(y_tr) < 100 or (WINDOW_MODE == "holdout" and len(y_va) < 30):
            reject[d] = f"rows train {len(y_tr)} / val {len(y_va)}"
            continue

        uni = [t for t in uni_year if t in panel.loc[d].index]
        if len(uni) < 2:
            reject[d] = f"universe {len(uni)} < 2"
            continue

        month_w = _obs_weights(tr_months, d)
        n_eff = (float(month_w.sum() ** 2 / (month_w ** 2).sum())
                 if month_w is not None else float(len(tr_months)))

        t0 = time.time()
        models, val_r2, val_ic, importance = _train(X_tr, y_tr, X_va, y_va, month_w)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i)

        th = _theta(d)
        Xd = panel.loc[d].reindex(uni)[feat_cols]
        pred = pd.Series(_predict(models, Xd), index=uni)
        r = pred.rank(pct=True).to_numpy()
        w = _apply_box(r if th == 1.0 else r ** th, *_weight_box(len(uni)))

        rows[d] = pd.Series(w, index=uni)
        preds[d] = pred
        imp_rows[d] = importance
        r2_sel[d] = val_r2
        ic_val[d] = val_ic

        reg_tag = ""
        if month_w is not None:
            reg_tag = f" n_eff={n_eff:.0f}/{len(tr_months)}"
        if th != 1.0:
            reg_tag += f" theta={th:.2f}"
        print(f"[xgb] {frequency} {d.date()}  {i:>3}/{len(todo)}  "
              f"train={len(tr_months)}mo/{len(y_tr)}r val={len(va_months)}mo names={len(uni)}  "
              f"{len(models)} models  {_fmt(dt)}  valIC={val_ic:+.3f}{reg_tag}  "
              f"elapsed {_fmt(elapsed)} / ETA {_fmt(eta)}", flush=True)

    targets = pd.DataFrame(rows).T

    first_solved = min(rows) if rows else None
    print(f"[xgb] {frequency}: first solved {first_solved.date() if first_solved else None}  "
          f"solved {len(rows)}/{len(reb_dates)}")
    if reject:
        r = pd.Series(reject).sort_index()
        early = r[r.index < (first_solved or r.index.max())]
        if len(early):
            print(f"[xgb] {frequency}: rejected {early.index.min().date()}..{early.index.max().date()} "
                  f"e.g. {early.index[0].date()} -> {early.iloc[0]}; "
                  f"{early.index[-1].date()} -> {early.iloc[-1]}")
    if win_log:
        nv = np.array([v for _, _, v in win_log])
        nt = np.array([t for _, t, _ in win_log])
        ics = np.array(list(ic_val.values()))
        print(f"[xgb] {frequency}: train months {nt.min()}-{nt.max()}, "
              f"val months min/med/max {nv.min()}/{int(np.median(nv))}/{nv.max()}  "
              f"(label overlap ~{max(1, h // 21) - 1}/{max(1, h // 21)})")
        if len(ics) and np.isfinite(ics).any():
            print(f"[xgb] {frequency}: ensemble validation IC min/med/max "
                  f"{np.nanmin(ics):+.3f}/{np.nanmedian(ics):+.3f}/{np.nanmax(ics):+.3f}")

    cs = feature_cache_stats()
    looked = cs["hits"] + cs["misses"]
    if looked:
        print(f"[xgb] {frequency}: feature panel cache {cs['hits']}/{looked} hits "
              f"({cs['hits'] / looked:.0%}), {cs['size']} unique (as-of, universe) frames "
              f"built (cumulative over run)")

    # ── out-of-sample skill
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

    hp_spec = pd.DataFrame(GRID).assign(**XGB_FIXED,
                                        seeds=", ".join(map(str, BASE_SEED)))
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
            f"REGIME = {REGIME!r} (hard-split spec) but DETECTOR = 'none', so regime_probs() "
            "is the stub (p_crisis = 0 everywhere) and every month labels 'calm'. "
            "Pick a real DETECTOR before running a hard split."
        )
    if stub and (USE_REGIME_WEIGHTS or USE_REGIME_FEATURES or USE_REGIME_THETA):
        print("[xgb] NOTE: regime channels enabled but DETECTOR='none' -> stub p_crisis=0; "
              "this run must reproduce the un-regimed result bit-for-bit (regression test).",
              flush=True)

    prices = load_prices()
    db = load_db()

    print(f"[xgb] ensemble (no selection): {len(GRID)} configs x {len(BASE_SEED)} seeds "
          f"= {FITS_PER_DATE} models averaged per rebalance date; panel rebuilt per "
          f"rebalance date on universe_for(year)", flush=True)

    for frequency in FREQUENCIES:
        targets, n_dates, diagnostics = xgb_targets(db, prices, frequency)
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"xgb/{MODEL_NAME}_{frequency.lower()}"

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
