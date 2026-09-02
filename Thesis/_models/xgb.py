"""
Extreme Gradient Boosting — cross-sectional return forecast -> portfolio weights.

At each rebalance the forecast horizon equals the rebalance interval (Monthly
21d, Quarterly 63d, Yearly 252d).  The model is a gradient-boosted regression
tree that predicts the *date-demeaned* forward return of every investable name
(fwd_t,i minus the cross-sectional mean on date t) from the feature panel in
``_metrics/features.py``; the predictions are ranked cross-sectionally and
turned into weights (weight proportional to predicted rank, all names held),
projected onto the config weight box [MIN_WEIGHT, MAX_WEIGHT] long-only / fully
invested.  Demeaning strips the market level the ranking step would discard
anyway while keeping the (regime-dependent) cross-sectional dispersion.

Training draws on price history back to TRAIN_START (1990) even though the
backtest and every reported number start at START_DATE (1998) — the extra
years are training-only and never enter the return series.

Walk-forward, refit at every rebalance date `d`.  Everything below happens
strictly before `d` on point-in-time data:
  1. expanding window: training = every month-start from TRAIN_START up to
     `d - VAL_MONTHS_XGB - embargo`; validation = the newest VAL_MONTHS_XGB
     months before `d`.  embargo = ceil(h / 21) months drops training months
     whose label horizon would overlap the validation block;
  2. a month-start `t` enters either set only once its forward label has fully
     resolved before `d` (no look-ahead) — for the 252-day horizon this also
     trims roughly the most recent year off the validation block;
  3. fit every (config, seed) in XGB_CONFIGS x BASE_SEED on XGB_FIXED — no
     search, no selection — and average all predictions.  The validation block
     is used only for early stopping (EARLY_STOPPING rounds), which adapts each
     model's tree count.  Reported specification count = 1;
  4. predict the cross-section as of `d`, rank -> weights -> weight box.

Regimes
-------
`REGIME` selects which regime the model is trained / run for; `regime_of(dates)`
maps each date to a regime in {0, 1, 2} and training rows are filtered to
`regime_of(t) == REGIME`.  `regime_of` is a stub returning 0 for every date —
build the real 3-regime classifier there later.  With the stub, only REGIME = 0
is valid; any other value raises until the classifier exists.

The shared portfolio engine handles the actual rebalancing, drift between
rebalances, turnover and costs.  Output tree: _output/xgb/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1] / "_metrics"))

import export
from config import (
    START_DATE,
    END_DATE,
    MIN_WEIGHT,
    MAX_WEIGHT,
    HORIZON_TRADING_DAYS,
    TRAIN_MONTHS_XGB,
    VAL_MONTHS_XGB,
    EARLY_STOPPING,
    BASE_SEED,
    XGB_FIXED,
    XGB_CONFIGS,
)
from features import load_db, features_panel
from portfolio import build_portfolio, load_prices, universe_for, REBALANCE_MONTHS

# ----
# Variables
# ----

MODEL_NAME  = "xgb_no_trans"        # change per run
REGIME      = 0                     # 0 | 1 | 2  — see regime_of()
TRAIN_START = "1990-01-01"          # training history start

MIN_TRAIN_MONTHS = TRAIN_MONTHS_XGB      # expanding window won't fit until this many

# At every rebalance date the model is refit from scratch on an expanding
# window.  There is NO hyperparameter search: every (config, seed) in
# XGB_CONFIGS x BASE_SEED is fit and all predictions are averaged.  The
# validation block is used only for early stopping, never for selection — so the
# reported specification count is one (the ensemble), not |configs| trials per
# rebalance.  Early stopping adapts each model's tree count within XGB_FIXED's
# n_estimators cap.

FREQUENCIES = [
    #"Monthly",
    #"Quarterly",
    "Yearly",
]


# ----
# Regime hook
# ----

def regime_of(dates) -> pd.Series:
    """
    Map each date to a regime label in {0, 1, 2}.  Placeholder: everything is
    regime 0.  Replace with the real 3-regime classifier later.
    """
    return pd.Series(0, index=pd.DatetimeIndex(dates), name="regime")


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
    """Project long-only weights onto {lo <= w_i <= hi, sum w = 1} by water-filling."""
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
    """Simple return from each date to h trading days later, aligned at the start date."""
    return prices.shift(-h) / prices - 1.0


def _month_firsts(prices: pd.DataFrame, start: str) -> pd.DatetimeIndex:
    cal = prices.loc[start:END_DATE].index
    return cal[~cal.to_period("M").duplicated()]


def _resolve_dates(prices: pd.DataFrame, month_firsts: pd.DatetimeIndex, h: int) -> pd.Series:
    """The trading day h steps after each month-start (NaT if it runs off the end)."""
    idx = prices.index
    pos = idx.get_indexer(month_firsts)
    out = pd.Series(pd.NaT, index=month_firsts, dtype="datetime64[ns]")
    ok = (pos >= 0) & (pos + h < len(idx))
    out.loc[month_firsts[ok]] = idx[pos[ok] + h]
    return out


# ----
# Model
# ----

MODEL_JOBS = [(cfg, seed) for cfg in XGB_CONFIGS for seed in BASE_SEED]


def _make_model(cfg: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        **XGB_FIXED,
        **cfg,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,                       # parallelism is across models (joblib), not within
        random_state=seed,
        early_stopping_rounds=EARLY_STOPPING,
    )


def _rank_ic(y_true, y_pred, dates) -> float:
    """Spearman(pred, realised) per date, averaged.  Dates with <3 names are skipped."""
    df = pd.DataFrame({"y": np.asarray(y_true, float),
                       "p": np.asarray(y_pred, float)},
                      index=pd.Index(np.asarray(dates), name="d"))
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby(level="d") if len(g) > 2]
    return float(np.nanmean(ics)) if ics else np.nan


def _fit_one(X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> XGBRegressor:
    m = _make_model(cfg, seed)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def _train(X_tr, y_tr, X_val, y_val):
    """
    Fit every (config, seed) in XGB_CONFIGS x BASE_SEED — no search, no argmax —
    and average all predictions.  The validation block drives early stopping
    only.  Returns (models, per-model param rows, ensemble val_r2, ensemble
    val_ic, mean feature importance).
    """
    y_val_np  = y_val.to_numpy()
    val_dates = y_val.index.get_level_values("date")
    sst = float(np.sum((y_val_np - y_val_np.mean()) ** 2)) or np.nan

    models = Parallel(n_jobs=-1, backend="threading")(
        delayed(_fit_one)(X_tr, y_tr, X_val, y_val, cfg, seed) for cfg, seed in MODEL_JOBS
    )
    params = [{**cfg, "seed": seed, "n_trees": int(m.best_iteration) + 1}
              for m, (cfg, seed) in zip(models, MODEL_JOBS)]

    val_pred = np.mean([m.predict(X_val) for m in models], axis=0)
    val_r2 = 1.0 - float(np.sum((y_val_np - val_pred) ** 2)) / sst
    val_ic = _rank_ic(y_val_np, val_pred, val_dates)
    importance = pd.Series(
        np.mean([m.feature_importances_ for m in models], axis=0), index=X_tr.columns
    )
    return models, params, val_r2, val_ic, importance


def _predict(models: list[XGBRegressor], X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


# ----
# Target weights
# ----

def xgb_targets(panel: pd.DataFrame, prices: pd.DataFrame, frequency: str):
    """
    One XGB target-weight row per rebalance date for `frequency`, plus a dict of
    per-date diagnostics frames.  `panel` is the (date, ticker) feature panel.
    """
    features = list(panel.columns)
    panel_dates = panel.index.get_level_values("date")

    h = HORIZON_TRADING_DAYS[frequency]
    train_firsts = _month_firsts(prices, TRAIN_START)                 # training pool
    reb_firsts   = _month_firsts(prices, START_DATE)                  # investment horizon
    reb_dates = reb_firsts[reb_firsts.month.isin(REBALANCE_MONTHS[frequency])]

    fwd = _forward_returns(prices, h)
    fwd_stack = fwd.stack()
    fwd_stack.index = fwd_stack.index.set_names(["date", "ticker"])
    # target is purely cross-sectional: strip the date-level (market) mean
    fwd_stack = fwd_stack - fwd_stack.groupby(level="date").transform("mean")
    resolve = _resolve_dates(prices, train_firsts, h)
    regime = regime_of(train_firsts)
    embargo = pd.DateOffset(months=int(np.ceil(h / 21)))
    panel_months = pd.DatetimeIndex(np.unique(panel_dates))

    def _slice(months):
        sub = panel[panel_dates.isin(months)]
        y = fwd_stack.reindex(sub.index)
        return sub[y.notna()], y.dropna()

    rows:  dict[pd.Timestamp, pd.Series] = {}
    preds: dict[pd.Timestamp, pd.Series] = {}
    hp_rows:  dict[pd.Timestamp, list] = {}
    imp_rows: dict[pd.Timestamp, pd.Series] = {}
    r2_sel:   dict[pd.Timestamp, float] = {}
    ic_val:   dict[pd.Timestamp, float] = {}
    win_log:  list[tuple] = []          # (d, n_train_months, n_val_months)
    reject:   dict[pd.Timestamp, str] = {}

    for d in reb_dates:
        if d not in panel_dates:
            reject[d] = "no feature panel row"
            continue

        # expanding point-in-time window ending strictly before d:
        #   [TRAIN_START, d - 2y - embargo)  -> training
        #   [d - 2y,      d)                 -> validation / HP tuning
        # a month t joins either set only once resolve[t] < d (label realised
        # before the rebalance).  the embargo drops training months whose label
        # horizon would still overlap the validation block.
        val_lo = d - pd.DateOffset(months=VAL_MONTHS_XGB)

        pit = train_firsts[
            (train_firsts < d)
            & train_firsts.isin(panel_months)                 # month must have feature rows
            & resolve.reindex(train_firsts).lt(d).to_numpy()
            & regime.reindex(train_firsts).eq(REGIME).to_numpy()
        ]
        tr_months = pit[pit < val_lo - embargo]
        va_months = pit[pit >= val_lo]
        win_log.append((d, len(tr_months), len(va_months)))
        if len(tr_months) < MIN_TRAIN_MONTHS:
            reject[d] = f"train months {len(tr_months)} < {MIN_TRAIN_MONTHS}"
            continue
        if len(va_months) < 6:
            reject[d] = f"val months {len(va_months)} < 6"
            continue

        X_tr, y_tr = _slice(tr_months)
        X_va, y_va = _slice(va_months)
        if len(y_tr) < 100 or len(y_va) < 30:
            reject[d] = f"rows train {len(y_tr)} / val {len(y_va)}"
            continue

        uni = [t for t in universe_for(d.year) if t in panel.loc[d].index]
        if len(uni) < 2:
            reject[d] = f"universe {len(uni)} < 2"
            continue

        models, params, val_r2, val_ic, importance = _train(X_tr, y_tr, X_va, y_va)

        Xd = panel.loc[d].reindex(uni)[features]
        pred = pd.Series(_predict(models, Xd), index=uni)

        w = _apply_box(pred.rank(pct=True).to_numpy(), *_weight_box(len(uni)))
        rows[d] = pd.Series(w, index=uni)
        preds[d] = pred
        hp_rows[d] = params
        imp_rows[d] = importance
        r2_sel[d] = val_r2
        ic_val[d] = val_ic

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
        if len(ics):
            print(f"[xgb] {frequency}: ensemble validation IC min/med/max "
                  f"{np.nanmin(ics):+.3f}/{np.nanmedian(ics):+.3f}/{np.nanmax(ics):+.3f}")

    # ── out-of-sample skill, once each horizon has resolved ──
    # realised return is demeaned within the date, matching the training target.
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

    diagnostics = {
        "hyperparameters":       (pd.concat({d: pd.DataFrame(p).rename_axis("model")
                                             for d, p in hp_rows.items()}, names=["date"])
                                  if hp_rows else pd.DataFrame()),
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
    if REGIME != 0 and regime_of(pd.DatetimeIndex([pd.Timestamp(START_DATE)])).eq(0).all():
        raise NotImplementedError(
            f"REGIME = {REGIME} but regime_of() is still the stub (returns 0). "
            "Build the 3-regime classifier before running a non-zero regime."
        )

    prices = load_prices()
    db = load_db()
    # ranks are formed within each year's investable universe, not the full panel
    panel = features_panel(db, _month_firsts(prices, TRAIN_START),
                           universe_fn=lambda d: universe_for(d.year))

    print(f"[xgb] {len(XGB_CONFIGS)} configs x {len(BASE_SEED)} seeds "
          f"= {len(MODEL_JOBS)} models averaged per rebalance date (no search)")

    for frequency in FREQUENCIES:
        targets, n_dates, diagnostics = xgb_targets(panel, prices, frequency)
        res = build_portfolio(targets, frequency=frequency, prices=prices)
        name = f"xgb/{MODEL_NAME}_r{REGIME}_{frequency.lower()}"

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
