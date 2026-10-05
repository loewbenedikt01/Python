"""
Compare the baseline run of one model with its three regime runs.

Reads the folders export.build_report wrote to _output/<MODEL>/ and prints,
per rebalance frequency:
  1. full period (1998-2025): annual return, volatility, Sharpe, Sortino,
     maximum drawdown, Calmar, Ulcer index, annual turnover, share of
     trading days held in the crisis portfolio, and the Sharpe difference to the baseline
     with its p-value;
  2. crisis windows (crises.py main_crises): cumulative return peak -> trough
     and peak -> recovery, and maximum drawdown over peak -> recovery;
  3. timing test (needs the baseline and the min_var control run): each regime
     run against a static mix that holds the min-variance portfolio for the
     same share of the time, without timing.  The mix's daily simple return is
         (1 - s) * R_baseline + s * R_min_var,   s = crisis share of the run,
     i.e. both portfolios held side by side in fixed proportion (rebalanced
     daily, no extra trading costs, which slightly favours the mix).  A regime
     run that beats its mix gains from *when* it switches, not from how often.
All tables are also saved to _output/_comparison/<MODEL>/.

Sharpe difference test: Ledoit & Wolf (2008, J. Empirical Finance 15,
850-859), HAC version (their sec. 3.1).  On daily returns of the two runs over
their common dates, delta method on the first and second moments with a
Newey-West (Bartlett) long-run covariance; two-sided H0: equal Sharpe ratios.
Sharpe here is the same one as in metrics.py: mean over standard deviation of
daily log returns in excess of the risk-free rate (risk_free.rf_daily, the
3-month T-bill unless RISK_FREE = "zero"), annualised with sqrt(252).

Usage: set MODEL and PREFIX below and run the file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import OUTPUT_ROOT
from risk_free_rate import rf_daily
from crises import main_crises

# ----
# Variables
# ----

MODEL  = "mvo"                  # output subfolder: "mvo" | "hrp" | "xgb" | "rf" | "lstm"
PREFIX = "mvo_t_10_lw"

RUNS = { 
    "baseline":         "",
    "changepoint":      "_changepoint_vix",
    "hmm":              "_hmm_gspc",
    "wasserstein":      "_wasserstein_gspc_vix",
    "changepoint_trig": "_changepoint_vix_trig",
    "hmm_trig":         "_hmm_gspc_trig",
    "wasserstein_trig": "_wasserstein_gspc_vix_trig",
    "min_var":          None,
}
# Always-min-variance control (mvo.py, ALWAYS_MIN_VAR = True).  MVO and HRP use
# the same covariance and weight box, so the MVO control serves both; for the
# ML models it is a plain reference row.  None = leave it out.
MIN_VAR_RUN = "mvo/mvo_t_10_lw_minvar"
BASELINE    = "baseline"                  # set to "min_var" to test the regime runs against it
FREQUENCIES = ["monthly", "quarterly", "yearly"]     # missing folders are skipped

CRISIS_NAMES = {                # crises.py labels -> short names for the table
    "crisis_1": "Dotcom",
    "crisis_2": "GFC",
    "crisis_3": "2018 Q4",
    "crisis_4": "Covid",
    "crisis_5": "2022 rates",
    "crisis_6": "2025 tariffs",
}

PERIODS_PER_YEAR = 252
SAVE_DIR = Path(OUTPUT_ROOT) / "_comparison" / MODEL


# ----
# Reading one run
# ----

def _read_series(path: Path, col: str | None = None) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[col] if col else df.iloc[:, 0]


def load_run(folder: Path) -> dict | None:
    if not folder.exists():
        return None
    raw, met = folder / "raw", folder / "metrics"
    run = {
        "returns":  _read_series(raw / "daily_returns.csv", "log_return"),
        "metrics":  pd.read_csv(met / "metrics.csv", index_col="period"),
        "turnover": _read_series(raw / "turnover.csv"),
        "crisis":   None,
    }
    regime = raw / "regime.csv"
    if regime.exists():
        r = pd.read_csv(regime, index_col=0, parse_dates=True)
        if "crisis" in r.columns:
            run["crisis"] = r["crisis"].astype(str).str.lower().eq("true")
    return run


def _crisis_share(run: dict) -> float:
    """Share of trading days held in the crisis portfolio: each decision is
    carried forward to the next one, so triggered (off-calendar) rebalances
    are weighted by how long they were held."""
    c, r = run["crisis"], run["returns"]
    if c is None or r is None:
        return np.nan
    c = c[~c.index.duplicated()].sort_index().astype(float)
    return float(c.reindex(r.index, method="ffill").dropna().mean())


def _turnover_pa(run: dict) -> float:
    """Annual turnover: sum over rebalances (initial deployment excluded) per
    year, so runs with different numbers of rebalances compare directly."""
    to, r = run["turnover"], run["returns"]
    if to is None or len(to) < 2 or r is None or len(r) < 2:
        return np.nan
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float(to.iloc[1:].sum() / years)


# ----
# Ledoit-Wolf (2008) Sharpe difference test, HAC version
# ----

def _newey_west(Y: np.ndarray, lags: int) -> np.ndarray:
    """Long-run covariance of the rows of Y (demeaned), Bartlett kernel."""
    T = len(Y)
    S = Y.T @ Y / T
    for j in range(1, lags + 1):
        G = Y[j:].T @ Y[:-j] / T
        S += (1.0 - j / (lags + 1)) * (G + G.T)
    return S


def sharpe_test(r1: pd.Series, r2: pd.Series) -> tuple[float, float]:
    """
    Annualised Sharpe(r1) - Sharpe(r2) and the two-sided p-value of
    Ledoit & Wolf (2008), on the dates both series share.
    """
    df = pd.concat([r1, r2], axis=1, join="inner").dropna()
    x = df.to_numpy() - rf_daily(df.index).to_numpy()[:, None]      # excess returns
    T = len(x)
    mu = x.mean(axis=0)
    gam = (x ** 2).mean(axis=0)
    a, b = mu
    c, d = gam
    sr1 = a / np.sqrt(c - a ** 2)
    sr2 = b / np.sqrt(d - b ** 2)
    grad = np.array([c / (c - a ** 2) ** 1.5,
                     -d / (d - b ** 2) ** 1.5,
                     -0.5 * a / (c - a ** 2) ** 1.5,
                     0.5 * b / (d - b ** 2) ** 1.5])
    Y = np.column_stack([x[:, 0] - a, x[:, 1] - b, x[:, 0] ** 2 - c, x[:, 1] ** 2 - d])
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    var = float(grad @ _newey_west(Y, lags) @ grad / T)
    diff = sr1 - sr2
    p = 2 * norm.sf(abs(diff) / np.sqrt(var)) if var > 0 else np.nan
    return float(diff * np.sqrt(PERIODS_PER_YEAR)), float(p)


# ----
# Tables
# ----

def overall_table(runs: dict) -> pd.DataFrame:
    base = runs.get(BASELINE)
    rows = {}
    for label, run in runs.items():
        m = run["metrics"].loc["overall"]
        row = {
            "ann_return": m["annual_return"],
            "ann_vol":    m["annual_volatility"],
            "sharpe":     m["sharpe_ratio"],
            "sortino":    m["sortino_ratio"],
            "max_dd":     m["maximum_drawdown"],
            "calmar":     m["calmar_ratio"],
            "ulcer":      m["ulcer_index"],
            "turnover_pa":  _turnover_pa(run),
            "crisis_share": _crisis_share(run),
            "d_sharpe":   np.nan,
            "p_value":    np.nan,
        }
        if base is not None and label != BASELINE:
            row["d_sharpe"], row["p_value"] = sharpe_test(run["returns"], base["returns"])
        rows[label] = row
    return pd.DataFrame(rows).T


def crisis_table(runs: dict) -> pd.DataFrame:
    rows = {}
    for c in main_crises:
        name = CRISIS_NAMES.get(c["label"], c["label"])
        for what, period, col in (("peak->trough", "ptt", "cumulative_return"),
                                  ("peak->recovery", "full", "cumulative_return"),
                                  ("max DD", "full", "maximum_drawdown")):
            key = f"{c['label']}_{period}"
            rows[(name, what)] = {label: run["metrics"].loc[key, col]
                                  if key in run["metrics"].index else np.nan
                                  for label, run in runs.items()}
    out = pd.DataFrame(rows).T
    out.index.names = ["crisis", "measure"]
    return out


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    x = r - rf_daily(r.index)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def _max_dd(r: pd.Series) -> float:
    level = np.exp(np.concatenate([[0.0], r.dropna().cumsum().to_numpy()]))
    return float((level / np.maximum.accumulate(level) - 1.0).min())


def timing_table(runs: dict) -> pd.DataFrame | None:
    if "baseline" not in runs or "min_var" not in runs:
        return None
    both = pd.concat([runs["baseline"]["returns"], runs["min_var"]["returns"]],
                     axis=1, join="inner").dropna()
    R_base, R_mv = np.expm1(both.iloc[:, 0]), np.expm1(both.iloc[:, 1])
    rows = {}
    for label, run in runs.items():
        if label in ("baseline", "min_var") or run["crisis"] is None:
            continue
        s = _crisis_share(run)
        mix = np.log1p((1.0 - s) * R_base + s * R_mv)
        r = run["returns"].reindex(mix.index)
        d, p = sharpe_test(r, mix)
        rows[label] = {"min_var_share": s,
                       "sharpe_run": _sharpe(r), "sharpe_mix": _sharpe(mix),
                       "d_sharpe": d, "p_value": p,
                       "max_dd_run": _max_dd(r), "max_dd_mix": _max_dd(mix)}
    return pd.DataFrame(rows).T if rows else None


def _fmt_overall(df: pd.DataFrame) -> pd.DataFrame:
    pct = ["ann_return", "ann_vol", "max_dd", "crisis_share"]
    num = ["sharpe", "sortino", "calmar", "ulcer", "turnover_pa", "d_sharpe"]
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        v = df[c].astype(float)
        if c in pct:
            out[c] = v.map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
        elif c in num:
            out[c] = v.map(lambda x: "" if pd.isna(x) else f"{x:+.3f}" if c == "d_sharpe" else f"{x:.3f}")
        else:
            out[c] = v.map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


# ----
# Main Part
# ----

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    for freq in FREQUENCIES:
        runs, missing = {}, []
        for label, tag in RUNS.items():
            if label == "min_var":
                if not MIN_VAR_RUN:
                    continue
                folder = Path(OUTPUT_ROOT) / f"{MIN_VAR_RUN}_{freq}"
            else:
                folder = Path(OUTPUT_ROOT) / MODEL / f"{PREFIX}{tag}_{freq}"
            run = load_run(folder)
            if run is None:
                missing.append(folder.name)
            else:
                runs[label] = run
        if not runs:
            continue

        print(f"\n=== {MODEL} / {PREFIX} / {freq} ===")
        if missing:
            print("  missing: " + ", ".join(missing))

        ov = overall_table(runs)
        cr = crisis_table(runs)
        print("\nFull period (d_sharpe, p_value: Ledoit-Wolf 2008 test against the baseline)")
        print(_fmt_overall(ov).to_string())
        print("\nCrisis windows (crises.py)")
        print(cr.map(lambda x: "" if pd.isna(x) else f"{x:.2%}").to_string())

        tt = timing_table(runs)
        if tt is not None:
            print("\nTiming test: regime run vs static mix with the same min-variance share")
            out = tt.copy().astype(object)
            for c in ("min_var_share", "max_dd_run", "max_dd_mix"):
                out[c] = tt[c].map(lambda x: f"{x:.2%}")
            for c in ("sharpe_run", "sharpe_mix", "p_value"):
                out[c] = tt[c].map(lambda x: f"{x:.3f}")
            out["d_sharpe"] = tt["d_sharpe"].map(lambda x: f"{x:+.3f}")
            print(out.to_string())
            tt.to_csv(SAVE_DIR / f"{PREFIX}_{freq}_timing.csv")

        ov.to_csv(SAVE_DIR / f"{PREFIX}_{freq}_overall.csv")
        cr.to_csv(SAVE_DIR / f"{PREFIX}_{freq}_crises.csv")

    print(f"\nsaved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
