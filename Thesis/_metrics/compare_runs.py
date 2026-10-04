"""
Compare the baseline run of one model with its three regime runs.

Reads the folders export.build_report wrote to _output/<MODEL>/ and prints,
per rebalance frequency:
  1. full period (1998-2025): annual return, volatility, Sharpe, Sortino,
     maximum drawdown, Calmar, Ulcer index, average turnover, share of
     rebalances classed as crisis, and the Sharpe difference to the baseline
     with its p-value;
  2. crisis windows (crises.py main_crises): cumulative return peak -> trough
     and peak -> recovery, and maximum drawdown over peak -> recovery.
Both tables are also saved to _output/_comparison/<MODEL>/.

Sharpe difference test: Ledoit & Wolf (2008, J. Empirical Finance 15,
850-859), HAC version (their sec. 3.1).  On daily returns of the two runs over
their common dates, delta method on the first and second moments with a
Newey-West (Bartlett) long-run covariance; two-sided H0: equal Sharpe ratios.
Sharpe here is the same one as in metrics.py: mean over standard deviation of
daily log returns (risk-free rate RISK_FREE_RATE), annualised with sqrt(252).

Usage: set MODEL and PREFIX below and run the file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import OUTPUT_ROOT, RISK_FREE_RATE
from crises import main_crises

# ----
# Variables
# ----

MODEL  = "mvo"                  # output subfolder: "mvo" | "hrp" | "xgb" | "rf" | "lstm"
PREFIX = "mvo_t_10_lw"          # the run's MODEL_NAME without regime tag and frequency
                                # (e.g. "hrp_t_10_lw_ward", or your MODEL_NAME in xgb.py)

RUNS = {                        # label -> regime_def.run_tag() of that run
    "baseline":       "",
    "changepoint":    "_changepoint_vix",
    "hmm":            "_hmm_gspc",
    "wasserstein":    "_wasserstein_gspc_vix",
    "min_var":        "_minvar",          # MVO control run (ALWAYS_MIN_VAR); skipped if absent
}
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
        r = pd.read_csv(regime, index_col=0)
        if "crisis" in r.columns:
            run["crisis"] = r["crisis"].astype(str).str.lower().eq("true")
    return run


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
    x = df.to_numpy() - RISK_FREE_RATE / PERIODS_PER_YEAR
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
        to = run["turnover"]
        row = {
            "ann_return": m["annual_return"],
            "ann_vol":    m["annual_volatility"],
            "sharpe":     m["sharpe_ratio"],
            "sortino":    m["sortino_ratio"],
            "max_dd":     m["maximum_drawdown"],
            "calmar":     m["calmar_ratio"],
            "ulcer":      m["ulcer_index"],
            "turnover":   to.iloc[1:].mean() if to is not None and len(to) > 1 else np.nan,
            "crisis_share": run["crisis"].mean() if run["crisis"] is not None else np.nan,
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


def _fmt_overall(df: pd.DataFrame) -> pd.DataFrame:
    pct = ["ann_return", "ann_vol", "max_dd", "crisis_share"]
    num = ["sharpe", "sortino", "calmar", "ulcer", "turnover", "d_sharpe"]
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

        ov.to_csv(SAVE_DIR / f"{PREFIX}_{freq}_overall.csv")
        cr.to_csv(SAVE_DIR / f"{PREFIX}_{freq}_crises.csv")

    print(f"\nsaved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
