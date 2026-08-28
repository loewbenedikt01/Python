
"""
Reporting — the connector between a model's raw output, the metric
definitions (metrics.py) and the crisis windows (crises.py).

Every model built under Thesis/_models/ calls `build_report(...)` once it has
produced a portfolio.  Reporting then writes a fixed folder layout so that all
models stay directly comparable:

    Thesis/_output/<model_name>/
        raw/                        one timeseries per file
            daily_returns.csv           log_return, cum_return, nav
            monthly_value.csv           month-end nav + monthly return
            weights.csv                 date x ticker portfolio weights
            hyperparameters.csv         \
            feature_importance.csv       |
            r2_raw_vs_zero.csv           |
            spearman_p.csv               | pass-through model diagnostics,
            directional_accuracy.csv     | one CSV per key in `diagnostics`
            r2_selected.csv              |
            turnover.csv                 |
            transaction_costs.csv       /
        metrics/
            metrics.csv                 period x (base + crisis-phase) metrics
            metrics_benchmark.csv       same table for the benchmark

The metrics table (see block at the bottom of this file for the original spec):
    rows    = overall + 6 main crises + 3 GFC sub-crises
    columns = sharpe, sortino, calmar, ulcer_index, maximum_drawdown,
              cumulative_return, annualized_return, annualized_volatility,
              beta, alpha (overall only),
              + every peak-to-trough / trough-to-peak / full crisis-phase metric
"""


from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RISK_FREE_RATE,
    OUTPUT_ROOT,
    DATABASE_PATH,
    BENCHMARK_TICKER,
)
from crises import (
    main_crises,
    sub_crises,
    crisis_metrics_ptt,
    crisis_metrics_ttp,
    crisis_metrics_full,
)
from metrics import (
    cumulative_return,
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    ulcer_index,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    beta,
    alpha,
)


# ----
# Constants
# ----
ALL_CRISES      = list(main_crises) + list(sub_crises)
PERIOD_ORDER    = ["overall"] + [c["label"] for c in ALL_CRISES]

# Base per-period metrics, in the column order used across the thesis.
BASE_METRICS    = [
    "sharpe",
    "sortino",
    "calmar",
    "ulcer_index",
    "maximum_drawdown",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
]
BENCHMARK_RELATIVE = ["beta", "alpha"]

# Crisis-phase column names come straight from crises.py so the two files never
# drift apart: ptt_* (peak->trough), ttp_* (trough->recovery), full_* (peak->recovery).
PHASE_COLUMNS   = [
    d["label"]
    for d in (*crisis_metrics_ptt, *crisis_metrics_ttp, *crisis_metrics_full)
]

METRICS_COLUMNS = BASE_METRICS + BENCHMARK_RELATIVE + PHASE_COLUMNS

# Canonical raw-timeseries filenames.  `daily_returns` / `monthly_value` /
# `weights` are built here; the rest are model diagnostics passed through
# `build_report(diagnostics=...)`.
RAW_TIMESERIES  = [
    "daily_returns",
    "monthly_value",
    "weights",
    "hyperparameters",
    "feature_importance",
    "r2_raw_vs_zero",
    "spearman_p",
    "directional_accuracy",
    "r2_selected",
    "turnover",
    "transaction_costs",
]
_DIAGNOSTIC_KEYS = set(RAW_TIMESERIES) - {"daily_returns", "monthly_value", "weights"}


# ----
# Price / return helpers
# ----
def _to_datetime_series(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _nav(log_returns: pd.Series) -> pd.Series:
    """Dated net-asset-value path from daily log returns, anchored one step
    before the first observation at 1.0."""
    return np.exp(log_returns.cumsum())


def _drawdown_price(log_returns: pd.Series) -> pd.Series:
    """Undated price path that *starts at 1.0* before the first return, so a
    segment's drawdown is measured from its own opening level.  Index is a
    plain range – only order matters for cummax-based metrics."""
    levels = np.exp(np.concatenate([[0.0], np.cumsum(log_returns.values)]))
    return pd.Series(levels)


def _slice_returns(log_returns: pd.Series, start, end) -> pd.Series:
    """Returns in (start, end] – the return earned *after* the peak date up to
    and including the trough / recovery date."""
    idx = log_returns.index
    mask = (idx > pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return log_returns.loc[mask]


# ----
# Per-period metrics
# ----
def period_metrics(log_returns: pd.Series, freq: str = "D") -> dict:
    """The 8 base metrics for one return window.  Returns NaNs when the window
    holds fewer than 2 observations (e.g. a crisis that predates the model)."""
    log_returns = log_returns.dropna()
    if len(log_returns) < 2:
        return {m: np.nan for m in BASE_METRICS}

    dd_price = _drawdown_price(log_returns)
    return {
        "sharpe":                sharpe_ratio(log_returns, RISK_FREE_RATE, freq),
        "sortino":               sortino_ratio(log_returns, RISK_FREE_RATE, freq),
        "calmar":                calmar_ratio(log_returns, dd_price, freq),
        "ulcer_index":           ulcer_index(dd_price),
        "maximum_drawdown":      maximum_drawdown(dd_price),
        "cumulative_return":     cumulative_return(log_returns),
        "annualized_return":     annualized_return(log_returns, freq),
        "annualized_volatility": annualized_volatility(log_returns, freq),
    }


def overall_metrics(
    portfolio_log_returns: pd.Series,
    benchmark_log_returns: pd.Series | None = None,
    freq: str = "D",
) -> dict:
    """Full-sample metrics, plus beta / alpha when a benchmark is supplied."""
    port = _to_datetime_series(portfolio_log_returns)
    out = period_metrics(port, freq)
    out["beta"] = np.nan
    out["alpha"] = np.nan

    if benchmark_log_returns is not None:
        bench = _to_datetime_series(benchmark_log_returns)
        p, b = port.align(bench, join="inner")
        if len(p.dropna()) >= 2:
            out["beta"] = beta(p, b)
            out["alpha"] = alpha(p, b, RISK_FREE_RATE, freq)
    return out


# ----
# Table builders
# ----
def crisis_phase_table(
    portfolio_log_returns: pd.Series,
    crises: list | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """One row per crisis, columns = ptt_* / ttp_* / full_* phase metrics."""
    crises = ALL_CRISES if crises is None else crises
    port = _to_datetime_series(portfolio_log_returns)

    rows = {}
    for c in crises:
        segments = {
            "ptt":  _slice_returns(port, c["peak"],   c["trough"]),
            "ttp":  _slice_returns(port, c["trough"], c["even"]),
            "full": _slice_returns(port, c["peak"],   c["even"]),
        }
        row = {}
        for prefix, seg in segments.items():
            m = period_metrics(seg, freq)
            row[f"{prefix}_sharpe"]  = m["sharpe"]
            row[f"{prefix}_sortino"] = m["sortino"]
            row[f"{prefix}_calmar"]  = m["calmar"]
            row[f"{prefix}_ulcer"]   = m["ulcer_index"]
            row[f"{prefix}_mdd"]     = m["maximum_drawdown"]
            row[f"{prefix}_return"]  = m["cumulative_return"]
        rows[c["label"]] = row

    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=PHASE_COLUMNS)


def metrics_table(
    portfolio_log_returns: pd.Series,
    benchmark_log_returns: pd.Series | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """The combined reporting table: periods x (base + benchmark-relative +
    crisis-phase) metrics.  This is what lands in metrics/metrics.csv."""
    port = _to_datetime_series(portfolio_log_returns)

    rows = {"overall": overall_metrics(port, benchmark_log_returns, freq)}
    for c in ALL_CRISES:
        seg = _slice_returns(port, c["peak"], c["even"])
        rows[c["label"]] = period_metrics(seg, freq)

    summary = pd.DataFrame.from_dict(rows, orient="index")
    phases = crisis_phase_table(port, freq=freq)

    table = summary.join(phases, how="left")
    table = table.reindex(index=PERIOD_ORDER, columns=METRICS_COLUMNS)
    table.index.name = "period"
    return table


# ----
# Raw timeseries builders
# ----
def daily_returns_frame(portfolio_log_returns: pd.Series) -> pd.DataFrame:
    port = _to_datetime_series(portfolio_log_returns)
    nav = _nav(port)
    df = pd.DataFrame(
        {
            "log_return": port,
            "cum_return": np.exp(port.cumsum()) - 1.0,
            "nav": nav,
        }
    )
    df.index.name = "date"
    return df


def monthly_value_frame(portfolio_log_returns: pd.Series) -> pd.DataFrame:
    port = _to_datetime_series(portfolio_log_returns)
    nav = _nav(port).resample("ME").last()
    df = pd.DataFrame({"nav": nav, "monthly_return": nav.pct_change()})
    df.index.name = "month_end"
    return df


# ----
# Benchmark loader
# ----
def load_benchmark(
    database_path: Path = DATABASE_PATH,
    ticker: str = BENCHMARK_TICKER,
) -> pd.Series:
    """Daily log returns of the benchmark from the shared price database."""
    prices = pd.read_parquet(database_path)
    adj_close = prices["Adj Close"][ticker].dropna()
    return np.log(adj_close / adj_close.shift(1)).dropna().rename(ticker)


# ----
# Main entry point
# ----
def build_report(
    model_name: str,
    portfolio_log_returns: pd.Series,
    *,
    benchmark_log_returns: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    diagnostics: dict | None = None,
    freq: str = "D",
    output_root: Path = OUTPUT_ROOT,
    include_benchmark: bool = True,
) -> Path:
    """
    Write the full raw/ + metrics/ folder tree for one model.

    Parameters
    ----------
    model_name
        Sub-folder created under `output_root`.
    portfolio_log_returns
        Daily log returns of the model's portfolio (DatetimeIndex).
    benchmark_log_returns
        Daily benchmark log returns; defaults to `load_benchmark()` when
        omitted.  Used for beta / alpha and for `metrics_benchmark.csv`.
    weights
        date x ticker weight matrix (written verbatim to raw/weights.csv).
    diagnostics
        Mapping of name -> Series/DataFrame written to raw/<name>.csv.
        Recognised names: hyperparameters, feature_importance, r2_raw_vs_zero,
        spearman_p, directional_accuracy, r2_selected, turnover,
        transaction_costs.  Unknown names are still written, with a warning.

    Returns
    -------
    Path to the model's output directory.
    """
    port = _to_datetime_series(portfolio_log_returns)
    if benchmark_log_returns is None:
        try:
            benchmark_log_returns = load_benchmark()
        except (FileNotFoundError, KeyError):
            benchmark_log_returns = None
    bench = (
        _to_datetime_series(benchmark_log_returns)
        if benchmark_log_returns is not None
        else None
    )

    model_dir = Path(output_root) / model_name
    raw_dir = model_dir / "raw"
    metrics_dir = model_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def _write(df: pd.DataFrame | pd.Series, path: Path) -> None:
        df.to_csv(path)
        written.append(path)

    # ── raw/ ──────────────────────────────────────────────────────────────────
    _write(daily_returns_frame(port), raw_dir / "daily_returns.csv")
    _write(monthly_value_frame(port), raw_dir / "monthly_value.csv")

    if weights is not None:
        w = weights.copy()
        w.index = pd.to_datetime(w.index)
        w.index.name = "date"
        _write(w, raw_dir / "weights.csv")

    for name, obj in (diagnostics or {}).items():
        if name not in _DIAGNOSTIC_KEYS:
            print(f"  [reporting] warning: unrecognised diagnostic '{name}' "
                  f"(still written to raw/{name}.csv)")
        _write(obj, raw_dir / f"{name}.csv")

    # ── metrics/ ──────────────────────────────────────────────────────────────
    _write(
        metrics_table(port, bench, freq).round(6),
        metrics_dir / "metrics.csv",
    )
    if include_benchmark and bench is not None:
        _write(
            metrics_table(bench, bench, freq).round(6),
            metrics_dir / "metrics_benchmark.csv",
        )

    print(f"[reporting] {model_name}: wrote {len(written)} files to {model_dir}")
    return model_dir


# ----
# Smoke test — equal-weight buy-and-hold over the price database
# ----
if __name__ == "__main__":
    from universe import tickers as universe_by_year

    prices = pd.read_parquet(DATABASE_PATH)["Adj Close"]

    demo_tickers = [t for t, _ in universe_by_year[2010] if t in prices.columns]
    px = prices[demo_tickers].dropna(how="all").ffill().dropna()

    asset_log_returns = np.log(px / px.shift(1)).dropna()
    portfolio = asset_log_returns.mean(axis=1)          # equal-weight, daily rebalanced
    weights = pd.DataFrame(
        1.0 / len(demo_tickers),
        index=asset_log_returns.index,
        columns=demo_tickers,
    )
    turnover = pd.Series(0.0, index=asset_log_returns.index, name="turnover")

    build_report(
        "_equal_weight_demo",
        portfolio,
        weights=weights,
        diagnostics={"turnover": turnover},
    )

    out = metrics_table(portfolio, load_benchmark())
    print(out.to_string(float_format=lambda x: f"{x:8.3f}"))
