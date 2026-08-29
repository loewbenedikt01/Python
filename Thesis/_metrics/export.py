
"""
Reporting — the connector between a model's raw output, the metric
definitions (metrics.py) and the crisis windows (crises.py).


"""




from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RISK_FREE_RATE,
    OUTPUT_ROOT,
)
from crises import (
    overall,
    main_crises,
    sub_crises,
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
)


# ----
# Extract Crises and combine
# ----
OVERALL     = overall[0]                       # full investment horizon (1998-2025)
MAIN_CRISES = list(main_crises)
SUB_CRISES  = list(sub_crises)


# ----
# Metrics to report, for full period, and per crisis ptt and ttp 
# ----
reporting_metrics  = [
    {'label': 'sharpe_ratio',       'format': '{:.3f}'},        # Sharpe Ratio
    {'label': 'sortino_ratio',      'format': '{:.3f}'},        # Sortino Ratio
    {'label': 'calmar_ratio',       'format': '{:.3f}'},        # Calmar Ratio
    {'label': 'ulcer_index',        'format': '{:.3f}'},        # Ulcer Index
    {'label': 'maximum_drawdown',   'format': '{:.2%}'},        # Maximum Drawdown
    {'label': 'cumulative_return',  'format': '{:.2%}'},        # Cumulative Return
    {'label': 'annual_return',      'format': '{:.2%}'},        # Annualized Return
    {'label': 'annual_volatility',  'format': '{:.2%}'},        # Annualized Volatility
]

REPORT_METRICS      = [d['label'] for d in reporting_metrics]

# Crisis sub-windows. ptt = peak->trough, ttp = trough->recovery, full = peak->recovery.
PHASES = ['ptt', 'ttp', 'full']


# ----
# Raw timeseries written per model
# ----
# Every model.
RAW_TIMESERIES = [
    'daily_returns',
    'monthly_value',
    'weights',
    'turnover',
    'transaction_costs',
]

RAW_TIMESERIES_ML = [
    'daily_returns',
    'monthly_value',
    'weights',
    'hyperparameters',
    'feature_importance',
    'r2_selected',
    'r2_raw_vs_zero',
    'spearman_p',
    'directional_accuracy',
    'turnover',
    'transaction_costs',
]

ML_MODELS = {'xgb', 'rf', 'lstm'}

# daily_returns / monthly_value / weights are built by this module; every other
# name above must be supplied through build_report(diagnostics=...).
_BUILT_HERE        = {'daily_returns', 'monthly_value', 'weights'}
RAW_DIAGNOSTICS    = [k for k in RAW_TIMESERIES    if k not in _BUILT_HERE]
RAW_DIAGNOSTICS_ML = [k for k in RAW_TIMESERIES_ML if k not in _BUILT_HERE]


# ----
# Price / return helpers
# ----
def _to_datetime_series(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def _nav(log_returns: pd.Series) -> pd.Series:
    """
    Dated net-asset-value path from daily log returns, anchored one step
    before the first observation at 1.0.
    """
    return np.exp(log_returns.cumsum())

def _drawdown_price(log_returns: pd.Series) -> pd.Series:
    """
    Undated price path that *starts at 1.0* before the first return, so a
    segment's drawdown is measured from its own opening level.  Index is a
    plain range – only order matters for cummax-based metrics.
    """
    levels = np.exp(np.concatenate([[0.0], np.cumsum(log_returns.values)]))
    return pd.Series(levels)

def _slice_returns(log_returns: pd.Series, start, end) -> pd.Series:
    """
    Returns in (start, end] – the return earned *after* the peak date up to
    and including the trough / recovery date.
    """
    idx = log_returns.index
    mask = (idx > pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return log_returns.loc[mask]


# ----
# Per-period metrics
# ----
def period_metrics(log_returns: pd.Series, freq: str = "D") -> dict:
    """
    The 8 reporting metrics for one return window, keyed by `reporting_metrics`
    labels.  All-NaN when the window holds fewer than 2 observations (e.g. a
    crisis that predates the model).
    """
    log_returns = log_returns.dropna()
    if len(log_returns) < 2:
        return {m: np.nan for m in REPORT_METRICS}

    dd_price = _drawdown_price(log_returns)
    return {
        "sharpe_ratio":      sharpe_ratio(log_returns, RISK_FREE_RATE, freq),
        "sortino_ratio":     sortino_ratio(log_returns, RISK_FREE_RATE, freq),
        "calmar_ratio":      calmar_ratio(log_returns, dd_price, freq),
        "ulcer_index":       ulcer_index(dd_price),
        "maximum_drawdown":  maximum_drawdown(dd_price),
        "cumulative_return": cumulative_return(log_returns),
        "annual_return":     annualized_return(log_returns, freq),
        "annual_volatility": annualized_volatility(log_returns, freq),
    }

# ----
# Table builders
# ----
def _phase_segment(log_returns: pd.Series, crisis: dict, phase: str) -> pd.Series:
    if phase == "ptt":
        return _slice_returns(log_returns, crisis["peak"], crisis["trough"])
    if phase == "ttp":
        return _slice_returns(log_returns, crisis["trough"], crisis["even"])
    if phase == "full":
        return _slice_returns(log_returns, crisis["peak"], crisis["even"])
    raise ValueError(f"unknown phase {phase!r}")

def metrics_table(
    portfolio_log_returns: pd.Series,
    crises: list,
    *,
    include_overall: bool = False,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Reporting table for one set of crises.
    """
    port = _to_datetime_series(portfolio_log_returns)

    rows = {}
    if include_overall:
        rows["overall"] = period_metrics(
            _slice_returns(port, OVERALL["peak"], OVERALL["even"]), freq
        )
    for c in crises:
        for phase in PHASES:
            rows[f"{c['label']}_{phase}"] = period_metrics(
                _phase_segment(port, c, phase), freq
            )

    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=REPORT_METRICS)
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
# Main entry point
# ----
def build_report(
    model_name: str,
    portfolio_log_returns: pd.Series,
    *,
    weights: pd.DataFrame | None = None,
    diagnostics: dict | None = None,
    ml: bool | None = None,
    freq: str = "D",
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """
    Write the full raw/ + metrics/ folder tree for one model.
    """
    if ml is None:
        ml = model_name in ML_MODELS
    port = _to_datetime_series(portfolio_log_returns)

    model_dir = Path(output_root) / model_name
    raw_dir = model_dir / "raw"
    metrics_dir = model_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def _write(df: pd.DataFrame | pd.Series, path: Path) -> None:
        df.to_csv(path)
        written.append(path)

    # raw 
    _write(daily_returns_frame(port), raw_dir / "daily_returns.csv")
    _write(monthly_value_frame(port), raw_dir / "monthly_value.csv")

    if weights is not None:
        w = weights.copy()
        w.index = pd.to_datetime(w.index)
        w.index.name = "date"
        _write(w, raw_dir / "weights.csv")

    expected = set(RAW_DIAGNOSTICS_ML if ml else RAW_DIAGNOSTICS)
    provided = dict(diagnostics or {})
    for name, obj in provided.items():
        if name not in expected:
            print(f"  [reporting] warning: unexpected diagnostic '{name}' for "
                  f"{'ML ' if ml else ''}model '{model_name}' "
                  f"(written to raw/{name}.csv anyway)")
        _write(obj, raw_dir / f"{name}.csv")
    for name in sorted(expected - provided.keys()):
        print(f"  [reporting] note: '{name}' not supplied for '{model_name}'")

    # ── metrics/ ──────────────────────────────────────────────────────────────
    _write(metrics_table(port, MAIN_CRISES, include_overall=True, freq=freq).round(6),
           metrics_dir / "metrics.csv")
    _write(metrics_table(port, SUB_CRISES, freq=freq).round(6),
           metrics_dir / "metrics_sub_crises.csv")

    print(f"[reporting] {model_name}: wrote {len(written)} files to {model_dir}")
    return model_dir
