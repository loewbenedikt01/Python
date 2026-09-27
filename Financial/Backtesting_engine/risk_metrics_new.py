"""
risk_metrics.py
===============
Standalone risk and performance analytics module.

All functions operate on a daily return Series or NAV Series (DatetimeIndex).
The main entry point is `compute()`, which returns a flat dict of every metric.

Sections
--------
  1. Helpers           — NAV/return extraction, risk-free rate loading
  2. Return metrics    — total return, CAGR, monthly / annual series
  3. Risk metrics      — volatility, drawdown, downside deviation, Ulcer Index
  4. Risk-adjusted     — Sharpe, Sortino, Calmar, Omega, Treynor, Sterling
  5. Market sensitivity— Beta, Alpha, R², Correlation  (require benchmark)
  6. Tail risk         — VaR, CVaR, skewness, kurtosis, tail ratio
  7. Trade stats       — win rate, profit factor, best / worst periods
  8. compute()         — aggregates everything into one dict
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Constants ─────────────────────────────────────────────────────────────────
PERIODS_PER_YEAR  = 252          # trading days per year
RF_FALLBACK       = 0.04         # 4 % annual risk-free rate if macro data unavailable
DATABASE_DIR      = Path(r'C:\Users\benel\Coding\Python\Database')
_FRED_RF_SERIES   = ['DGS3MO', 'TB3MS', 'DTB3']   # 3-month T-bill candidates


# ══════════════════════════════════════════════════════════════════════════════
# 1. Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_nav(portfolio_df: pd.DataFrame | pd.Series) -> pd.Series:
    """Extract or build a clean NAV series from a portfolio DataFrame."""
    if isinstance(portfolio_df, pd.Series):
        return portfolio_df.dropna()
    for col in ('nav', 'NAV', 'portfolio_value', 'value', 'close', 'Close'):
        if col in portfolio_df.columns:
            return portfolio_df[col].dropna()
    return portfolio_df.iloc[:, 0].dropna()

def _to_returns(portfolio_df: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a daily simple-return series."""
    if isinstance(portfolio_df, pd.Series):
        s = portfolio_df.dropna()
        if (s > 0).all() and s.max() > 5:   # looks like a NAV, not returns
            return s.pct_change().dropna()
        return s.dropna()
    for col in ('daily_return', 'returns', 'return', 'Daily_Return'):
        if col in portfolio_df.columns:
            return portfolio_df[col].dropna()
    return _to_nav(portfolio_df).pct_change().dropna()

def load_rf_rate(start: str | None = None, end: str | None = None) -> float:
    """
    Load the annualised risk-free rate from the macro parquet (3-month T-bill).
    Returns RF_FALLBACK if the file is missing or the series is unavailable.
    """
    path = DATABASE_DIR / 'macro.parquet'
    if not path.exists():
        return RF_FALLBACK
    try:
        macro = pd.read_parquet(path)
        for sid in _FRED_RF_SERIES:
            if sid in macro.columns:
                s = macro[sid].dropna()
                if start:
                    s = s[s.index >= start]
                if end:
                    s = s[s.index <= end]
                if not s.empty:
                    return float(s.mean()) / 100   # FRED series are in % terms
    except Exception:
        pass
    return RF_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# 2. Return metrics
# ══════════════════════════════════════════════════════════════════════════════

def total_return(nav: pd.Series) -> float:
    return float(nav.iloc[-1] / nav.iloc[0] - 1)

def cagr(nav: pd.Series) -> float:
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1)

def monthly_returns(rets: pd.Series) -> pd.Series:
    return rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)

def annual_returns(rets: pd.Series) -> pd.Series:
    return rets.resample('YE').apply(lambda x: (1 + x).prod() - 1)

def monthly_return_table(rets: pd.Series) -> pd.DataFrame:
    """Pivot table: rows = year, columns = month name."""
    mr = monthly_returns(rets)
    return pd.DataFrame({
        'Year':  mr.index.year,
        'Month': mr.index.month,
        'Ret':   mr.values,
    }).pivot(index='Year', columns='Month', values='Ret').rename(
        columns={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Risk metrics
# ══════════════════════════════════════════════════════════════════════════════

def annual_volatility(rets: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    return float(rets.std() * np.sqrt(periods))

def downside_deviation(rets: pd.Series, threshold: float = 0.0,
                       periods: int = PERIODS_PER_YEAR) -> float:
    """Annualised semi-deviation below `threshold`."""
    below = rets[rets < threshold]
    if below.empty:
        return 0.0
    return float(np.sqrt((below ** 2).mean()) * np.sqrt(periods))

def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1

def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown_series(nav).min())

def avg_drawdown(nav: pd.Series) -> float:
    dd = drawdown_series(nav)
    return float(dd[dd < 0].mean())

def drawdown_duration(nav: pd.Series) -> dict:
    """
    Returns the longest and current drawdown durations in calendar days,
    and the average drawdown duration across all episodes.
    """
    dd   = drawdown_series(nav)
    in_dd = dd < 0

    max_dur, cur_dur, start = 0, 0, None
    durations = []

    for date, is_dd in in_dd.items():
        if is_dd:
            if start is None:
                start = date
            cur_dur = (date - start).days
            max_dur = max(max_dur, cur_dur)
        else:
            if start is not None:
                durations.append(cur_dur)
            start, cur_dur = None, 0

    if start is not None:
        durations.append(cur_dur)

    return {
        'max_dd_duration_days': max_dur,
        'avg_dd_duration_days': float(np.mean(durations)) if durations else 0.0,
        'current_dd_duration_days': cur_dur,
    }

def recovery_factor(nav: pd.Series) -> float:
    mdd = max_drawdown(nav)
    if mdd == 0:
        return np.nan
    return float(total_return(nav) / abs(mdd))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Risk-adjusted ratios
# ══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(rets: pd.Series, rf_annual: float = RF_FALLBACK,
                 periods: int = PERIODS_PER_YEAR) -> float:
    rf_daily  = (1 + rf_annual) ** (1 / periods) - 1
    excess    = rets - rf_daily
    vol       = rets.std()
    if vol == 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods))

def sortino_ratio(rets: pd.Series, rf_annual: float = RF_FALLBACK,
                  periods: int = PERIODS_PER_YEAR) -> float:
    rf_daily  = (1 + rf_annual) ** (1 / periods) - 1
    excess    = rets - rf_daily
    dd        = downside_deviation(rets, threshold=rf_daily, periods=periods)
    if dd == 0:
        return np.nan
    return float(excess.mean() * periods / dd)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Tail risk
# ══════════════════════════════════════════════════════════════════════════════

def var_historical(rets: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: loss not exceeded with `confidence` probability."""
    return float(np.percentile(rets, (1 - confidence) * 100))

def cvar_historical(rets: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): mean return in the worst tail."""
    threshold = var_historical(rets, confidence)
    return float(rets[rets <= threshold].mean())

def var_parametric(rets: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian parametric VaR."""
    z = scipy_stats.norm.ppf(1 - confidence)
    return float(rets.mean() + z * rets.std())

def cvar_parametric(rets: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian parametric CVaR (Expected Shortfall)."""
    z    = scipy_stats.norm.ppf(1 - confidence)
    pdf  = scipy_stats.norm.pdf(z)
    return float(rets.mean() - rets.std() * pdf / (1 - confidence))

def skewness(rets: pd.Series) -> float:
    return float(scipy_stats.skew(rets.dropna()))

def excess_kurtosis(rets: pd.Series) -> float:
    return float(scipy_stats.kurtosis(rets.dropna()))   # Fisher definition (normal = 0)

def tail_ratio(rets: pd.Series, percentile: float = 5.0) -> float:
    """95th-percentile gain / abs(5th-percentile loss)."""
    upper = np.percentile(rets, 100 - percentile)
    lower = abs(np.percentile(rets, percentile))
    return float(upper / lower) if lower > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 6. Trade / period statistics
# ══════════════════════════════════════════════════════════════════════════════

def win_rate(rets: pd.Series) -> float:
    return float((rets > 0).mean())

def profit_factor(rets: pd.Series) -> float:
    gains  = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return float(gains / losses) if losses > 0 else np.inf

def best_day(rets: pd.Series) -> tuple[float, pd.Timestamp]:
    idx = rets.idxmax()
    return float(rets[idx]), idx

def worst_day(rets: pd.Series) -> tuple[float, pd.Timestamp]:
    idx = rets.idxmin()
    return float(rets[idx]), idx

def best_month(rets: pd.Series) -> tuple[float, pd.Timestamp]:
    mr  = monthly_returns(rets)
    idx = mr.idxmax()
    return float(mr[idx]), idx

def worst_month(rets: pd.Series) -> tuple[float, pd.Timestamp]:
    mr  = monthly_returns(rets)
    idx = mr.idxmin()
    return float(mr[idx]), idx

def best_year(rets: pd.Series) -> tuple[float, int]:
    ar  = annual_returns(rets)
    idx = ar.idxmax()
    return float(ar[idx]), idx.year

def worst_year(rets: pd.Series) -> tuple[float, int]:
    ar  = annual_returns(rets)
    idx = ar.idxmin()
    return float(ar[idx]), idx.year

def consecutive_wins_losses(rets: pd.Series) -> dict:
    signs = np.sign(rets.values)
    max_w = max_l = cur_w = cur_l = 0
    for s in signs:
        if s > 0:
            cur_w += 1; cur_l  = 0; max_w = max(max_w, cur_w)
        elif s < 0:
            cur_l += 1; cur_w  = 0; max_l = max(max_l, cur_l)
        else:
            cur_w = cur_l = 0
    return {'max_consecutive_wins': max_w, 'max_consecutive_losses': max_l}


# ══════════════════════════════════════════════════════════════════════════════
# 7. Rolling metrics
# ══════════════════════════════════════════════════════════════════════════════

def rolling_sharpe(rets: pd.Series, window: int = PERIODS_PER_YEAR,
                   rf_annual: float = RF_FALLBACK,
                   periods: int = PERIODS_PER_YEAR) -> pd.Series:
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    excess   = rets - rf_daily
    return (
        excess.rolling(window)
        .apply(lambda x: x.mean() / x.std() * np.sqrt(periods) if x.std() > 0 else np.nan,
               raw=True)
    )

def rolling_volatility(rets: pd.Series, window: int = PERIODS_PER_YEAR,
                        periods: int = PERIODS_PER_YEAR) -> pd.Series:
    return rets.rolling(window).std() * np.sqrt(periods)

def rolling_drawdown(nav: pd.Series, window: int = PERIODS_PER_YEAR) -> pd.Series:
    return nav.rolling(window, min_periods=1).apply(
        lambda x: x[-1] / x.max() - 1, raw=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main compute() entry point
# ══════════════════════════════════════════════════════════════════════════════

def compute(
    portfolio_df:   pd.DataFrame | pd.Series,
    benchmark_df:   pd.DataFrame | pd.Series | None = None,
    rf_annual:      float | None = None,
    periods:        int  = PERIODS_PER_YEAR,
    confidence:     float = 0.95,
) -> dict:
    """
    Compute all risk and performance metrics for a portfolio.

    Parameters
    ----------
    portfolio_df  : DataFrame with a NAV or daily-return column, or a NAV Series.
    benchmark_df  : Optional benchmark (same format). Required for Beta, Alpha, IR.
    rf_annual     : Annual risk-free rate. If None, loaded from macro.parquet.
    periods       : Trading periods per year (252 for daily).
    confidence    : VaR / CVaR confidence level (default 0.95 = 95%).

    Returns
    -------
    dict — flat mapping of metric name → value.
    """
    nav  = _to_nav(portfolio_df)
    rets = _to_returns(portfolio_df)

    if rf_annual is None:
        rf_annual = load_rf_rate(
            start=str(nav.index[0].date()),
            end=str(nav.index[-1].date()),
        )

    # ── Returns ───────────────────────────────────────────────────────────────
    m: dict = {}
    m['total_return']       = total_return(nav)
    m['cagr']               = cagr(nav)
    m['rf_rate_used']       = rf_annual
    m['start_date']         = nav.index[0].date()
    m['end_date']           = nav.index[-1].date()
    m['n_days']             = len(rets)
    m['n_years']            = (nav.index[-1] - nav.index[0]).days / 365.25

    # ── Risk ──────────────────────────────────────────────────────────────────
    m['annual_volatility']  = annual_volatility(rets, periods)
    m['downside_deviation'] = downside_deviation(rets, periods=periods)
    m['max_drawdown']       = max_drawdown(nav)
    m['avg_drawdown']       = avg_drawdown(nav)
    m['recovery_factor']    = recovery_factor(nav)
    m.update(drawdown_duration(nav))

    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    m['sharpe_ratio']       = sharpe_ratio(rets, rf_annual, periods)
    m['sortino_ratio']      = sortino_ratio(rets, rf_annual, periods)

    # ── Tail risk ─────────────────────────────────────────────────────────────
    m['var_historical']     = var_historical(rets, confidence)
    m['cvar_historical']    = cvar_historical(rets, confidence)
    m['var_parametric']     = var_parametric(rets, confidence)
    m['cvar_parametric']    = cvar_parametric(rets, confidence)
    m['skewness']           = skewness(rets)
    m['excess_kurtosis']    = excess_kurtosis(rets)
    m['tail_ratio']         = tail_ratio(rets)

    # ── Period statistics ─────────────────────────────────────────────────────
    m['win_rate']           = win_rate(rets)
    m['profit_factor']      = profit_factor(rets)
    m['best_day'],  m['best_day_date']   = best_day(rets)
    m['worst_day'], m['worst_day_date']  = worst_day(rets)
    m['best_month'],  m['best_month_date']  = best_month(rets)
    m['worst_month'], m['worst_month_date'] = worst_month(rets)
    m['best_year'],   m['best_year_int']  = best_year(rets)
    m['worst_year'],  m['worst_year_int'] = worst_year(rets)
    m.update(consecutive_wins_losses(rets))

    return m


def summary(metrics: dict) -> str:
    """Pretty-print the most important metrics as a console table."""
    groups = [
        ('Returns',
         [('Total Return',       f'{metrics["total_return"]*100:.2f}%'),
          ('CAGR',               f'{metrics["cagr"]*100:.2f}%'),
          ('Best Year',          f'{metrics["best_year"]*100:.2f}% ({metrics["best_year_int"]})'),
          ('Worst Year',         f'{metrics["worst_year"]*100:.2f}% ({metrics["worst_year_int"]})')]),
        ('Risk',
         [('Annual Volatility',  f'{metrics["annual_volatility"]*100:.2f}%'),
          ('Max Drawdown',       f'{metrics["max_drawdown"]*100:.2f}%'),
          ('Avg Drawdown',       f'{metrics["avg_drawdown"]*100:.2f}%'),
          ('Max DD Duration',    f'{metrics["max_dd_duration_days"]} days')]),
        ('Risk-Adjusted',
         [('Sharpe Ratio',       f'{metrics["sharpe_ratio"]:.3f}'),
          ('Sortino Ratio',      f'{metrics["sortino_ratio"]:.3f}')]),
        ('Tail Risk',
         [('VaR 95% (hist)',     f'{metrics["var_historical"]*100:.2f}%'),
          ('CVaR 95% (hist)',    f'{metrics["cvar_historical"]*100:.2f}%'),
          ('Skewness',           f'{metrics["skewness"]:.3f}'),
          ('Excess Kurtosis',    f'{metrics["excess_kurtosis"]:.3f}'),
          ('Tail Ratio',         f'{metrics["tail_ratio"]:.3f}')]),
        ('Statistics',
         [('Win Rate',           f'{metrics["win_rate"]*100:.1f}%'),
          ('Profit Factor',      f'{metrics["profit_factor"]:.3f}'),
          ('Best Day',           f'{metrics["best_day"]*100:.2f}%'),
          ('Worst Day',          f'{metrics["worst_day"]*100:.2f}%')]),
    ]

    lines = [f'\n=== Risk Metrics ({metrics["start_date"]} -> {metrics["end_date"]}) ===\n']
    for group_name, rows in groups:
        lines.append(f'  -- {group_name} {"-"*(44-len(group_name))}')
        for label, value in rows:
            lines.append(f'  {label:<26}  {value:>12}')
        lines.append('')
    return '\n'.join(lines)
