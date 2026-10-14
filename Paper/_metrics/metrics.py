
'''
Definitions of Metrics

Metrics are used for Calculation and Reporting

6 Main Metrics:
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Max. Drawdown, Cumulative Return

Tail and drawdown-duration metrics (added 2026-10-14):
Value at Risk and Conditional Value at Risk (historical, 95%), maximum
drawdown duration, downside capture ratio against a benchmark.
'''


import numpy as np
import pandas as pd

from config import (
    TRADING_DAYS_PER_YEAR, 
    MONTHS_PER_YEAR
)
from risk_free_rate import rf_period

# ----
# Annualization
# ----
def _annualized_factor(freq: str) -> int:
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    raise ValueError(f'freq must be D or M, got {freq!r}')


# ----
# B. RETURN METRICS
# ----
def cumulative_return(log_returns: pd.Series) -> float:
    '''
    Total cumulative return: exp(sum(log_returns)) - 1
    '''
    log_returns = log_returns.dropna()
    return float(np.exp(log_returns.sum()) - 1)

def annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    '''
    Geometric annualised return accounting for compounding.
    '''
    log_returns = log_returns.dropna()
    n     = len(log_returns)
    if n == 0:
        return np.nan
    ann_f = _annualized_factor(freq)
    return float(np.exp((log_returns.sum() / n) * ann_f) - 1)

def arith_annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    '''
    Arithmetic annualised mean of log returns (mean * periods_per_year).
    Matches the (log, arithmetic) units of annualized_volatility, so it is
    the right numerator for risk-adjusted ratios.
    '''
    return float(log_returns.mean() * _annualized_factor(freq))


# ----
# C. RISK METRICS
# ----
def annualized_volatility(log_returns: pd.Series, freq: str = 'D') -> float:
    '''
    Annualised standard deviation of log returns (ddof=1).
    '''
    return float(log_returns.std(ddof=1) * np.sqrt(_annualized_factor(freq)))

def maximum_drawdown(price_series: pd.Series) -> float:
    '''
    Worst peak-to-trough decline in the price series.
    '''
    rolling_max = price_series.cummax()
    return float(((price_series / rolling_max) - 1).min())

def ulcer_index(price_series: pd.Series) -> float:
    '''
    Ulcer Index = sqrt(mean(drawdown²)).
    Measures both depth and duration of being underwater.
    All drawdowns computed from the running peak (standard definition).
    Higher values = more painful drawdown profile.
    '''
    rolling_max = price_series.cummax()
    dd          = (price_series / rolling_max) - 1
    return float(np.sqrt((dd ** 2).mean()))

def value_at_risk(log_returns: pd.Series, level: float = 0.95) -> float:
    '''
    Historical one-period Value at Risk: the (1 - level) quantile of simple
    returns.  Negative number = loss, e.g. -0.02 = 2% loss on the worst
    5% of days at level 0.95.
    '''
    r = np.expm1(log_returns.dropna())
    return float(r.quantile(1.0 - level)) if len(r) else np.nan

def conditional_value_at_risk(log_returns: pd.Series, level: float = 0.95) -> float:
    '''
    Historical Conditional Value at Risk (expected shortfall): the mean simple
    return on the periods at or below the Value at Risk.  Coherent risk
    measure (Artzner et al. 1999; Acerbi & Tasche 2002), unlike VaR it also
    reflects how large the losses beyond the cutoff are.
    '''
    r = np.expm1(log_returns.dropna())
    if not len(r):
        return np.nan
    return float(r[r <= r.quantile(1.0 - level)].mean())

def max_drawdown_duration(price_series: pd.Series) -> int:
    '''
    Longest time under water, in periods (trading days for daily data): the
    longest run of consecutive periods below the running peak, i.e. from a
    peak until the first value back at or above it.  A drawdown that has not
    recovered by the end of the sample counts up to the last date.
    '''
    p = price_series.dropna().to_numpy(dtype=float)
    under = p < np.maximum.accumulate(p)
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


# ----
# D. RISK-ADJUSTED RATIOS
# ----
def _excess(log_returns: pd.Series, rf=None, freq: str = 'D') -> pd.Series:
    '''
    Log returns in excess of the risk-free rate, period by period.
    rf None:   the configured series (risk_free.py, RISK_FREE in config.py),
               aligned to the dates of log_returns
    rf Series: per-period log risk-free returns, aligned by date
    rf float:  constant annual rate, spread evenly over the periods
    '''
    r = log_returns.dropna()
    if rf is None:
        rf = rf_period(r.index, freq)
    if isinstance(rf, pd.Series):
        rf = rf.reindex(r.index)
        if rf.isna().any():
            raise ValueError('risk-free series does not cover all return dates')
        return r - rf
    return r - float(rf) / _annualized_factor(freq)

def sharpe_ratio(log_returns: pd.Series, rf=None, freq: str = 'D') -> float:
    '''
    Annualised Sharpe Ratio on excess log returns:
    mean(r - rf) * ann_factor / (std(r - rf) * sqrt(ann_factor)).
    Uses full return distribution (not downside-only).
    '''
    ex      = _excess(log_returns, rf, freq)
    ann_f   = _annualized_factor(freq)
    ann_vol = ex.std(ddof=1) * np.sqrt(ann_f)
    return float(ex.mean() * ann_f / ann_vol) if ann_vol != 0 else np.nan

def sortino_ratio(log_returns: pd.Series, rf=None, freq: str = 'D') -> float:
    '''
    Annualised Sortino Ratio on excess log returns, target = rf:
    mean(r - rf) * ann_factor / downside_deviation,
    downside_deviation = sqrt(mean(min(r - rf, 0)^2)) * sqrt(ann_factor).
    Penalises only returns below the risk-free rate, not upside volatility.
    '''
    ex           = _excess(log_returns, rf, freq)
    ann_f        = _annualized_factor(freq)
    downside_vol = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ann_f)
    return float(ex.mean() * ann_f / downside_vol) if downside_vol != 0 else np.nan

def calmar_ratio(log_returns: pd.Series, price_series: pd.Series,
                 freq: str = 'D') -> float:
    '''
    Calmar Ratio = annualized_return / |maximum_drawdown|.
    Reward per unit of the worst observed drawdown.
    '''
    ann_ret = annualized_return(log_returns, freq)
    mdd     = abs(maximum_drawdown(price_series))
    return float(ann_ret / mdd) if mdd != 0 else np.nan


# ----
# E. BENCHMARK-RELATIVE METRICS
# ----
def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    '''
    Beta — sensitivity of portfolio returns to benchmark returns.
    Formula: cov(p, b) / var(b).
    Beta > 1: amplifies market moves. Beta < 1: dampens them.
    '''
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    aligned_p, aligned_b = df.iloc[:, 0], df.iloc[:, 1]
    cov = np.cov(aligned_p.values, aligned_b.values)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan
    
def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          rf=None, freq: str = 'D') -> float:
    '''
    Jensen's Alpha, annualised, from excess log returns on common dates:
    (mean(p - rf) - beta * mean(b - rf)) * ann_factor, beta estimated on the
    excess returns.  Positive alpha = portfolio outperforms its CAPM expectation.
    '''
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    pe = _excess(df.iloc[:, 0], rf, freq)
    be = _excess(df.iloc[:, 1], rf, freq)
    b  = beta(pe, be)
    return float((pe.mean() - b * be.mean()) * _annualized_factor(freq))


def downside_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    '''
    Monthly downside capture ratio from daily log returns: the portfolio's
    mean monthly simple return in the months where the benchmark falls,
    divided by the benchmark's mean in those months.  Below 1 = the portfolio
    loses less than the benchmark in down months.
    '''
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if df.empty:
        return np.nan
    m = np.expm1(df.groupby(df.index.to_period('M')).sum())
    down = m.iloc[:, 1] < 0
    if not down.any():
        return np.nan
    return float(m.loc[down].iloc[:, 0].mean() / m.loc[down].iloc[:, 1].mean())
