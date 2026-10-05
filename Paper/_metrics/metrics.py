
'''
Definitions of Metrics

Metrics are used for Calculation and Reporting

6 Main Metrics:
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Max. Drawdown, Cumulative Return
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
    raise ValueError(f'freq must be "D" or "M", got "{freq}"')


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


# ----
# D. RISK-ADJUSTED RATIOS
# ----

def _excess(log_returns: pd.Series, rf=None, freq: str = 'D') -> pd.Series:
    '''
    Log returns in excess of the risk-free rate, period by period.
    rf None:   the configured series (risk_free_rate.py, RISK_FREE in config.py),
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
            raise ValueError('Risk-Free series does not cover all return dates. Fix.')
        return r - rf
    return r - float(rf) / _annualized_factor(freq)

def sharpe_ratio(log_returns: pd.Series, rf=None, freq: str = 'D') -> float:
    '''
    Annualised Sharpe Ratio on excess log returns:
    mean(r - rf) * ann_factor / (std(r - rf) * sqrt(ann_factor)).
    Uses full return distribution (not downside-only).
    '''
    excess          = _excess(log_returns, rf, freq)
    ann_f           = _annualized_factor(freq)
    ann_vol         = excess.std(ddof=1) * np.sqrt(ann_f)
    return float(excess.mean() * ann_f / ann_vol) if ann_vol != 0 else np.nan

def sortino_ratio(log_returns: pd.Series, rf=None, freq: str = 'D') -> float:
    '''
    Annualised Sortino Ratio on excess log returns, target = rf:
    mean(r - rf) * ann_factor / downside_deviation,
    downside_deviation = sqrt(mean(min(r - rf, 0)^2)) * sqrt(ann_factor).
    Penalises only returns below the risk-free rate, not upside volatility.
    '''
    excess          = _excess(log_returns, rf, freq)
    ann_f           = _annualized_factor(freq)
    downside_vol    = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(ann_f)
    return float(excess.mean() * ann_f / downside_vol) if downside_vol != 0 else np.nan

def calmar_ratio(log_returns: pd.Series, price_series: pd.Series, freq: str = 'D') -> float:
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
    
def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf=None, freq: str = 'D') -> float:
    '''
    Jensen's Alpha, annualised, from excess log returns on common dates:
    (mean(p - rf) - beta * mean(b - rf)) * ann_factor, beta estimated on the
    excess returns.  Positive alpha = portfolio outperforms its CAPM expectation.
    '''
    df              = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    port_excess     = _excess(df.iloc[:, 0], rf, freq)
    bench_excess    = _excess(df.iloc[:, 1], rf, freq)
    b               = beta(port_excess, bench_excess)
    return float((port_excess.mean() - b * bench_excess.mean()) * _annualized_factor(freq))

