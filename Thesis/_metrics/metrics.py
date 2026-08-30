
"""
Definitions of Metrics

Metrics are used for Calculation and Reporting

6 Main Metrics:
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Max. Drawdown, Cumulative Return
"""


import numpy as np
import pandas as pd

from config import (
    RISK_FREE_RATE, 
    TRADING_DAYS_PER_YEAR, 
    MONTHS_PER_YEAR
)

# ----
# Annualization
# ----
def _annualized_factor(freq: str) -> int:
    if freq == 'D':
        return TRADING_DAYS_PER_YEAR
    elif freq == 'M':
        return MONTHS_PER_YEAR
    raise ValueError(f"freq must be 'D' or 'M', got '{freq}'")


# ----
# B. RETURN METRICS
# ----
def cumulative_return(log_returns: pd.Series) -> float:
    """
    Total cumulative return: exp(sum(log_returns)) - 1
    """
    log_returns = log_returns.dropna()
    return float(np.exp(log_returns.sum()) - 1)

def annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    """
    Geometric annualised return accounting for compounding.
    """
    log_returns = log_returns.dropna()
    n     = len(log_returns)
    if n == 0:
        return np.nan
    ann_f = _annualized_factor(freq)
    return float(np.exp((log_returns.sum() / n) * ann_f) - 1)

def arith_annualized_return(log_returns: pd.Series, freq: str = 'D') -> float:
    """
    Arithmetic annualised mean of log returns (mean * periods_per_year).
    Matches the (log, arithmetic) units of annualized_volatility, so it is
    the right numerator for risk-adjusted ratios.
    """
    return float(log_returns.mean() * _annualized_factor(freq))


# ----
# C. RISK METRICS
# ----
def annualized_volatility(log_returns: pd.Series, freq: str = 'D') -> float:
    """
    Annualised standard deviation of log returns (ddof=1).
    """
    return float(log_returns.std(ddof=1) * np.sqrt(_annualized_factor(freq)))

def maximum_drawdown(price_series: pd.Series) -> float:
    """
    Worst peak-to-trough decline in the price series.
    """
    rolling_max = price_series.cummax()
    return float(((price_series / rolling_max) - 1).min())

def ulcer_index(price_series: pd.Series) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown²)).
    Measures both depth and duration of being underwater.
    All drawdowns computed from the running peak (standard definition).
    Higher values = more painful drawdown profile.
    """
    rolling_max = price_series.cummax()
    dd          = (price_series / rolling_max) - 1
    return float(np.sqrt((dd ** 2).mean()))


# ----
# D. RISK-ADJUSTED RATIOS
# ----
def sharpe_ratio(log_returns: pd.Series, rf: float = RISK_FREE_RATE,
                 freq: str = 'D') -> float:
    """
    Annualised Sharpe Ratio.
    (annualized_return - rf) / annualized_volatility.
    Uses full return distribution (not downside-only).
    """
    ann_ret     = arith_annualized_return(log_returns, freq)
    ann_vol     = annualized_volatility(log_returns, freq)
    return float((ann_ret - rf) / ann_vol) if ann_vol != 0 else np.nan

def sortino_ratio(log_returns: pd.Series, rf: float = RISK_FREE_RATE,
                  freq: str = 'D') -> float:
    """
    Annualised Sortino Ratio.
    (annualized_return - rf) / downside_deviation.
    Downside deviation = sqrt(mean(min(r, 0)²)) * sqrt(ann_factor).
    Penalises only negative returns, not upside volatility.
    """
    ann_ret         = arith_annualized_return(log_returns, freq)
    ann_f           = _annualized_factor(freq)
    downside_vol    = np.sqrt((np.minimum(log_returns, 0) ** 2).mean()) * np.sqrt(ann_f)
    return float((ann_ret - rf) / downside_vol) if downside_vol != 0 else np.nan

def calmar_ratio(log_returns: pd.Series, price_series: pd.Series,
                 freq: str = 'D') -> float:
    """
    Calmar Ratio = annualized_return / |maximum_drawdown|.
    Reward per unit of the worst observed drawdown.
    """
    ann_ret = annualized_return(log_returns, freq)
    mdd     = abs(maximum_drawdown(price_series))
    return float(ann_ret / mdd) if mdd != 0 else np.nan


# ----
# E. BENCHMARK-RELATIVE METRICS
# ----
def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta — sensitivity of portfolio returns to benchmark returns.
    Formula: cov(p, b) / var(b).
    Beta > 1: amplifies market moves. Beta < 1: dampens them.
    """
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    aligned_p, aligned_b = df.iloc[:, 0], df.iloc[:, 1]
    cov = np.cov(aligned_p.values, aligned_b.values)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan
    
def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          rf: float = RISK_FREE_RATE, freq: str = 'D') -> float:
    """
    Jensen's Alpha — annualised excess return vs CAPM expectation.
    Formula: ann_port - (rfr + beta * (ann_bench - rf)).
    Positive alpha = portfolio outperforms its risk-adjusted benchmark expectation.
    """
    b      = beta(portfolio_returns, benchmark_returns)
    ann_p  = annualized_return(portfolio_returns, freq)
    ann_bm = annualized_return(benchmark_returns, freq)
    return float(ann_p - (rf + b * (ann_bm - rf)))

