"""
output.py
=========
Saves all backtest results to Output/<backtest_name>/.

Generated files
---------------
  annual_returns.csv          — calendar-year return for the portfolio
  monthly_returns.csv         — month-by-month return (also used for heatmap)
  portfolio_timeseries.csv    — daily NAV, cumulative return, drawdown
  portfolio_components.csv    — monthly: ticker, target weight, drift weight,
                                price at rebalance, price change since rebalance
  sharpe_ratio.csv            — rolling 12-month and full-period Sharpe
  annual_volatility.csv       — rolling 12-month and full-period annualised vol
  report.html                 — self-contained HTML: metrics table + embedded charts
"""

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe to call from any thread
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Output root ───────────────────────────────────────────────────────────────
_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'Output'


# ── Public entry point ────────────────────────────────────────────────────────

def save_results(
    backtest_name: str,
    portfolio_df: pd.DataFrame,
    allocations: list[dict],
    equity_df: pd.DataFrame,
    metrics: dict,
) -> Path:
    """
    Write all output files and return the output directory path.

    Parameters
    ----------
    backtest_name   : folder name (e.g. 'test_01')
    portfolio_df    : daily DataFrame with columns: nav, daily_return, cumulative_return
    allocations     : list of {date, tickers, weights} dicts from the backtest loop
    equity_df       : full equity price DataFrame (MultiIndex Date/Ticker)
    metrics         : dict returned by risk_metrics.compute()
    """
    out = _OUTPUT_ROOT / backtest_name
    out.mkdir(parents=True, exist_ok=True)
    print(f'\nSaving results to {out} ...')

    _save_annual_returns(out, portfolio_df)
    _save_monthly_returns(out, portfolio_df)
    _save_portfolio_timeseries(out, portfolio_df)
    _save_portfolio_components(out, allocations, equity_df)
    _save_sharpe_ratio(out, portfolio_df, metrics)
    _save_annual_volatility(out, portfolio_df, metrics)
    _save_html_report(out, backtest_name, portfolio_df, metrics)

    print(f'  Done — {len(list(out.iterdir()))} files written.')
    return out


# ── CSV writers ───────────────────────────────────────────────────────────────

def _save_annual_returns(out: Path, portfolio_df: pd.DataFrame) -> None:
    nav = _get_nav(portfolio_df)
    annual = nav.resample('YE').last().pct_change().dropna()
    annual.index = annual.index.year
    annual.index.name = 'Year'
    df = annual.rename('Annual_Return').to_frame()
    df['Annual_Return_%'] = (df['Annual_Return'] * 100).round(2)
    df.to_csv(out / 'annual_returns.csv')


def _save_monthly_returns(out: Path, portfolio_df: pd.DataFrame) -> None:
    nav = _get_nav(portfolio_df)
    monthly = nav.resample('ME').last().pct_change().dropna()
    monthly.index.name = 'Date'
    df = monthly.rename('Monthly_Return').to_frame()
    df['Monthly_Return_%'] = (df['Monthly_Return'] * 100).round(2)
    df.to_csv(out / 'monthly_returns.csv')


def _save_portfolio_timeseries(out: Path, portfolio_df: pd.DataFrame) -> None:
    nav = _get_nav(portfolio_df)
    ts = pd.DataFrame({'NAV': nav})
    ts['Cumulative_Return']   = (nav / nav.iloc[0]) - 1
    ts['Daily_Return']        = nav.pct_change()
    ts['Drawdown']            = nav / nav.cummax() - 1
    ts.index.name = 'Date'
    ts.round(6).to_csv(out / 'portfolio_timeseries.csv')


def _save_portfolio_components(
    out: Path, allocations: list[dict], equity_df: pd.DataFrame
) -> None:
    close = equity_df['Close'].unstack('Ticker') if 'Close' in equity_df.columns else None
    rows  = []

    for i, alloc in enumerate(allocations):
        date    = alloc['date']
        weights = alloc.get('weights', {})
        if not weights:
            continue

        next_date = allocations[i + 1]['date'] if i + 1 < len(allocations) else None

        for ticker, target_w in weights.items():
            price_at_rebal = (
                close.loc[:date, ticker].dropna().iloc[-1]
                if close is not None and ticker in close.columns
                else np.nan
            )

            # Drift weight at next rebalance date
            if next_date is not None and close is not None and ticker in close.columns:
                price_at_next = close.loc[:next_date, ticker].dropna()
                price_at_next = price_at_next.iloc[-1] if not price_at_next.empty else np.nan
                price_chg     = (price_at_next / price_at_rebal - 1) if price_at_rebal else np.nan

                # Simple drift: w_drift = w_target * (1 + r_i) / (1 + r_portfolio)
                port_ret = sum(
                    weights.get(t, 0) * (
                        (close.loc[:next_date, t].dropna().iloc[-1] /
                         close.loc[:date, t].dropna().iloc[-1] - 1)
                        if t in close.columns
                        and not close.loc[:date, t].dropna().empty
                        and not close.loc[:next_date, t].dropna().empty
                        else 0
                    )
                    for t in weights
                )
                ticker_ret  = price_chg if not np.isnan(price_chg) else 0
                drift_w     = target_w * (1 + ticker_ret) / (1 + port_ret) if (1 + port_ret) else np.nan
            else:
                price_chg = np.nan
                drift_w   = np.nan

            rows.append({
                'Rebalance_Date':      date.date(),
                'Ticker':              ticker,
                'Target_Weight':       round(target_w, 6),
                'Price_At_Rebalance':  round(price_at_rebal, 4) if not np.isnan(price_at_rebal) else np.nan,
                'Price_Change':        round(price_chg, 6)      if not np.isnan(price_chg)      else np.nan,
                'Drift_Weight':        round(drift_w, 6)        if not np.isnan(drift_w)        else np.nan,
            })

    if rows:
        pd.DataFrame(rows).to_csv(out / 'portfolio_components.csv', index=False)


def _save_sharpe_ratio(out: Path, portfolio_df: pd.DataFrame, metrics: dict) -> None:
    nav  = _get_nav(portfolio_df)
    rets = nav.pct_change().dropna()

    rolling_sharpe = (
        rets.rolling(252)
        .apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else np.nan, raw=True)
    )

    df = pd.DataFrame({
        'Rolling_12M_Sharpe': rolling_sharpe.round(4),
    })
    df.index.name = 'Date'
    df['Full_Period_Sharpe'] = round(metrics.get('sharpe_ratio', np.nan), 4)
    df.to_csv(out / 'sharpe_ratio.csv')


def _save_annual_volatility(out: Path, portfolio_df: pd.DataFrame, metrics: dict) -> None:
    nav  = _get_nav(portfolio_df)
    rets = nav.pct_change().dropna()

    rolling_vol = rets.rolling(252).std() * np.sqrt(252)

    df = pd.DataFrame({
        'Rolling_12M_Volatility': rolling_vol.round(6),
    })
    df.index.name = 'Date'
    df['Full_Period_Volatility'] = round(metrics.get('annual_volatility', np.nan), 6)
    df.to_csv(out / 'annual_volatility.csv')


# ── HTML report ───────────────────────────────────────────────────────────────

def _save_html_report(
    out: Path, backtest_name: str,
    portfolio_df: pd.DataFrame, metrics: dict,
) -> None:
    nav     = _get_nav(portfolio_df)
    rets    = nav.pct_change().dropna()
    cum_ret = (nav / nav.iloc[0]) - 1
    dd      = nav / nav.cummax() - 1

    img_curve    = _chart_to_b64(_plot_return_curve(nav, cum_ret, dd))
    img_monthly  = _chart_to_b64(_plot_monthly_heatmap(rets))
    img_rolling  = _chart_to_b64(_plot_rolling_metrics(rets))

    metrics_html = _metrics_table(metrics, nav, rets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report — {backtest_name}</title>
<style>
  body      {{ font-family: 'Segoe UI', sans-serif; background:#1a1a2e; color:#ecf0f1; margin:0; padding:24px; }}
  h1        {{ color:#f1c40f; border-bottom:1px solid #444; padding-bottom:8px; }}
  h2        {{ color:#3498db; margin-top:40px; }}
  table     {{ border-collapse:collapse; width:100%; max-width:680px; margin-bottom:24px; }}
  th        {{ background:#16213e; color:#f1c40f; padding:8px 14px; text-align:left; font-size:13px; }}
  td        {{ padding:7px 14px; border-bottom:1px solid #2c3e60; font-size:13px; }}
  td.val    {{ text-align:right; font-family:monospace; }}
  .pos      {{ color:#2ecc71; }}
  .neg      {{ color:#e74c3c; }}
  img       {{ max-width:100%; border-radius:6px; margin:12px 0; }}
  .section  {{ max-width:1100px; margin:0 auto; }}
</style>
</head>
<body>
<div class="section">
  <h1>Backtest Report — {backtest_name}</h1>
  <p style="color:#aaa">{nav.index[0].date()} → {nav.index[-1].date()}</p>

  <h2>Key Metrics</h2>
  {metrics_html}

  <h2>Portfolio NAV &amp; Drawdown</h2>
  <img src="data:image/png;base64,{img_curve}" alt="Return curve">

  <h2>Rolling Sharpe &amp; Volatility</h2>
  <img src="data:image/png;base64,{img_rolling}" alt="Rolling metrics">

  <h2>Monthly Returns Heatmap</h2>
  <img src="data:image/png;base64,{img_monthly}" alt="Monthly heatmap">
</div>
</body>
</html>"""

    (out / 'report.html').write_text(html, encoding='utf-8')


# ── Chart helpers ─────────────────────────────────────────────────────────────

_DARK  = '#1a1a2e'
_PANEL = '#16213e'
_GRID  = '#2c3e60'

def _dark_fig(nrows=1, ncols=1, figsize=(14, 5), height_ratios=None):
    kw = {'height_ratios': height_ratios} if height_ratios else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw=kw)
    fig.patch.set_facecolor(_DARK)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(color=_GRID, linewidth=0.5, linestyle='--')
    return fig, axes


def _plot_return_curve(nav, cum_ret, dd):
    fig, (ax1, ax2) = _dark_fig(2, 1, figsize=(14, 7), height_ratios=[2, 1])
    fig.subplots_adjust(hspace=0.08)

    ax1.plot(cum_ret.index, cum_ret * 100, color='#f1c40f', linewidth=1.6)
    ax1.fill_between(cum_ret.index, 0, cum_ret * 100, alpha=0.15, color='#f1c40f')
    ax1.set_ylabel('Cumulative Return (%)', color='#cccccc')
    ax1.set_title('Portfolio Performance', color='#ecf0f1', fontsize=13, pad=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.tick_params(labelbottom=False)

    ax2.fill_between(dd.index, 0, dd * 100, color='#e74c3c', alpha=0.7)
    ax2.plot(dd.index, dd * 100, color='#e74c3c', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)', color='#cccccc')
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.set_xlabel('Date', color='#cccccc')

    return fig


def _plot_rolling_metrics(rets):
    fig, (ax1, ax2) = _dark_fig(2, 1, figsize=(14, 6), height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.08)

    rolling_sharpe = (
        rets.rolling(252)
        .apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else np.nan, raw=True)
    )
    rolling_vol = rets.rolling(252).std() * np.sqrt(252) * 100

    ax1.plot(rolling_sharpe.index, rolling_sharpe, color='#3498db', linewidth=1.4)
    ax1.axhline(0, color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax1.axhline(1, color='#2ecc71', linewidth=0.8, linestyle=':')
    ax1.set_ylabel('Rolling Sharpe (12M)', color='#cccccc')
    ax1.tick_params(labelbottom=False)

    ax2.plot(rolling_vol.index, rolling_vol, color='#f39c12', linewidth=1.4)
    ax2.set_ylabel('Rolling Vol % (12M)', color='#cccccc')
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.set_xlabel('Date', color='#cccccc')

    return fig


def _plot_monthly_heatmap(rets):
    monthly = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    pivot   = pd.DataFrame({
        'Year':  monthly.index.year,
        'Month': monthly.index.month,
        'Ret':   monthly.values,
    }).pivot(index='Year', columns='Month', values='Ret')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = _dark_fig(figsize=(14, max(4, len(pivot) * 0.45)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im   = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, color='#cccccc', fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, color='#cccccc', fontsize=9)
    ax.set_title('Monthly Returns Heatmap', color='#ecf0f1', fontsize=12, pad=10)

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            if not np.isnan(val):
                ax.text(x, y, f'{val*100:.1f}%', ha='center', va='center',
                        fontsize=7, color='#111' if abs(val) < vmax * 0.6 else '#eee')

    plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    return fig


def _chart_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _metrics_table(metrics: dict, nav: pd.Series, rets: pd.Series) -> str:
    total_ret  = (nav.iloc[-1] / nav.iloc[0]) - 1
    n_years    = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr       = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    max_dd     = (nav / nav.cummax() - 1).min()
    calmar     = cagr / abs(max_dd) if max_dd != 0 else np.nan
    win_rate   = (rets > 0).mean()

    rows = [
        ('Total Return',       f'{total_ret*100:.2f}%',  total_ret  > 0),
        ('CAGR',               f'{cagr*100:.2f}%',       cagr       > 0),
        ('Annual Volatility',  f'{metrics.get("annual_volatility", np.nan)*100:.2f}%', None),
        ('Sharpe Ratio',       f'{metrics.get("sharpe_ratio", np.nan):.3f}',
                               metrics.get("sharpe_ratio", 0) > 1),
        ('Max Drawdown',       f'{max_dd*100:.2f}%',     False),
        ('Calmar Ratio',       f'{calmar:.3f}',           calmar     > 1),
        ('Daily Win Rate',     f'{win_rate*100:.1f}%',   win_rate   > 0.5),
    ]

    def _cls(pos):
        if pos is None: return ''
        return 'class="val pos"' if pos else 'class="val neg"'

    header = '<tr><th>Metric</th><th>Value</th></tr>'
    body   = ''.join(
        f'<tr><td>{label}</td><td {_cls(pos)}>{val}</td></tr>'
        for label, val, pos in rows
    )
    return f'<table>{header}{body}</table>'


# ── Utility ───────────────────────────────────────────────────────────────────

def _get_nav(portfolio_df: pd.DataFrame) -> pd.Series:
    for col in ('nav', 'NAV', 'portfolio_value', 'value'):
        if col in portfolio_df.columns:
            return portfolio_df[col].dropna()
    return portfolio_df.iloc[:, 0].dropna()
