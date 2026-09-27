"""
plotting.py
===========
Interactive chart suite for backtest results.

All functions display figures on screen (interactive backend).
For file-saving / HTML embedding, see output.py.

Public API
----------
  plot(portfolio_df, metrics, regimes)   — full dashboard (calls all below)

  plot_performance(portfolio_df, metrics, regimes)   — NAV + drawdown + regime shading
  plot_rolling_metrics(portfolio_df, metrics)        — rolling Sharpe & volatility
  plot_annual_returns(portfolio_df)                  — annual bar chart
  plot_monthly_heatmap(portfolio_df)                 — monthly returns pivot heatmap
  plot_return_distribution(portfolio_df)             — histogram + normal overlay
  plot_regime_overlay(portfolio_df, regimes)         — each regime's state over time
  plot_metrics_summary(metrics)                      — text-only key metrics panel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as scipy_stats

import Finance.Backtesting_engine.risk_metrics_new as rm

# ── Visual theme ──────────────────────────────────────────────────────────────
_BG       = '#1a1a2e'
_PANEL    = '#16213e'
_GRID     = '#2c3e60'
_TEXT     = '#ecf0f1'
_MUTED    = '#95a5a6'
_YELLOW   = '#f1c40f'
_GREEN    = '#2ecc71'
_RED      = '#e74c3c'
_BLUE     = '#3498db'
_ORANGE   = '#f39c12'
_PURPLE   = '#9b59b6'

_REGIME_COLORS = {
    # Equity
    'Bull':    _GREEN,
    'Neutral': _ORANGE,
    'Bear':    _RED,
    # Bond
    'Bond Bull':    _GREEN,
    'Bond Neutral': _ORANGE,
    'Bond Bear':    _RED,
    # Commodity
    'Commodity Bull':    _GREEN,
    'Commodity Neutral': _ORANGE,
    'Commodity Bear':    _RED,
    # Forex
    'Risk On': _GREEN,
    'Stress':  _RED,
    # Growth
    'Expansion':   _GREEN,
    'Slowdown':    _ORANGE,
    'Recovery':    _BLUE,
    'Contraction': _RED,
    # Fallback
    'Unknown': '#555577',
}


def _style_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
    return ax


def _dark_fig(nrows=1, ncols=1, figsize=(16, 8), title='', **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(_BG)
    if title:
        fig.suptitle(title, color=_TEXT, fontsize=13, y=0.98)
    axs = axes if hasattr(axes, '__iter__') else [axes]
    for ax in np.array(axs).flat:
        _style_ax(ax)
    return fig, axes


def _shade_regimes(ax, regime_series: pd.Series, color_map: dict = _REGIME_COLORS,
                   alpha: float = 0.20):
    """Shade axis background by regime label."""
    prev, start = None, regime_series.index[0]
    for date, val in regime_series.items():
        if val != prev:
            if prev is not None:
                ax.axvspan(start, date,
                           color=color_map.get(prev, '#555577'),
                           alpha=alpha, linewidth=0)
            start, prev = date, val
    if prev is not None:
        ax.axvspan(start, regime_series.index[-1],
                   color=color_map.get(prev, '#555577'),
                   alpha=alpha, linewidth=0)


def _pct_fmt(ax, axis='y'):
    fmt = mticker.FuncFormatter(lambda v, _: f'{v*100:.0f}%')
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Performance chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance(
    portfolio_df,
    metrics:  dict | None = None,
    regimes:  dict | None = None,
    benchmark_df=None,
    title: str = 'Portfolio Performance',
) -> plt.Figure:
    """
    4-panel performance dashboard:
      Top    : Cumulative return (+ optional benchmark) with regime shading
      Panel 2: Drawdown
      Panel 3: Rolling 12M Sharpe
      Panel 4: Rolling 12M Volatility
    """
    nav  = rm._to_nav(portfolio_df)
    rets = rm._to_returns(portfolio_df)
    cum  = (nav / nav.iloc[0]) - 1
    dd   = rm.drawdown_series(nav)

    rf   = metrics.get('rf_rate_used', rm.RF_FALLBACK) if metrics else rm.RF_FALLBACK
    roll_sh  = rm.rolling_sharpe(rets, rf_annual=rf)
    roll_vol = rm.rolling_volatility(rets)

    fig, axes = _dark_fig(
        4, 1, figsize=(16, 14), title=title,
        gridspec_kw={'height_ratios': [3, 1.2, 1.2, 1.2], 'hspace': 0.06},
        sharex=True,
    )
    ax1, ax2, ax3, ax4 = axes

    # ── Panel 1: Cumulative return ────────────────────────────────────────────
    equity_regime = None
    if regimes:
        for key in ('equity', 'growth'):
            df = regimes.get(key)
            if df is not None and not df.empty:
                col = next((c for c in ('global_regime', 'growth_regime', 'regime')
                            if c in df.columns), None)
                if col:
                    equity_regime = df[col].reindex(cum.index, method='ffill')
                    break

    if equity_regime is not None:
        _shade_regimes(ax1, equity_regime)

    ax1.plot(cum.index, cum * 100, color=_YELLOW, linewidth=1.8, label='Portfolio', zorder=3)
    ax1.fill_between(cum.index, 0, cum * 100, alpha=0.08, color=_YELLOW)

    if benchmark_df is not None:
        bench_nav  = rm._to_nav(benchmark_df).reindex(nav.index, method='ffill').dropna()
        bench_cum  = (bench_nav / bench_nav.iloc[0]) - 1
        ax1.plot(bench_cum.index, bench_cum * 100,
                 color=_MUTED, linewidth=1.2, linestyle='--', label='Benchmark', zorder=2)

    ax1.axhline(0, color=_GRID, linewidth=0.8)
    _pct_fmt(ax1)
    ax1.set_ylabel('Cumulative Return', color=_MUTED)
    ax1.legend(loc='upper left', framealpha=0.3, labelcolor=_TEXT,
               facecolor=_BG, fontsize=9)

    if metrics:
        cagr_s  = f"CAGR {metrics['cagr']*100:.1f}%"
        sharpe_s = f"Sharpe {metrics['sharpe_ratio']:.2f}"
        mdd_s   = f"Max DD {metrics['max_drawdown']*100:.1f}%"
        ax1.set_title(f'{title}   |   {cagr_s}   {sharpe_s}   {mdd_s}',
                      color=_TEXT, fontsize=11, pad=8)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2.fill_between(dd.index, 0, dd * 100, color=_RED, alpha=0.6)
    ax2.plot(dd.index, dd * 100, color=_RED, linewidth=0.8)
    _pct_fmt(ax2)
    ax2.set_ylabel('Drawdown', color=_MUTED)

    # ── Panel 3: Rolling Sharpe ───────────────────────────────────────────────
    ax3.plot(roll_sh.index, roll_sh, color=_BLUE, linewidth=1.2)
    ax3.axhline(0, color=_GRID, linewidth=0.8)
    ax3.axhline(1, color=_GREEN, linewidth=0.6, linestyle=':')
    ax3.set_ylabel('Sharpe (12M)', color=_MUTED)

    # ── Panel 4: Rolling Volatility ───────────────────────────────────────────
    ax4.plot(roll_vol.index, roll_vol * 100, color=_ORANGE, linewidth=1.2)
    _pct_fmt(ax4)
    ax4.set_ylabel('Volatility (12M)', color=_MUTED)
    ax4.set_xlabel('Date', color=_MUTED)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Rolling metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_metrics(portfolio_df, metrics: dict | None = None) -> plt.Figure:
    """Rolling Sharpe, Sortino, Calmar, and volatility over time."""
    nav  = rm._to_nav(portfolio_df)
    rets = rm._to_returns(portfolio_df)
    rf   = metrics.get('rf_rate_used', rm.RF_FALLBACK) if metrics else rm.RF_FALLBACK

    roll_sh  = rm.rolling_sharpe(rets, rf_annual=rf)
    roll_vol = rm.rolling_volatility(rets)
    roll_dd  = rm.rolling_drawdown(nav)

    # rolling Sortino approximation
    roll_sort = (
        rets.rolling(rm.PERIODS_PER_YEAR)
        .apply(lambda x: rm.sortino_ratio(pd.Series(x), rf), raw=False)
    )

    fig, axes = _dark_fig(
        3, 1, figsize=(16, 10), title='Rolling Risk Metrics (12-Month Window)',
        gridspec_kw={'hspace': 0.10}, sharex=True,
    )
    ax1, ax2, ax3 = axes

    ax1.plot(roll_sh.index, roll_sh, color=_BLUE, linewidth=1.3, label='Sharpe')
    ax1.plot(roll_sort.index, roll_sort, color=_PURPLE, linewidth=1.0,
             linestyle='--', alpha=0.8, label='Sortino')
    ax1.axhline(0, color=_GRID, linewidth=0.8)
    ax1.axhline(1, color=_GREEN, linewidth=0.6, linestyle=':')
    ax1.set_ylabel('Ratio', color=_MUTED)
    ax1.legend(loc='upper left', framealpha=0.3, labelcolor=_TEXT,
               facecolor=_BG, fontsize=9)
    ax1.set_title('Rolling Sharpe & Sortino', color=_TEXT, fontsize=10)

    ax2.plot(roll_vol.index, roll_vol * 100, color=_ORANGE, linewidth=1.3)
    ax2.fill_between(roll_vol.index, 0, roll_vol * 100, alpha=0.15, color=_ORANGE)
    _pct_fmt(ax2)
    ax2.set_ylabel('Volatility', color=_MUTED)
    ax2.set_title('Rolling Annualised Volatility', color=_TEXT, fontsize=10)

    ax3.fill_between(roll_dd.index, 0, roll_dd * 100, color=_RED, alpha=0.55)
    ax3.plot(roll_dd.index, roll_dd * 100, color=_RED, linewidth=0.8)
    _pct_fmt(ax3)
    ax3.set_ylabel('Drawdown', color=_MUTED)
    ax3.set_xlabel('Date', color=_MUTED)
    ax3.set_title('Rolling Max Drawdown (12M window)', color=_TEXT, fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Annual returns bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_annual_returns(portfolio_df, benchmark_df=None) -> plt.Figure:
    """Grouped bar chart of calendar-year returns."""
    rets = rm._to_returns(portfolio_df)
    ar   = rm.annual_returns(rets)
    years = ar.index.year

    fig, ax = _dark_fig(figsize=(14, 6), title='Annual Returns')
    ax = ax

    width = 0.35 if benchmark_df is not None else 0.6
    x     = np.arange(len(years))

    colors = [_GREEN if v >= 0 else _RED for v in ar.values]
    bars   = ax.bar(x, ar.values * 100, width=width, color=colors,
                    alpha=0.85, zorder=3)

    if benchmark_df is not None:
        bench_rets = rm._to_returns(benchmark_df)
        bench_ar   = rm.annual_returns(bench_rets).reindex(ar.index)
        ax.bar(x + width, bench_ar.values * 100, width=width,
               color=_MUTED, alpha=0.55, label='Benchmark', zorder=3)
        ax.legend(framealpha=0.3, labelcolor=_TEXT, facecolor=_BG, fontsize=9)

    # value labels on bars
    for bar, val in zip(bars, ar.values):
        ypos = bar.get_height() + (0.3 if val >= 0 else -1.5)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{val*100:.1f}%', ha='center', va='bottom',
                color=_TEXT, fontsize=8)

    ax.set_xticks(x + (width / 2 if benchmark_df is not None else 0))
    ax.set_xticklabels(years, color=_MUTED)
    ax.axhline(0, color=_GRID, linewidth=0.8)
    _pct_fmt(ax)
    ax.set_ylabel('Return (%)', color=_MUTED)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Monthly returns heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(portfolio_df) -> plt.Figure:
    """Calendar heatmap — rows = year, columns = month."""
    rets  = rm._to_returns(portfolio_df)
    pivot = rm.monthly_return_table(rets)

    n_years = len(pivot)
    fig, ax = _dark_fig(figsize=(14, max(4, n_years * 0.45)),
                        title='Monthly Returns Heatmap')
    ax = ax

    vals = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(vals))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(vals, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, color=_MUTED, fontsize=9)
    ax.set_yticks(range(n_years))
    ax.set_yticklabels(pivot.index, color=_MUTED, fontsize=9)

    for y in range(vals.shape[0]):
        for x in range(vals.shape[1]):
            v = vals[y, x]
            if not np.isnan(v):
                text_color = '#111' if abs(v) < vmax * 0.5 else '#eee'
                ax.text(x, y, f'{v*100:.1f}%', ha='center', va='center',
                        fontsize=7.5, color=text_color, fontweight='bold')

    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v*100:.0f}%')
    )
    cb.ax.tick_params(colors=_MUTED, labelsize=8)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Return distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_return_distribution(portfolio_df, benchmark_df=None) -> plt.Figure:
    """Histogram of daily returns with fitted normal distribution overlay."""
    rets = rm._to_returns(portfolio_df)

    fig, axes = _dark_fig(1, 2, figsize=(15, 5),
                          title='Return Distribution')
    ax_hist, ax_qq = axes

    # ── Histogram ─────────────────────────────────────────────────────────────
    n_bins = min(80, max(30, len(rets) // 20))
    ax_hist.hist(rets * 100, bins=n_bins, color=_BLUE, alpha=0.7,
                 density=True, label='Daily returns', zorder=3)

    mu, sigma = rets.mean() * 100, rets.std() * 100
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax_hist.plot(x, scipy_stats.norm.pdf(x, mu, sigma),
                 color=_YELLOW, linewidth=1.8, label='Normal fit', zorder=4)

    # VaR lines
    var95 = rm.var_historical(rets)
    cvar95 = rm.cvar_historical(rets)
    ax_hist.axvline(var95 * 100, color=_ORANGE, linewidth=1.2, linestyle='--',
                    label=f'VaR 95%: {var95*100:.2f}%')
    ax_hist.axvline(cvar95 * 100, color=_RED, linewidth=1.2, linestyle='--',
                    label=f'CVaR 95%: {cvar95*100:.2f}%')

    if benchmark_df is not None:
        bench_rets = rm._to_returns(benchmark_df)
        ax_hist.hist(bench_rets * 100, bins=n_bins, color=_MUTED, alpha=0.35,
                     density=True, label='Benchmark', zorder=2)

    ax_hist.set_xlabel('Daily Return (%)', color=_MUTED)
    ax_hist.set_ylabel('Density', color=_MUTED)
    ax_hist.set_title('Distribution of Daily Returns', color=_TEXT, fontsize=10)
    skew = rm.skewness(rets)
    kurt = rm.excess_kurtosis(rets)
    ax_hist.set_title(
        f'Daily Return Distribution   Skew={skew:.2f}  Kurt={kurt:.2f}',
        color=_TEXT, fontsize=10,
    )
    ax_hist.legend(framealpha=0.3, labelcolor=_TEXT, facecolor=_BG, fontsize=8)

    # ── Q-Q plot ──────────────────────────────────────────────────────────────
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(rets, dist='norm')
    ax_qq.scatter(osm, osr * 100, color=_BLUE, s=4, alpha=0.5, zorder=3)
    line_x = np.array([osm[0], osm[-1]])
    ax_qq.plot(line_x, (slope * line_x + intercept) * 100,
               color=_YELLOW, linewidth=1.5, zorder=4, label=f'R²={r**2:.3f}')
    ax_qq.set_xlabel('Theoretical Quantiles', color=_MUTED)
    ax_qq.set_ylabel('Sample Quantiles (%)', color=_MUTED)
    ax_qq.set_title('Q-Q Plot vs Normal', color=_TEXT, fontsize=10)
    ax_qq.legend(framealpha=0.3, labelcolor=_TEXT, facecolor=_BG, fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Regime overlay timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_overlay(portfolio_df, regimes: dict) -> plt.Figure:
    """
    Horizontal timeline bar for each regime — shows when each was Bull/Bear/etc.
    Also overlays portfolio cumulative return for reference.
    """
    nav = rm._to_nav(portfolio_df)
    cum = (nav / nav.iloc[0]) - 1

    _REGIME_COL = {
        'equity':        'global_regime',
        'bond':          'bond_regime',
        'commodity':     'commodity_regime',
        'forex':         'forex_regime',
        'crypto':        'crypto_regime',
        'growth':        'growth_regime',
        'inflation':     'inflation_regime',
        'liquidity':     'liquidity_regime',
        'risk_appetite': 'risk_regime',
        'hidden_markov': 'hmm_regime',
    }

    active = {}
    for name, df in regimes.items():
        if df is None or df.empty:
            continue
        col = _REGIME_COL.get(name)
        if col and col in df.columns:
            active[name] = df[col].reindex(nav.index, method='ffill').dropna()

    if not active:
        print('[plotting] No regime data to overlay.')
        return None

    n = len(active)
    fig = plt.figure(figsize=(16, 4 + n * 0.8), facecolor=_BG)
    gs  = gridspec.GridSpec(n + 1, 1, hspace=0.05,
                            height_ratios=[2] + [0.6] * n)

    # Cumulative return on top
    ax_ret = fig.add_subplot(gs[0])
    _style_ax(ax_ret)
    ax_ret.plot(cum.index, cum * 100, color=_YELLOW, linewidth=1.5)
    ax_ret.fill_between(cum.index, 0, cum * 100, alpha=0.10, color=_YELLOW)
    _pct_fmt(ax_ret)
    ax_ret.set_ylabel('Cum. Return', color=_MUTED)
    ax_ret.set_title('Portfolio + Regime Timeline', color=_TEXT, fontsize=11)
    ax_ret.tick_params(labelbottom=False)

    # Regime bars
    for i, (name, series) in enumerate(active.items()):
        ax = fig.add_subplot(gs[i + 1], sharex=ax_ret)
        _style_ax(ax)
        ax.set_yticks([])
        ax.set_ylabel(name.replace('_', ' ').title(),
                      color=_MUTED, fontsize=8, rotation=0,
                      labelpad=60, va='center')
        ax.grid(False)

        _shade_regimes(ax, series, alpha=0.85)

        # Unique legend patches for this regime
        seen = series.dropna().unique()
        patches = [mpatches.Patch(facecolor=_REGIME_COLORS.get(r, '#555577'),
                                  label=r) for r in seen]
        ax.legend(handles=patches, loc='lower right', framealpha=0.25,
                  labelcolor=_TEXT, facecolor=_BG, fontsize=7,
                  ncol=min(4, len(patches)), handlelength=1)

        if i < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Date', color=_MUTED)

    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Metrics summary panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_summary(metrics: dict) -> plt.Figure:
    """Text-based figure showing key metrics in a 3-column layout."""
    groups = [
        ('Returns', [
            ('Total Return',      f'{metrics["total_return"]*100:.2f}%'),
            ('CAGR',              f'{metrics["cagr"]*100:.2f}%'),
            ('Best Year',         f'{metrics["best_year"]*100:.2f}% ({metrics["best_year_int"]})'),
            ('Worst Year',        f'{metrics["worst_year"]*100:.2f}% ({metrics["worst_year_int"]})'),
            ('Best Day',          f'{metrics["best_day"]*100:.2f}%'),
            ('Worst Day',         f'{metrics["worst_day"]*100:.2f}%'),
        ]),
        ('Risk', [
            ('Annual Volatility', f'{metrics["annual_volatility"]*100:.2f}%'),
            ('Downside Dev.',     f'{metrics["downside_deviation"]*100:.2f}%'),
            ('Max Drawdown',      f'{metrics["max_drawdown"]*100:.2f}%'),
            ('Avg Drawdown',      f'{metrics["avg_drawdown"]*100:.2f}%'),
            ('Max DD Duration',   f'{metrics["max_dd_duration_days"]} days'),
            ('Recovery Factor',   f'{metrics["recovery_factor"]:.2f}'),
        ]),
        ('Risk-Adjusted', [
            ('Sharpe Ratio',      f'{metrics["sharpe_ratio"]:.3f}'),
            ('Sortino Ratio',     f'{metrics["sortino_ratio"]:.3f}'),
            ('RF Rate Used',      f'{metrics["rf_rate_used"]*100:.2f}%'),
            ('Win Rate',          f'{metrics["win_rate"]*100:.1f}%'),
            ('Profit Factor',     f'{metrics["profit_factor"]:.3f}'),
            ('Tail Ratio',        f'{metrics["tail_ratio"]:.3f}'),
        ]),
        ('Tail Risk', [
            ('VaR 95% (hist)',    f'{metrics["var_historical"]*100:.2f}%'),
            ('CVaR 95% (hist)',   f'{metrics["cvar_historical"]*100:.2f}%'),
            ('VaR 95% (param)',   f'{metrics["var_parametric"]*100:.2f}%'),
            ('CVaR 95% (param)',  f'{metrics["cvar_parametric"]*100:.2f}%'),
            ('Skewness',          f'{metrics["skewness"]:.3f}'),
            ('Excess Kurtosis',   f'{metrics["excess_kurtosis"]:.3f}'),
        ]),
        ('Statistics', [
            ('Best Day',          f'{metrics["best_day"]*100:.2f}%'),
            ('Worst Day',         f'{metrics["worst_day"]*100:.2f}%'),
            ('Best Month',        f'{metrics["best_month"]*100:.2f}%'),
            ('Worst Month',       f'{metrics["worst_month"]*100:.2f}%'),
            ('Max Consec. Wins',  f'{metrics["max_consecutive_wins"]}'),
            ('Max Consec. Loss',  f'{metrics["max_consecutive_losses"]}'),
        ]),
        ('Benchmark', [
            ('Beta',              f'{metrics.get("beta", float("nan")):.3f}'),
            ('Alpha',             f'{metrics.get("alpha", float("nan"))*100:.2f}%'
                                  if not np.isnan(metrics.get("alpha", float("nan"))) else 'n/a'),
            ('R²',                f'{metrics.get("r_squared", float("nan")):.3f}'),
            ('Correlation',       f'{metrics.get("correlation", float("nan")):.3f}'),
            ('Info Ratio',        f'{metrics.get("information_ratio", float("nan")):.3f}'),
            ('Tracking Error',    f'{metrics.get("tracking_error", float("nan"))*100:.2f}%'
                                  if not np.isnan(metrics.get("tracking_error", float("nan"))) else 'n/a'),
        ]),
    ]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        f'Risk Metrics Summary   {metrics["start_date"]} -> {metrics["end_date"]}',
        color=_TEXT, fontsize=12, y=0.98,
    )

    for ax, (group_name, rows) in zip(axes.flat, groups):
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.5, 0.96, group_name, color=_YELLOW, fontsize=10,
                fontweight='bold', ha='center', va='top',
                transform=ax.transAxes)
        ax.axhline(0.90, color=_GRID, linewidth=0.8)  # y in axes coords (0-1 xlim)

        for j, (label, value) in enumerate(rows):
            y = 0.82 - j * 0.135
            ax.text(0.05, y, label, color=_MUTED, fontsize=9,
                    va='center', transform=ax.transAxes)
            ax.text(0.95, y, value, color=_TEXT, fontsize=9,
                    va='center', ha='right', transform=ax.transAxes,
                    fontfamily='monospace')

        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def plot(
    portfolio_df,
    metrics:      dict | None = None,
    regimes:      dict | None = None,
    benchmark_df=None,
    title: str = 'Backtest Results',
) -> None:
    """
    Full dashboard — displays all charts in sequence.

    Parameters
    ----------
    portfolio_df  : DataFrame with NAV / daily-return column (DatetimeIndex).
    metrics       : dict returned by risk_metrics.compute(). If None, computed here.
    regimes       : dict returned by load_regime_data(). Optional.
    benchmark_df  : Optional benchmark DataFrame (same format as portfolio_df).
    title         : Window / suptitle prefix.
    """
    if metrics is None:
        metrics = rm.compute(portfolio_df, benchmark_df=benchmark_df)

    print(rm.summary(metrics))

    plot_performance(portfolio_df, metrics, regimes, benchmark_df, title=title)
    plot_rolling_metrics(portfolio_df, metrics)
    plot_annual_returns(portfolio_df, benchmark_df)
    plot_monthly_heatmap(portfolio_df)
    plot_return_distribution(portfolio_df, benchmark_df)
    plot_metrics_summary(metrics)

    if regimes:
        plot_regime_overlay(portfolio_df, regimes)
