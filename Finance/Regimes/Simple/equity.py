import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Finance.Regimes.config import (
    EQUITY_BULL_THRESHOLD, EQUITY_BEAR_THRESHOLD,
    EQUITY_MIN_STOCKS, EQUITY_MA_WINDOW,
    EQUITY_SMOOTH_WINDOW, EQUITY_CONFIRM_WINDOW, EQUITY_CONFIRM_MIN,
)

_SCORE_COLORS = {1.0: '#2ecc71', 0.0: '#f39c12', -1.0: '#e74c3c'}
_SCORE_LABELS = {1: 'Bull', 0: 'Neutral', -1: 'Bear'}


def _hysteresis(score_series: pd.Series, bull_entry: float, bear_entry: float,
                bull_exit: float | None = None, bear_exit: float | None = None) -> list:
    """Generic hysteresis filter — returns 'Bull' / 'Neutral' / 'Bear' / 'Unknown'.

    bull_exit / bear_exit set the dead-band: once in Bull the regime only exits when
    the score drops *below* bull_exit (≤ bull_entry). Omit for a symmetric threshold.
    """
    bull_exit = bull_exit if bull_exit is not None else bull_entry
    bear_exit = bear_exit if bear_exit is not None else bear_entry
    regimes, state = [], None

    for score in score_series:
        if pd.isna(score):
            regimes.append('Unknown')
            continue
        if state is None:
            if   score >= bull_entry: state = 'Bull'
            elif score <= bear_entry: state = 'Bear'
            else:                     state = 'Neutral'
        elif state == 'Bull':
            if   score <= bear_entry: state = 'Bear'
            elif score <  bull_exit:  state = 'Neutral'
        elif state == 'Bear':
            if   score >= bull_entry: state = 'Bull'
            elif score >  bear_exit:  state = 'Neutral'
        else:  # Neutral
            if   score >= bull_entry: state = 'Bull'
            elif score <= bear_entry: state = 'Bear'
        regimes.append(state)
    return regimes


def _shade_score(ax, score_series: pd.Series, alpha: float = 0.25) -> None:
    """Shade axis background by score value (1 / 0 / -1)."""
    prev, start = None, score_series.index[0]
    for date, val in score_series.items():
        if val != prev:
            if prev is not None:
                ax.axvspan(start, date, color=_SCORE_COLORS.get(prev, '#555577'),
                           alpha=alpha, linewidth=0)
            start, prev = date, val
    if prev is not None:
        ax.axvspan(start, score_series.index[-1],
                   color=_SCORE_COLORS.get(prev, '#555577'), alpha=alpha, linewidth=0)


def plot_equity_regions(continent_dfs: dict,
                        regions: tuple[str, ...] = ('North America', 'Europe')) -> None:
    """Plot breadth + regime shading for the given regions (default: NA and EU)."""
    fig, axes = plt.subplots(len(regions), 1, figsize=(16, 5 * len(regions)), sharex=True)
    if len(regions) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1a1a2e')

    for ax, region in zip(axes, regions):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

        cdf = continent_dfs.get(region)
        if cdf is None:
            ax.set_title(f'{region} — no data', color='#ecf0f1')
            continue

        data = cdf.dropna(subset=['breadth_smooth'])

        _shade_score(ax, data['score'])

        ax.plot(data.index, data['breadth'] * 100,
                color='#ffffff', linewidth=0.6, alpha=0.35, label='Breadth (daily)')
        ax.plot(data.index, data['breadth_smooth'] * 100,
                color='#f1c40f', linewidth=1.8, label='Breadth (smooth)')
        ax.axhline(EQUITY_BULL_THRESHOLD * 100, color='#2ecc71',
                   linewidth=1, linestyle=':', alpha=0.7, label=f'Bull {EQUITY_BULL_THRESHOLD:.0%}')
        ax.axhline(EQUITY_BEAR_THRESHOLD * 100, color='#e74c3c',
                   linewidth=1, linestyle=':', alpha=0.7, label=f'Bear {EQUITY_BEAR_THRESHOLD:.0%}')

        latest = data.iloc[-1]
        s      = latest['score']
        score  = int(s) if pd.notna(s) else '?'
        regime = _SCORE_LABELS.get(score, '?')
        b      = latest['breadth_smooth']
        date_s = data.index[-1].strftime('%Y-%m-%d')
        ax.set_title(
            f'{region}   Score: {score}  ({regime})   '
            f'Breadth: {b:.1%}   [{date_s}]',
            color='#ecf0f1', fontsize=12, pad=8,
        )

        patches = [mpatches.Patch(facecolor=c, alpha=0.6, label=l)
                   for l, c in [('Bull', '#2ecc71'), ('Neutral', '#f39c12'), ('Bear', '#e74c3c')]]
        line_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=line_handles + patches, loc='upper left',
                  framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e', fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel('% Above MA200', color='#cccccc')

    axes[-1].set_xlabel('Date', color='#cccccc')
    plt.tight_layout()
    plt.show()


def compute_equity_regime(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compute equity market regime per continent and globally.

    Parameters
    ----------
    df : DataFrame with a (Date, Ticker) MultiIndex and columns 'Close', 'Continent'

    Returns
    -------
    continent_dfs : dict {continent_name → DataFrame}
        Each DataFrame (Date index) has:
          n_available    — tickers with a valid MA on that day
          n_above_ma     — tickers whose close > MA
          breadth        — n_above_ma / n_available  (0–1, NaN if < EQUITY_MIN_STOCKS)
          breadth_smooth — EQUITY_SMOOTH_WINDOW rolling mean of breadth
          score          — confirmed regime:  1 (Bull) | 0 (Neutral) | -1 (Bear)

    global_df : DataFrame (Date index)
          global_score   —  1 | 0 | -1
          global_regime  — 'Bull' | 'Neutral' | 'Bear' | 'Unknown'

    Global logic
    ------------
    Bull    : ALL continents with valid data have score == 1
    Bear    : ALL continents with valid data have score == -1
    Neutral : any continent disagrees
    """
    close         = df['Close'].unstack('Ticker')
    continent_map = df['Continent'].groupby(level='Ticker').last()

    # ── MA per ticker (one pass for all tickers) ──────────────────────────────
    ma = (
        df['Close'].dropna()
        .groupby(level='Ticker')
        .transform(lambda s: s.rolling(EQUITY_MA_WINDOW, min_periods=EQUITY_MA_WINDOW).mean())
        .reindex(df.index)
        .unstack('Ticker')
    )
    valid = ma.notna()           # ticker has enough history for MA on this day
    above = (close > ma) & valid # ticker is above its MA on this day

    continent_dfs = {}
    score_matrix  = pd.DataFrame(index=close.index)

    # ── Per-continent breadth + hysteresis + confirmation ─────────────────────
    for continent in sorted(continent_map.unique()):
        tickers = continent_map[continent_map == continent].index.intersection(close.columns)
        if tickers.empty:
            continue

        n_available = valid[tickers].sum(axis=1)
        n_above_ma  = above[tickers].sum(axis=1)

        # Breadth: ignore days where too few stocks have MA history
        breadth = (
            (n_above_ma / n_available.replace(0, np.nan))
            .where(n_available >= EQUITY_MIN_STOCKS)
            .ffill(limit=3)   # bridge 1-3 day holiday gaps without masking real moves
        )
        breadth_smooth = (
            breadth.dropna()
            .rolling(EQUITY_SMOOTH_WINDOW, min_periods=EQUITY_SMOOTH_WINDOW // 2)
            .mean()
            .reindex(close.index)
        )

        raw = pd.Series(
            _hysteresis(
                breadth_smooth,
                bull_entry=EQUITY_BULL_THRESHOLD,
                bear_entry=EQUITY_BEAR_THRESHOLD,
            ),
            index=close.index,
        ).map({'Bull': 1, 'Neutral': 0, 'Bear': -1, 'Unknown': np.nan})

        # Confirmation: only switch regime when the new state holds for at least
        # EQUITY_CONFIRM_MIN of the last EQUITY_CONFIRM_WINDOW trading days.
        # Runs on the non-NaN subset so weekends don't dilute the window.
        trading_idx = breadth_smooth.dropna().index
        confirmed = (
            raw.loc[trading_idx]
            .rolling(EQUITY_CONFIRM_WINDOW)
            .apply(lambda x: x[-1] if (x == x[-1]).sum() >= EQUITY_CONFIRM_MIN else np.nan, raw=True)
            .ffill()
            .reindex(close.index)
            .ffill()   # forward-fill weekends
        )

        continent_dfs[continent] = pd.DataFrame({
            'n_available':    n_available,
            'n_above_ma':     n_above_ma,
            'breadth':        breadth,
            'breadth_smooth': breadth_smooth,
            'score':          confirmed,
        })
        score_matrix[continent] = confirmed

    # ── Global regime ─────────────────────────────────────────────────────────
    def _global_score(row: pd.Series) -> float:
        known = row.dropna()
        if known.empty:
            return np.nan
        if (known == 1).all():
            return  1.0
        if (known == -1).all():
            return -1.0
        return 0.0

    global_score  = score_matrix.apply(_global_score, axis=1)
    global_regime = global_score.map({1.0: 'Bull', 0.0: 'Neutral', -1.0: 'Bear'}).fillna('Unknown')
    global_df     = pd.DataFrame({'global_score': global_score, 'global_regime': global_regime})

    # ── Snapshot print ────────────────────────────────────────────────────────
    latest_date = score_matrix.dropna(how='all').index[-1]
    print(f'\n=== Equity Regime Snapshot ({latest_date.strftime("%Y-%m-%d")}) ===\n')
    print(f'  {"Region":<22}  {"Score":>5}  {"Regime":<8}  Breadth')
    print(f'  {"-"*55}')
    for continent in sorted(continent_dfs):
        row    = continent_dfs[continent].loc[latest_date]
        s      = row['score']
        score  = int(s) if pd.notna(s) else '?'
        regime = _SCORE_LABELS.get(score, '?')
        b      = row['breadth_smooth']
        bstr   = f'{b:.1%}' if pd.notna(b) else '  n/a'
        bar    = '#' * int(b * 20) if pd.notna(b) else ''
        print(f'  {continent:<22}  {score:>5}  {regime:<8}  {bstr}  {bar}')

    g_score  = global_df.loc[latest_date, 'global_score']
    g_regime = global_df.loc[latest_date, 'global_regime']
    gs       = int(g_score) if pd.notna(g_score) else '?'
    print(f'\n  {"GLOBAL":<22}  {gs:>5}  {g_regime}')
    print()

    return continent_dfs, global_df
