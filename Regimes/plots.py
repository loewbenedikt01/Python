import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import REGIME_COLORS, CURVE_COLORS

COMMODITY_REGIME_COLORS = {
    'Commodity Bull':    '#2ecc71',
    'Commodity Neutral': '#ffc04d',
    'Commodity Bear':    '#e74c3c',
    'Unknown':           '#555577',
}

CRYPTO_REGIME_COLORS = {
    'Crypto Bull':    '#2ecc71',
    'Crypto Neutral': '#ffc04d',
    'Crypto Bear':    '#e74c3c',
    'Unknown':        '#555577',
}

FOREX_REGIME_COLORS = {
    'Risk On': '#2ecc71',
    'Neutral': '#ffc04d',
    'Stress':  '#e74c3c',
    'Unknown': '#555577',
}

_STRENGTH_COLORS = {
    'strength_usd': '#f1c40f',
    'strength_eur': '#3498db',
    'strength_jpy': '#e74c3c',
    'strength_aud': '#2ecc71',
    'strength_gbp': '#9b59b6',
    'strength_chf': '#1abc9c',
}

_BREADTH_META = {'breadth_weighted', 'breadth_smooth', 'breadth_trend',
                 'breadth_delta', 'breadth_rising', 'breadth_direction'}

_CONT_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']


def _dark_axes(*axes):
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')


def _shade(ax, data, col, color_map, alpha=0.25):
    prev, start = None, data.index[0]
    for date, val in data[col].items():
        if val != prev:
            if prev is not None:
                ax.axvspan(start, date, color=color_map.get(prev, '#cccccc'),
                           alpha=alpha, linewidth=0)
            start, prev = date, val
    if prev is not None:
        ax.axvspan(start, data.index[-1], color=color_map.get(prev, '#cccccc'),
                   alpha=alpha, linewidth=0)


def plot_equity_regime(regime_df) -> None:
    region_cols = sorted(
        c for c in regime_df.columns
        if c.startswith('breadth_') and c not in _BREADTH_META
    )
    data = regime_df.dropna(subset=['breadth_smooth']).copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    _dark_axes(ax1, ax2)

    _shade(ax1, data, 'equity_regime', REGIME_COLORS)

    ax1.plot(data.index, data['breadth_weighted'] * 100,
             color='#ffffff', linewidth=0.8, alpha=0.5, label='Daily breadth')
    ax1.plot(data.index, data['breadth_smooth'] * 100,
             color='#f1c40f', linewidth=1.8, label='Smooth (10d)')
    ax1.plot(data.index, data['breadth_trend'] * 100,
             color='#e74c3c', linewidth=1.5, linestyle='--', label='Trend (63d)')
    ax1.axhline(57, color='#7f8c8d', linewidth=1, linestyle=':')
    ax1.axhline(53, color='#7f8c8d', linewidth=1, linestyle=':')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('% Above 200-MA', color='#cccccc')
    ax1.set_title('Global Equity Regime  —  Breadth Above 200-Day MA',
                  color='#ecf0f1', fontsize=13, pad=10)

    line_handles, _ = ax1.get_legend_handles_labels()
    regime_patches = [
        mpatches.Patch(facecolor=REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ['Bull', 'Neutral', 'Bear']
    ]
    ax1.legend(handles=line_handles + regime_patches, loc='upper left',
               framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e')

    for col, color in zip(region_cols, _CONT_COLORS):
        label  = col.replace('breadth_', '').replace('_', ' ').title()
        td     = data[col].dropna()
        series = td.rolling(10, min_periods=5).mean().reindex(data.index) * 100
        ax2.plot(data.index, series, linewidth=1.2, color=color, label=label)

    ax2.axhline(55, color='#7f8c8d', linewidth=1, linestyle=':')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('% Above 200-MA', color='#cccccc')
    ax2.set_xlabel('Date', color='#cccccc')
    ax2.legend(loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', ncol=3, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_bond_regime(bond_df) -> None:
    data = bond_df.dropna(subset=['curve_2s10s_smooth', 'rate_ma200']).copy()

    fig, (ax1, ax_strip, ax2) = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={'height_ratios': [3, 0.25, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    _dark_axes(ax1, ax_strip, ax2)

    _shade(ax1, data, 'bond_regime', REGIME_COLORS, alpha=0.28)

    ax1.plot(data.index, data['curve_level'], color='#ffffff', linewidth=1.0,
             alpha=0.6, label='10yr Yield')
    ax1.plot(data.index, data['rate_ma50'],   color='#f1c40f', linewidth=1.4, label='MA50')
    ax1.plot(data.index, data['rate_ma200'],  color='#e74c3c', linewidth=1.4,
             linestyle='--', label='MA200')
    ax1.axhline(0, color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax1.set_ylabel('10-Year Yield (%)', color='#cccccc')
    ax1.set_title('Bond Regime  —  Yield Curve & Rate Trend',
                  color='#ecf0f1', fontsize=13, pad=10)

    line_handles, _ = ax1.get_legend_handles_labels()
    regime_patches  = [
        mpatches.Patch(facecolor=REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ['Bond Bull', 'Bond Neutral', 'Bond Bear']
    ]
    ax1.legend(handles=line_handles + regime_patches, loc='upper right',
               framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e')

    _shade(ax_strip, data, 'curve_regime', CURVE_COLORS, alpha=0.85)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('Curve', color='#aaaaaa', fontsize=7, rotation=0, labelpad=28)
    curve_patches = [
        mpatches.Patch(facecolor=CURVE_COLORS[s], alpha=0.85, label=s)
        for s in ['Steep', 'Normal', 'Flat', 'Inverted']
    ]
    ax_strip.legend(handles=curve_patches, loc='upper right', framealpha=0.3,
                    labelcolor='#ecf0f1', facecolor='#1a1a2e', fontsize=7,
                    ncol=4, borderpad=0.3, handlelength=1)

    ax2.plot(data.index, data['curve_2s10s'],
             color='#ffffff', linewidth=0.8, alpha=0.4, label='2s10s (raw)')
    ax2.plot(data.index, data['curve_2s10s_smooth'],
             color='#3498db', linewidth=1.6, label='2s10s (5d smooth)')
    ax2.plot(data.index, data['curve_5s30s'],
             color='#9b59b6', linewidth=1.2, linestyle='--', label='5s30s')
    ax2.axhline(0,    color='#e74c3c', linewidth=1.0, alpha=0.8)
    ax2.axhline(1.0,  color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax2.axhline(-0.5, color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax2.fill_between(data.index, data['curve_2s10s_smooth'], 0,
                     where=data['curve_2s10s_smooth'] < 0,
                     color='#e74c3c', alpha=0.15, label='Inverted zone')
    ax2.set_ylabel('Spread (%)', color='#cccccc')
    ax2.set_xlabel('Date',       color='#cccccc')
    ax2.legend(loc='upper right', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    plt.show()


def plot_crypto_regime(crypto_df) -> None:
    data = crypto_df.dropna(subset=['btc_ma200', 'eth_btc_ma']).copy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={'height_ratios': [3, 1.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    _dark_axes(ax1, ax2, ax3)

    # ── Top: BTC price + MA200 with regime shading ───────────────────────────
    _shade(ax1, data, 'crypto_regime', CRYPTO_REGIME_COLORS, alpha=0.28)

    ax1.plot(data.index, data['btc_smooth'], color='#ffffff', linewidth=1.0,
             alpha=0.7, label='BTC (5d smooth)')
    ax1.plot(data.index, data['btc_ma200'],  color='#e74c3c', linewidth=1.4,
             linestyle='--', label='BTC MA200')
    ax1.set_yscale('log')
    ax1.set_ylabel('BTC Price (log)', color='#cccccc')
    ax1.set_title('Crypto Regime  —  BTC Trend, Momentum & ETH/BTC',
                  color='#ecf0f1', fontsize=13, pad=10)

    ax1_score = ax1.twinx()
    ax1_score.set_facecolor('none')
    ax1_score.plot(data.index, data['crypto_score_smooth'], color='#f1c40f',
                   linewidth=1.2, alpha=0.6, linestyle=':', label='Score (smooth)')
    ax1_score.axhline(3, color='#2ecc71', linewidth=0.8, linestyle=':', alpha=0.5)
    ax1_score.axhline(1, color='#e74c3c', linewidth=0.8, linestyle=':', alpha=0.5)
    ax1_score.set_ylim(-0.5, 5)
    ax1_score.set_ylabel('Bull Score', color='#f1c40f')
    ax1_score.tick_params(colors='#f1c40f')

    line_handles, _ = ax1.get_legend_handles_labels()
    regime_patches  = [
        mpatches.Patch(facecolor=CRYPTO_REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ['Crypto Bull', 'Crypto Neutral', 'Crypto Bear']
    ]
    ax1.legend(handles=line_handles + regime_patches, loc='upper left',
               framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e')

    # ── Middle: BTC 63d momentum ──────────────────────────────────────────────
    ax2.plot(data.index, data['btc_return_smooth'] * 100,
             color='#f1c40f', linewidth=1.4, label='BTC 63d Return (smooth)')
    ax2.axhline(0, color='#e74c3c', linewidth=1.0, alpha=0.8)
    ax2.fill_between(data.index, data['btc_return_smooth'] * 100, 0,
                     where=data['btc_return_smooth'] > 0,
                     color='#2ecc71', alpha=0.15)
    ax2.fill_between(data.index, data['btc_return_smooth'] * 100, 0,
                     where=data['btc_return_smooth'] < 0,
                     color='#e74c3c', alpha=0.15)
    ax2.set_ylabel('63d Return (%)', color='#cccccc')
    ax2.legend(loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    # ── Bottom: ETH/BTC ratio vs its MA ──────────────────────────────────────
    ax3.plot(data.index, data['eth_btc_ratio'], color='#9b59b6', linewidth=1.0,
             alpha=0.6, label='ETH/BTC ratio')
    ax3.plot(data.index, data['eth_btc_ma'],    color='#3498db', linewidth=1.4,
             label=f'ETH/BTC MA50')
    ax3.fill_between(data.index, data['eth_btc_ratio'], data['eth_btc_ma'],
                     where=data['eth_btc_ratio'] > data['eth_btc_ma'],
                     color='#2ecc71', alpha=0.15, label='ETH Leading')
    ax3.fill_between(data.index, data['eth_btc_ratio'], data['eth_btc_ma'],
                     where=data['eth_btc_ratio'] < data['eth_btc_ma'],
                     color='#e74c3c', alpha=0.15, label='BTC Dominant')
    ax3.set_ylabel('ETH/BTC', color='#cccccc')
    ax3.set_xlabel('Date',    color='#cccccc')
    ax3.legend(loc='upper right', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    plt.tight_layout()
    plt.show()


def _smooth_plot(series: pd.Series, window: int) -> pd.Series:
    """Light smoothing for display only — dropna → rolling → reindex."""
    return series.dropna().rolling(window, min_periods=max(1, window // 2)).mean().reindex(series.index)


def plot_forex_regime(forex_df) -> None:
    data = forex_df.dropna(subset=['dxy_ma200']).copy()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(16, 13),
        gridspec_kw={'height_ratios': [3, 1.5, 2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    _dark_axes(ax1, ax2, ax3, ax4)

    # ── Top: DXY + MA50/MA200 with master regime shading ─────────────────────
    _shade(ax1, data, 'forex_regime', FOREX_REGIME_COLORS, alpha=0.28)

    if 'dxy' in data.columns:
        ax1.plot(data.index, data['dxy'], color='#ffffff', linewidth=0.9,
                 alpha=0.6, label='DXY')
    ax1.plot(data.index, data['dxy_ma50'],  color='#f1c40f', linewidth=1.4, label='MA50')
    ax1.plot(data.index, data['dxy_ma200'], color='#e74c3c', linewidth=1.4,
             linestyle='--', label='MA200')
    ax1.set_ylabel('DXY (USD Index)', color='#cccccc')
    ax1.set_title('Forex Regime  —  USD × Carry Framework',
                  color='#ecf0f1', fontsize=13, pad=10)

    line_handles, _ = ax1.get_legend_handles_labels()
    regime_patches  = [
        mpatches.Patch(facecolor=FOREX_REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ['Risk On', 'Neutral', 'Stress']
    ]
    ax1.legend(handles=line_handles + regime_patches, loc='upper left',
               framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e', fontsize=8)

    # ── Second: AUD/JPY carry proxy with carry regime shading ─────────────────
    CARRY_COLORS = {'Carry Bull': '#2ecc71', 'Carry Neutral': '#ffc04d',
                    'Carry Bear': '#e74c3c', 'Unknown': '#555577'}
    if 'carry_regime' in data.columns:
        _shade(ax2, data, 'carry_regime', CARRY_COLORS, alpha=0.35)

    if 'audjpy' in data.columns:
        ax2.plot(data.index, data['audjpy'], color='#ffffff', linewidth=0.9,
                 alpha=0.6, label='AUD/JPY')
    if 'audjpy_ma50' in data.columns:
        ax2.plot(data.index, data['audjpy_ma50'], color='#3498db', linewidth=1.2,
                 linestyle=':', alpha=0.7, label='MA50')
    if 'audjpy_ma200' in data.columns:
        ax2.plot(data.index, data['audjpy_ma200'], color='#f1c40f', linewidth=1.5,
                 label='MA200')
    ax2.set_ylabel('AUD/JPY (Carry)', color='#cccccc')

    carry_patches = [mpatches.Patch(facecolor=CARRY_COLORS[r], alpha=0.6, label=r)
                     for r in ['Carry Bull', 'Carry Neutral', 'Carry Bear']]
    ax2.legend(handles=carry_patches + ax2.get_legend_handles_labels()[0],
               loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    # ── Third: Currency strength index ────────────────────────────────────────
    for col, color in _STRENGTH_COLORS.items():
        if col not in data.columns:
            continue
        label  = col.replace('strength_', '').upper()
        # Winsorise at 1st/99th percentile to prevent 2008/2020 spikes compressing the panel
        raw    = data[col]
        clipped = raw.clip(raw.quantile(0.01), raw.quantile(0.99))
        series  = _smooth_plot(clipped, window=15)
        ax3.plot(data.index, series, color=color, linewidth=1.3, label=label)

    ax3.axhline(0, color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax3.set_ylabel('Currency Strength', color='#cccccc')
    ax3.legend(loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', ncol=6, fontsize=8)

    # ── Bottom: EM stress z-score ─────────────────────────────────────────────
    if 'em_stress_smooth' in data.columns:
        em = data['em_stress_smooth']
        ax4.plot(data.index, em, color='#e74c3c', linewidth=1.2, label='EM Stress (z-score)')
        ax4.fill_between(data.index, em, 0,
                         where=em >  1.0, color='#e74c3c', alpha=0.25, label='Stress zone')
        ax4.fill_between(data.index, em, 0,
                         where=em < -0.5, color='#2ecc71', alpha=0.25, label='Relief zone')
        ax4.axhline( 1.0, color='#e74c3c', linewidth=0.8, linestyle=':', alpha=0.7)
        ax4.axhline(-0.5, color='#2ecc71', linewidth=0.8, linestyle=':', alpha=0.7)
        ax4.axhline( 0,   color='#7f8c8d', linewidth=0.8, linestyle=':')
    ax4.set_ylabel('EM Stress (σ)', color='#cccccc')
    ax4.set_xlabel('Date', color='#cccccc')
    ax4.legend(loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    plt.show()


def plot_commodity_regime(commodity_df) -> None:
    data = commodity_df.dropna(subset=['bull_score_smooth']).copy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 11),
        gridspec_kw={'height_ratios': [3, 1.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor('#1a1a2e')
    _dark_axes(ax1, ax2, ax3)

    # ── Top: Oil price + MA200 with commodity regime shading ─────────────────
    _shade(ax1, data, 'commodity_regime', COMMODITY_REGIME_COLORS, alpha=0.28)

    if 'oil' in data.columns:
        ax1.plot(data.index, data['oil'], color='#ffffff', linewidth=0.9,
                 alpha=0.6, label='WTI Crude')
    if 'oil_smooth' in data.columns:
        ax1.plot(data.index, data['oil_smooth'], color='#f1c40f', linewidth=1.4,
                 label='WTI (20d smooth)')
    if 'oil_ma200' in data.columns:
        ax1.plot(data.index, data['oil_ma200'], color='#e74c3c', linewidth=1.4,
                 linestyle='--', label='WTI MA200')
    ax1.set_ylabel('WTI Crude (USD)', color='#cccccc')
    ax1.set_title('Commodity Regime  —  Energy · Industrial · Precious · Agriculture',
                  color='#ecf0f1', fontsize=13, pad=10)

    line_handles, _ = ax1.get_legend_handles_labels()
    regime_patches  = [
        mpatches.Patch(facecolor=COMMODITY_REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ['Commodity Bull', 'Commodity Neutral', 'Commodity Bear']
    ]
    ax1.legend(handles=line_handles + regime_patches, loc='upper right',
               framealpha=0.3, labelcolor='#ecf0f1', facecolor='#1a1a2e', fontsize=8)

    # ── Middle: Copper / Gold ratio + MA200 ───────────────────────────────────
    if 'copper_gold_ratio_smooth' in data.columns:
        ax2.plot(data.index, data['copper_gold_ratio_smooth'], color='#e67e22',
                 linewidth=1.4, label='Cu/Au Ratio (smooth)')
    if 'copper_gold_ratio_ma200' in data.columns:
        ax2.plot(data.index, data['copper_gold_ratio_ma200'], color='#e74c3c',
                 linewidth=1.4, linestyle='--', label='Cu/Au MA200')
    if 'copper_gold_ratio_smooth' in data.columns and 'copper_gold_ratio_ma200' in data.columns:
        above = data['copper_gold_ratio_smooth'] > data['copper_gold_ratio_ma200']
        ax2.fill_between(data.index, data['copper_gold_ratio_smooth'],
                         data['copper_gold_ratio_ma200'],
                         where=above,  color='#2ecc71', alpha=0.35, label='Growth > Fear')
        ax2.fill_between(data.index, data['copper_gold_ratio_smooth'],
                         data['copper_gold_ratio_ma200'],
                         where=~above, color='#e74c3c', alpha=0.35, label='Fear > Growth')
    ax2.set_ylabel('Cu / Au Ratio', color='#cccccc')
    ax2.legend(loc='upper right', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8)

    # ── Bottom: Sub-group scores smoothed to continuous 0-1 lines ────────────
    # Raw binary signals (0/1) look like square waves — smooth them into
    # continuous score lines so the panel reads as a macro dashboard
    if 'energy_score' in data.columns:
        energy_plot = data['energy_score'].rolling(40, min_periods=20).mean() / 3
        ax3.plot(data.index, energy_plot, color='#e74c3c', linewidth=1.3,
                 alpha=0.85, label='Energy (0-1)')

    if 'gold_score' in data.columns:
        gold_plot = data['gold_score'].rolling(40, min_periods=20).mean() / 3
        ax3.plot(data.index, gold_plot, color='#f1c40f', linewidth=1.3,
                 alpha=0.85, label='Gold (0-1)')

    if 'ag_signal' in data.columns:
        ag_plot = data['ag_signal'].rolling(40, min_periods=20).mean()
        ax3.plot(data.index, ag_plot, color='#2ecc71', linewidth=1.3,
                 alpha=0.85, label='Agriculture (0-1)')

    # Industrial: z-score of 20d-smoothed Cu/Au ratio, clipped ±2, scaled to 0-1
    if 'copper_gold_ratio' in data.columns:
        cu_au_sm   = data['copper_gold_ratio'].rolling(40, min_periods=20).mean()
        cu_au_mean = cu_au_sm.rolling(252, min_periods=126).mean()
        cu_au_std  = cu_au_sm.rolling(252, min_periods=126).std()
        indus_plot = ((cu_au_sm - cu_au_mean) / cu_au_std.replace(0, np.nan)
                      ).clip(-2, 2) / 4 + 0.5
        ax3.plot(data.index, indus_plot, color='#e67e22', linewidth=1.3,
                 alpha=0.85, label='Industrial Cu/Au (z-score)')

    # Overlay smoothed bull score on right axis
    if 'bull_score_smooth' in data.columns:
        ax3_r = ax3.twinx()
        ax3_r.set_facecolor('none')
        ax3_r.plot(data.index, data['bull_score_smooth'], color='#ffffff',
                   linewidth=1.6, alpha=0.45, linestyle=':', label='Bull Score')
        ax3_r.axhline(3.0, color='#2ecc71', linewidth=0.7, linestyle=':', alpha=0.5)
        ax3_r.axhline(1.0, color='#e74c3c', linewidth=0.7, linestyle=':', alpha=0.5)
        ax3_r.set_ylim(-0.2, 4.5)
        ax3_r.set_ylabel('Bull Score (0-4)', color='#aaaaaa', fontsize=8)
        ax3_r.tick_params(colors='#aaaaaa')

    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel('Sub-group Score (0-1)', color='#cccccc')
    ax3.set_xlabel('Date', color='#cccccc')
    ax3.legend(loc='upper left', framealpha=0.3, labelcolor='#ecf0f1',
               facecolor='#1a1a2e', fontsize=8, ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    plt.show()
