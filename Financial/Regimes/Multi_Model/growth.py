import pandas as pd
import numpy as np
from Finance.Regimes.config import (
    GROWTH_THRESHOLD,
    GROWTH_SCORE_SMOOTH, GROWTH_DELTA_WINDOW,
    GROWTH_CONFIRM_WINDOW, GROWTH_CONFIRM_MIN,
    GROWTH_WEIGHT_YIELD_CURVE, GROWTH_WEIGHT_COPPER_GOLD,
    GROWTH_WEIGHT_EQUITY, GROWTH_WEIGHT_BOND,
    GROWTH_WEIGHT_USD, GROWTH_WEIGHT_EM, GROWTH_WEIGHT_PMI,
)

_REGIME_TO_NUM = {
    'Expansion': 3, 'Slowdown': 2, 'Recovery': 1, 'Contraction': 0, 'Unknown': -1,
}
_NUM_TO_REGIME = {v: k for k, v in _REGIME_TO_NUM.items()}


def _smooth(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(1, window // 2)
    return series.dropna().rolling(window, min_periods=mp).mean().reindex(series.index)


def _confirm(raw: pd.Series) -> pd.Series:
    num = raw.map(_REGIME_TO_NUM).astype(float)
    confirmed = (
        num
        .rolling(GROWTH_CONFIRM_WINDOW, min_periods=GROWTH_CONFIRM_MIN)
        .apply(lambda x: x[-1] if (x == x[-1]).sum() >= GROWTH_CONFIRM_MIN else np.nan, raw=True)
        .bfill()
        .ffill()
    )
    return confirmed.map(_NUM_TO_REGIME).fillna('Unknown')


def compute_growth_regime(
    equity_regime: tuple[dict, pd.DataFrame],
    bond_df:       pd.DataFrame,
    commodity_df:  pd.DataFrame,
    forex_df:      pd.DataFrame,
    macro_df:      pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    6-signal weighted growth regime using the Bridgewater All Weather quadrant framework.

    Inputs span leading → coincident → lagging to capture the full growth cycle:
      Leading  (6-12mo): yield curve, copper/gold ratio
      Coincident:        equity breadth, bond regime, USD regime, EM stress
      Optional:          PMI from macro_df

    Output:
      growth_regime    — 4-state: Expansion / Slowdown / Recovery / Contraction
      growth_simple    — 3-state: Growth / Transition / Contraction
      growth_score_smooth — normalised 0-1 composite score
      growth_score_delta  — 21d momentum of score (+ = accelerating, - = decelerating)
    """
    continent_dfs, global_df = equity_regime

    required = {
        'equity_regime (RUN_EQUITY)':   global_df,
        'bond_df (RUN_BOND)':           bond_df,
        'commodity_df (RUN_COMMODITY)': commodity_df,
        'forex_df (RUN_FOREX)':         forex_df,
    }
    missing = [name for name, df in required.items() if df is None or df.empty]
    if missing:
        raise ValueError(
            'compute_growth_regime requires all single-asset regimes to be run first. '
            f'Missing: {missing}. Enable the corresponding flags in main.py.'
        )

    idx       = global_df.index
    growth_df = pd.DataFrame(index=idx)

    # ── Signal 1: Yield curve — leading (6-12 months) ─────────────────────────
    curve = bond_df['curve_2s10s_smooth'].reindex(idx)
    growth_df['sig_yield_curve'] = (curve > 0).astype(float)
    growth_df['curve_2s10s']     = curve

    # ── Signal 2: Copper/Gold ratio deviation — leading (3-6 months) ──────────
    cu_au_sm  = commodity_df['copper_gold_ratio_smooth'].reindex(idx)
    cu_au_ma  = commodity_df['copper_gold_ratio_ma200'].reindex(idx)
    cu_au_dev = (cu_au_sm / cu_au_ma - 1).where(cu_au_ma.notna())
    growth_df['sig_copper_gold'] = (cu_au_dev > 0).astype(float)
    growth_df['copper_gold_dev'] = cu_au_dev

    # ── Signal 3: Equity breadth — coincident ─────────────────────────────────
    breadth = pd.concat(
        {c: cdf['breadth_smooth'] for c, cdf in continent_dfs.items()}, axis=1
    ).mean(axis=1).reindex(idx)
    growth_df['sig_equity_breadth'] = (breadth > 0.55).astype(float)
    growth_df['equity_breadth']     = breadth

    # ── Signal 4: Bond regime not bearish — coincident ────────────────────────
    growth_df['sig_bond_supportive'] = (
        bond_df['bond_regime'].reindex(idx) != 'Bond Bear'
    ).astype(float)

    # ── Signal 5: USD not dominant — coincident-lagging ───────────────────────
    growth_df['sig_usd_benign'] = (
        forex_df['usd_regime'].reindex(idx) != 'USD Bull'
    ).astype(float)

    # ── Signal 6: EM not stressed — coincident ────────────────────────────────
    em_stress = forex_df['em_stress_smooth'].reindex(idx)
    growth_df['sig_em_ok']  = (em_stress < 0.5).astype(float)
    growth_df['em_stress']  = em_stress

    # ── Optional Signal 7: PMI — coincident ───────────────────────────────────
    has_pmi = False
    if macro_df is not None and 'pmi_composite' in macro_df.columns:
        pmi = macro_df['pmi_composite'].reindex(idx).ffill()
        growth_df['sig_pmi'] = (pmi > 50).astype(float)
        growth_df['pmi']     = pmi
        has_pmi = True

    # ── Weighted score (normalised 0-1) ───────────────────────────────────────
    weights = {
        'sig_yield_curve':     GROWTH_WEIGHT_YIELD_CURVE,
        'sig_copper_gold':     GROWTH_WEIGHT_COPPER_GOLD,
        'sig_equity_breadth':  GROWTH_WEIGHT_EQUITY,
        'sig_bond_supportive': GROWTH_WEIGHT_BOND,
        'sig_usd_benign':      GROWTH_WEIGHT_USD,
        'sig_em_ok':           GROWTH_WEIGHT_EM,
    }
    if has_pmi:
        weights['sig_pmi'] = GROWTH_WEIGHT_PMI

    max_score = sum(weights.values())
    raw_score = sum(growth_df[col].fillna(0) * w for col, w in weights.items())

    growth_df['growth_score_raw']    = raw_score / max_score
    growth_df['growth_score_smooth'] = _smooth(
        growth_df['growth_score_raw'], window=GROWTH_SCORE_SMOOTH
    )
    # 21-day delta: positive = score improving (growth accelerating)
    growth_df['growth_score_delta'] = (
        growth_df['growth_score_smooth']
        .dropna()
        .diff(GROWTH_DELTA_WINDOW)
        .reindex(idx)
    )

    # ── 4-quadrant regime (Bridgewater All Weather framework) ─────────────────
    above = growth_df['growth_score_smooth'] >= GROWTH_THRESHOLD
    accel = growth_df['growth_score_delta']  >= 0

    raw_regime = pd.Series(
        np.select(
            [above &  accel,   # Expansion:   above trend + accelerating
             above & ~accel,   # Slowdown:    above trend + decelerating
             ~above & accel,   # Recovery:    below trend + accelerating
             ~above & ~accel], # Contraction: below trend + decelerating
            ['Expansion', 'Slowdown', 'Recovery', 'Contraction'],
            default='Unknown',
        ),
        index=idx,
    )
    growth_df['growth_regime_raw'] = raw_regime
    growth_df['growth_regime']     = _confirm(raw_regime)

    # ── Simple 3-state for downstream consumption ─────────────────────────────
    growth_df['growth_simple'] = np.select(
        [growth_df['growth_regime'].isin(['Expansion', 'Recovery']),
         growth_df['growth_regime'] == 'Contraction'],
        ['Growth', 'Contraction'],
        default='Transition',
    )

    # ── snapshot print ────────────────────────────────────────────────────────
    valid    = growth_df.dropna(subset=['growth_score_smooth'])
    latest   = valid.iloc[-1]
    date_str = valid.index[-1].strftime('%Y-%m-%d')

    print(f'\n=== Growth Regime Snapshot ({date_str}) ===\n')
    print(f'  {"Growth Regime:":<30} {latest["growth_regime"]}')
    print(f'  {"Growth Simple:":<30} {latest["growth_simple"]}')
    print(f'  {"Score (smooth):":<30} {latest["growth_score_smooth"]:.3f}  (threshold: {GROWTH_THRESHOLD})')
    print(f'  {"Direction (21d delta):":<30} {"↑ Accelerating" if latest["growth_score_delta"] >= 0 else "↓ Decelerating"}')

    print('\n  --- Signal Breakdown ---')
    signal_labels = [
        ('sig_yield_curve',     f'Yield Curve Positive (2s10s = {latest["curve_2s10s"]:+.2f}%)'),
        ('sig_copper_gold',     f'Cu/Au Above MA200 (dev = {latest["copper_gold_dev"]:+.1%})'),
        ('sig_equity_breadth',  f'Equity Breadth > 55% ({latest["equity_breadth"]:.1%})'),
        ('sig_bond_supportive', 'Bond Regime Not Bearish'),
        ('sig_usd_benign',      'USD Not Dominant'),
        ('sig_em_ok',           f'EM Not Stressed (z = {latest["em_stress"]:+.2f})'),
    ]
    if has_pmi and 'pmi' in latest.index:
        signal_labels.append(('sig_pmi', f'PMI > 50 ({latest["pmi"]:.1f})'))
    for col, label in signal_labels:
        val = latest.get(col, np.nan)
        print(f'    {"YES" if val == 1.0 else "NO ":<4} {label}')

    print('\n  --- Growth Regime Distribution (full history) ---')
    for regime in ['Expansion', 'Slowdown', 'Recovery', 'Contraction']:
        pct = (growth_df['growth_regime'] == regime).mean()
        print(f'  {regime:<16} {pct:>5.1%}  {"#"*int(pct*40)}')
    print()

    return growth_df
