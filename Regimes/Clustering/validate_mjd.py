"""
validate_mjd.py
================
Reproduces the accuracy-scoring methodology of paper Section 4.1.1 / Exhibit 4, but on
synthetic Merton jump-diffusion (MJD) paths calibrated to a real ticker (AAPL) instead of
the paper's hand-picked parameter sets.

Real AAPL returns have no known "true" regime label, so accuracy can't be scored directly
on them. Instead: estimate bull (theta0) and bear (theta1) MJD parameters from AAPL's own
history, simulate a fresh synthetic path from those parameters with known injected regime
switches, cluster it with UnivariateWKMeans (p=1, matching the paper's "Uni-d 1-WK-means"
row), and score cluster labels against the injected ground truth. Repeated over n_trials
independent draws, reporting mean +/- 95% CI for total / regime-on / regime-off accuracy,
matching Exhibit 4's columns exactly.

Calibration is a pragmatic moment-matching heuristic (rolling-window bull/bear split, jump
days = outliers beyond `jump_z` local std), not a formal MJD MLE fit -- good enough to get
realistic-scale parameters for a validation experiment, not a research-grade calibration.
"""
import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REGIMES = os.path.dirname(_HERE)
if _REGIMES not in sys.path:
    sys.path.insert(0, _REGIMES)

import numpy as np
import pandas as pd

from data import load_all
from uni_1d_wk import UnivariateWKMeans, H1, H2


# ── 1. Calibrate bull/bear MJD parameters from a real return series ────────────────

def estimate_regime_params(returns: np.ndarray, above_ma: np.ndarray,
                            trading_days: int = 252, jump_z: float = 3.0
                            ) -> tuple[dict, dict]:
    """
    Bull/bear day labels come from price vs. its 200-day SMA (the equities pipeline's own
    `Above_SMA200` signal) rather than a quantile split of the return series itself.
    Selecting on an extreme quantile of a rolling *mean return* and then re-measuring that
    same mean return on the selected days double-counts the same noise and inflates
    mu/sigma to unrealistic levels (a first version of this did exactly that and produced
    a 163%/-160% annualized drift). SMA200 is a heavily smoothed, independent trend signal,
    so conditioning on it doesn't have that self-referential bias.

    `mu`/`sigma` are annualized; `gamma`/`delta` describe the jump-size distribution
    directly in log-return units (per paper Appendix C.2, jumps aren't scaled by dt).
    """
    bull_mask = above_ma == True
    bear_mask = above_ma == False

    def _fit(mask):
        r = returns[mask]
        mu = r.mean() * trading_days
        sigma = max(r.std() * np.sqrt(trading_days), 1e-4)
        z = (r - r.mean()) / r.std()
        jumps = r[np.abs(z) > jump_z]
        lam = (len(jumps) / len(r)) * trading_days if len(r) else 0.0
        gamma = float(jumps.mean()) if len(jumps) else 0.0
        delta = float(jumps.std()) if len(jumps) > 1 else abs(gamma) * 0.5 + 1e-4
        return {'mu': mu, 'sigma': sigma, 'lam': max(lam, 0.0), 'gamma': gamma, 'delta': delta}

    return _fit(bull_mask), _fit(bear_mask)


# ── 2. Simulate MJD paths and inject known regime switches ─────────────────────────

def simulate_mjd(n_days: int, mu: float, sigma: float, lam: float, gamma: float, delta: float,
                  trading_days: int = 252, rng: np.random.Generator | None = None) -> np.ndarray:
    """Daily log-returns from a Merton jump-diffusion process, discretized per paper
    Appendix C.2: annualized drift `mu`, vol `sigma`, jump intensity `lam`, jump size ~
    Normal(gamma, delta^2)."""
    rng = rng if rng is not None else np.random.default_rng()
    dt = 1.0 / trading_days
    kappa = np.exp(gamma + 0.5 * delta ** 2) - 1.0
    z = rng.standard_normal(n_days)
    diffusion = (mu - lam * kappa - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    n_jumps = rng.poisson(lam * dt, size=n_days)
    has_jump = n_jumps > 0
    jump_component = np.zeros(n_days)
    jump_component[has_jump] = rng.normal(
        n_jumps[has_jump] * gamma, delta * np.sqrt(n_jumps[has_jump]), size=has_jump.sum()
    )
    return diffusion + jump_component


def generate_labeled_path(n_days: int, theta_bull: dict, theta_bear: dict, n_switches: int = 10,
                           switch_frac: float = 0.25, min_gap: int = 20,
                           rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a path mostly from theta_bull, with `switch_frac` of days replaced by
    theta_bear over `n_switches` randomly-placed, non-overlapping windows -- the paper's
    Section 4.1.1 injection scheme. Returns (returns, day_labels), day_labels in {0, 1}.
    """
    rng = rng if rng is not None else np.random.default_rng()
    returns = simulate_mjd(n_days, rng=rng, **theta_bull)
    labels = np.zeros(n_days, dtype=int)

    total_bear_days = int(n_days * switch_frac)
    base_len = total_bear_days // n_switches
    lengths = [base_len] * (n_switches - 1) + [total_bear_days - base_len * (n_switches - 1)]

    placed = []
    tries = 0
    while len(placed) < n_switches and tries < 2000:
        tries += 1
        length = lengths[len(placed)]
        start = int(rng.integers(0, n_days - length))
        end = start + length
        if all(end + min_gap <= s or start - min_gap >= e for s, e in placed):
            placed.append((start, end))

    for start, end in placed:
        returns[start:end] = simulate_mjd(end - start, rng=rng, **theta_bear)
        labels[start:end] = 1

    return returns, labels


# ── 3. Score clustering against known ground truth ─────────────────────────────────

def _window_ground_truth(day_labels: np.ndarray, n_windows: int, h1: int, step: int) -> np.ndarray:
    """Majority-vote ground-truth regime label for each lifted window."""
    truth = np.empty(n_windows, dtype=int)
    for i in range(n_windows):
        start = i * step
        truth[i] = round(day_labels[start:start + h1].mean())
    return truth


def score_accuracy(pred_labels: np.ndarray, truth: np.ndarray) -> dict:
    """
    Best-match predicted cluster ids to {bull=0, bear=1} ground truth -- cluster ids are
    arbitrary/unsupervised, so try both label permutations and keep whichever scores
    higher -- then report total / regime-on (bear windows) / regime-off (bull windows)
    accuracy, matching Exhibit 4's columns.
    """
    direct = (pred_labels == truth).mean()
    flipped = (pred_labels == (1 - truth)).mean()
    aligned = pred_labels if direct >= flipped else (1 - pred_labels)

    correct = aligned == truth
    on_mask, off_mask = truth == 1, truth == 0
    return {
        'total':      correct.mean(),
        'regime_on':  correct[on_mask].mean() if on_mask.any() else np.nan,
        'regime_off': correct[off_mask].mean() if off_mask.any() else np.nan,
    }


# ── 4. Experiment runner ────────────────────────────────────────────────────────────

def run_experiment(theta_bull: dict, theta_bear: dict, n_trials: int = 50, n_days: int = 2000,
                    h1: int = H1, h2: int = H2, seed0: int = 0) -> pd.DataFrame:
    """
    Fits both p=1 (uni-d 1-WK-means) and p=2 (uni-d 2-WK-means) on the *same* synthetic
    path each trial -- same seed, same injected regime switches -- so the two algorithms
    are compared on identical data, matching how the paper's Exhibit 4 puts both rows
    side by side rather than scoring them on independently-drawn paths.
    """
    step = h1 - h2
    rows = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed0 + trial)
        returns, day_labels = generate_labeled_path(n_days, theta_bull, theta_bear, rng=rng)

        trial_scores = {}
        for p in (1, 2):
            model = UnivariateWKMeans(k=2, p=p, random_state=seed0 + trial)
            measures = model.lift_stream(returns, h1=h1, h2=h2)
            model.fit(measures)

            truth = _window_ground_truth(day_labels, len(measures), h1, step)
            scores = score_accuracy(model.labels_, truth)
            scores['p'] = p
            scores['trial'] = trial
            rows.append(scores)
            trial_scores[p] = scores

        print(f'  trial {trial + 1}/{n_trials}: '
              f'p=1 total={trial_scores[1]["total"]:.3f}  p=2 total={trial_scores[2]["total"]:.3f}')

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    n = df['trial'].nunique()
    labels = {1: 'Uni-d 1-WK-means', 2: 'Uni-d 2-WK-means'}
    print(f'\n=== Accuracy over {n} trials (mean +/- 95% CI) ===')
    print(f'  {"Algorithm":<18} {"Total":>18} {"Regime-on":>18} {"Regime-off":>18}')
    for p in (1, 2):
        sub = df[df['p'] == p]
        cells = []
        for col in ['total', 'regime_on', 'regime_off']:
            mean = sub[col].mean()
            ci = 1.96 * sub[col].std() / np.sqrt(n)
            cells.append(f'{mean:.2%} +/- {ci:.2%}')
        print(f'  {labels[p]:<18} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}')


if __name__ == '__main__':
    N_TRIALS = 50
    N_DAYS = 2000

    data = load_all()
    log_ret  = data['equities']['Log_Return'].unstack('Ticker')
    above_ma = data['equities']['Above_SMA200'].unstack('Ticker')
    aapl = pd.DataFrame({'ret': log_ret['AAPL'], 'above_ma': above_ma['AAPL']}).dropna(subset=['ret'])
    aapl_returns = aapl['ret'].to_numpy()
    aapl_above_ma = aapl['above_ma'].to_numpy()

    theta_bull, theta_bear = estimate_regime_params(aapl_returns, aapl_above_ma)
    print('Calibrated from AAPL:')
    print(f'  bull (theta0): {theta_bull}')
    print(f'  bear (theta1): {theta_bear}')

    print(f'\nRunning {N_TRIALS} synthetic MJD trials ({N_DAYS} days each)...')
    results = run_experiment(theta_bull, theta_bear, n_trials=N_TRIALS, n_days=N_DAYS)
    summarize(results)
