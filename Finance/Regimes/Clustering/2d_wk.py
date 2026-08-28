"""
2d_wk.py
========
The 2-d WK-means model (Section 3.1): multivariate (2-D) Wasserstein k-means on
copula-transformed (uniform-margin) return pairs, isolating the correlation/dependence
regime between two assets once each asset's own marginal mean-variance effect has been
removed via the univariate uni-d 1-WK-means step (uni_1d_wk.py). This is "Step 2" of the
paper's two-step real-data pipeline (Section 5.2) -- Step 1 is uni_1d_wk.py's
transform_to_uniform.

Distance: true 2-Wasserstein distance between two h1-point clouds in R^2, computed via
the Hungarian algorithm (scipy.optimize.linear_sum_assignment) on the squared-Euclidean
cost matrix (scipy.spatial.distance.cdist) -- Section 3.1.2's exact recipe. This is
expensive (O(h1^3) per pair, O(M^2) pairs), which is exactly why the paper restricts the
barycenter search to real data points instead of a free-support barycenter (infeasible
in 2-D) -- the barycenter of a cluster is just the member whose distance-matrix row sum
is smallest, i.e. plain k-medoids on a precomputed distance matrix. `make_segments` and
`kmedoids` below are metric-agnostic (they don't care whether the distance matrix came
from MMD or true Wasserstein), so this file is self-contained rather than importing them
from elsewhere -- only the distance metric (true W2) and the pipeline wiring are new.

h1=140, h2=133 matches the paper's own choice for the AAPL-AMZN pair (Section 5.2.1) --
correlation regimes are believed to shift more slowly than mean-variance regimes, hence
the much wider window than the univariate step's.
"""
import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REGIMES = os.path.dirname(_HERE)
if _REGIMES not in sys.path:
    sys.path.insert(0, _REGIMES)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from Finance.Regimes.data import load_all
from Finance.Regimes.Clustering.uni_1d_wk import UnivariateWKMeans, select_k as select_k_uni, H1 as UNI_H1, H2 as UNI_H2

H1 = 140   # paper's AAPL-AMZN correlation-step window (Section 5.2.1)
H2 = 133


# ── 0. Generic segment/k-medoids helpers (distance-metric-agnostic) ────────────────

def make_segments(values: np.ndarray, h1: int, h2: int) -> np.ndarray:
    """Slide a window of length h1 (overlap h2) across a (T, d) return series,
    returning (M, h1, d) overlapping segments."""
    step = h1 - h2
    if step < 1:
        raise ValueError(f'h2 ({h2}) must be smaller than h1 ({h1})')
    if values.ndim == 1:
        values = values[:, None]
    windows = sliding_window_view(values, window_shape=h1, axis=0)[::step]  # (M, d, h1)
    return np.ascontiguousarray(np.moveaxis(windows, -1, 1))                # (M, h1, d)


def regime_timeline(index: pd.DatetimeIndex, labels: np.ndarray, h1: int, h2: int) -> pd.Series:
    """Expand per-segment cluster labels into a daily series, ffilled between segment starts."""
    step = h1 - h2
    starts = index[[i * step for i in range(len(labels))]]
    return pd.Series(labels, index=starts).reindex(index).ffill()


def _kmedoids_pp_init(dist: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ style seeding: each new medoid is sampled weighted by its squared
    distance to the nearest medoid already chosen."""
    M = dist.shape[0]
    medoids = [int(rng.integers(M))]
    for _ in range(1, k):
        d_nearest = dist[:, medoids].min(axis=1)
        weights = d_nearest ** 2
        total = weights.sum()
        probs = weights / total if total > 0 else np.full(M, 1.0 / M)
        medoids.append(int(rng.choice(M, p=probs)))
    return np.array(medoids)


def kmedoids(dist: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100,
             random_state: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Partition-around-medoids on a precomputed distance matrix. The barycenter update is
    a medoid search restricted to points already in the dataset: the row-sum minimiser
    within each cluster (paper Section 3.1.2's exact recipe for the barycenter, needed
    since a free-support 2-D Wasserstein barycenter is intractable).
    """
    M = dist.shape[0]
    rng = np.random.default_rng(random_state)
    best_labels, best_medoids, best_inertia = None, None, np.inf

    for _ in range(n_init):
        medoids = _kmedoids_pp_init(dist, k, rng)
        for _ in range(max_iter):
            labels = np.argmin(dist[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                sub = dist[np.ix_(members, members)]
                new_medoids[c] = members[np.argmin(sub.sum(axis=1))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
        labels = np.argmin(dist[:, medoids], axis=1)
        inertia = dist[np.arange(M), medoids[labels]].sum()
        if inertia < best_inertia:
            best_labels, best_medoids, best_inertia = labels, medoids, inertia

    return best_labels, best_medoids, best_inertia


# ── 1. True 2-Wasserstein distance between two 2-D point clouds ────────────────────

def wasserstein2_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    2-Wasserstein distance between two h1-point clouds in R^2 (paper eq. in 3.1.2):
    solve the optimal assignment between X and Y on squared-Euclidean cost via the
    Hungarian algorithm, then take the sqrt of the total matched cost.
    """
    cost = cdist(X, Y, metric='sqeuclidean')
    row, col = linear_sum_assignment(cost)
    return float(np.sqrt(cost[row, col].sum()))


def wasserstein2_distance_matrix(segments: np.ndarray) -> np.ndarray:
    """
    M x M matrix of pairwise 2-Wasserstein distances between all M 2-D segments.
    O(M^2) Hungarian-algorithm solves -- keep M modest (recent-history slice, or a
    correlation-screened shortlist of pairs) rather than running this on full history.
    """
    M = segments.shape[0]
    dist = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            d = wasserstein2_distance(segments[i], segments[j])
            dist[i, j] = dist[j, i] = d
    return dist


# ── 2. Univariate copula transform (Step 1) then 2-d WK-means (Step 2) ──────────────

def uniform_margin(returns: pd.Series, h1: int = UNI_H1, h2: int = UNI_H2, p: int = 1,
                    k_min: int = 2, k_max: int = 6, alpha: float = 0.05,
                    random_state: int = 42) -> pd.Series:
    """
    Step 1: univariate WK-means with auto-selected k, then transform to uniform margins.
    `p=1` is uni-d 1-WK-means (uni_1d_wk.py's model), `p=2` is uni-d 2-WK-means
    (uni_2d_wk.py's model) -- same UnivariateWKMeans class either way, just a different
    distance/barycenter choice, so no separate import is needed to switch.
    """
    raw = returns.to_numpy()
    measures = UnivariateWKMeans.lift_stream(raw, h1=h1, h2=h2)
    k, model = select_k_uni(measures, k_min=k_min, k_max=k_max, alpha=alpha,
                             p=p, random_state=random_state, verbose=False)
    uniform = model.transform_to_uniform(raw, h1=h1, h2=h2)
    return pd.Series(uniform, index=returns.index), k


def cluster_pair(uni_a: pd.Series, uni_b: pd.Series, h1: int = H1, h2: int = H2,
                  k: int = 2, n_init: int = 10, max_iter: int = 100,
                  random_state: int = 42) -> dict | None:
    """
    Step 2: 2-d WK-means on a pair's copula-transformed (uniform-margin) returns. With
    marginals flattened to uniform, what's left to cluster is the correlation/dependence
    structure between the two tickers -- the correlation regime.
    """
    joint = pd.concat([uni_a, uni_b], axis=1).dropna()
    joint.columns = ['a', 'b']
    if len(joint) < h1 + 1:
        return None

    segments = make_segments(joint.to_numpy(), h1, h2)
    if segments.shape[0] < 2:
        return None

    dist = wasserstein2_distance_matrix(segments)
    labels, medoids, inertia = kmedoids(dist, k, n_init=n_init, max_iter=max_iter,
                                         random_state=random_state)
    regime = regime_timeline(joint.index, labels, h1, h2)

    cluster_stats = pd.DataFrame({
        'n_segments': [int((labels == c).sum()) for c in range(k)],
        'corr': [
            float(np.corrcoef(segments[labels == c].reshape(-1, 2).T)[0, 1])
            if (labels == c).sum() > 0 else np.nan
            for c in range(k)
        ],
    })

    return {
        'k': k, 'labels': labels, 'medoids': medoids, 'segments': segments,
        'regime': regime, 'inertia': inertia, 'cluster_stats': cluster_stats,
    }


if __name__ == '__main__':
    TICKER_A, TICKER_B = 'AAPL', 'AMZN'
    YEARS_BACK = 10   # O(M^2) Hungarian solves -- keep the demo fast with a recent slice
    P = 1   # 1 = uni-d 1-WK-means (uni_1d_wk.py), 2 = uni-d 2-WK-means (uni_2d_wk.py)

    data = load_all()
    log_ret = data['equities']['Log_Return'].unstack('Ticker')
    recent = log_ret.loc[log_ret.index.max() - pd.DateOffset(years=YEARS_BACK):]

    print(f'Step 1: univariate uni-d {P}-WK-means for {TICKER_A} and {TICKER_B}...')
    uniform = {}
    for ticker in (TICKER_A, TICKER_B):
        returns = recent[ticker].dropna()
        u, k = uniform_margin(returns, p=P)
        uniform[ticker] = u
        print(f'  {ticker}: k={k}, n={len(returns)}')

    print(f'\nStep 2: 2-d WK-means correlation clustering for {TICKER_A}-{TICKER_B} '
          f'(h1={H1}, h2={H2})...')
    result = cluster_pair(uniform[TICKER_A], uniform[TICKER_B])
    if result is None:
        print('  Not enough overlapping data for the 2-d step.')
    else:
        print(f'  k={result["k"]}')
        print(result['cluster_stats'].to_string())
        latest = int(result['regime'].dropna().iloc[-1])
        print(f'  latest correlation regime: cluster {latest}')
