'''
Wasserstein k-means (WK-means) on windows of returns.

Follows Mc Greevy, Muguruza, Issa, Salvi, Chan & Zuric (2024), 'Detecting
multivariate market regimes via clustering algorithms', SSRN 4758243,
section 2.3 and 3.1:
  * every window of h1 returns (d series) is an empirical measure: a cloud of
    h1 points in R^d with equal weights;
  * distance: 2-Wasserstein between two clouds of equal size, solved exactly
    as an optimal assignment (Hungarian algorithm via scipy
    linear_sum_assignment on the cdist cost matrix), sec. 3.1.2:
        W2(X, Y)^2 = min_pi (1/h1) sum_i ||X_i - Y_pi(i)||^2.
    The 1/h1 is the weight of each atom of the empirical measure (def. B.5);
    the formula in sec. 3.1.2 omits it.  With equal h1 everywhere it scales
    every distance by the same constant, so no assignment, medoid or
    p_crisis changes;
  * centroid: the barycentre (def. B.6, argmin of the summed W2) is
    restricted to the measures being clustered, i.e. the member with the
    smallest row sum of the cluster's W2 matrix (sec. 3.1.2, Exhibit 1);
  * k-means: k-means++ initialisation with W2 as the distance (D^2
    weighting), assign each measure to its nearest centroid, recompute
    centroids (sec. 2.3, steps 1-3).  Stopping rule (def. B.4): centroid
    shift below epsilon.  Medoids are discrete, so this is applied with
    epsilon -> 0: stop when the assignment, hence the medoids, no longer
    change.  n_init restarts keep the lowest total distance (our addition;
    the paper does not state restarts).
The same code handles d = 1 (then W2 is the distance between sorted windows;
the paper's 'uni-d 2-WK-means', sec. 4.1.1).
'''

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def w2(X: np.ndarray, Y: np.ndarray) -> float:
    '''2-Wasserstein distance between two equal-size point clouds (h1, d).'''
    C = cdist(X, Y, 'sqeuclidean')
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].mean()))


def _rows(W: np.ndarray, rows: range) -> list[np.ndarray]:
    return [np.array([w2(W[i], W[j]) for j in range(i + 1, len(W))]) for i in rows]


def distance_matrix(W: np.ndarray, n_jobs: int = -1, chunk: int = 50) -> np.ndarray:
    '''Symmetric matrix of pairwise W2 distances between windows W (M, h1, d).'''
    M = len(W)
    chunks = [range(a, min(a + chunk, M)) for a in range(0, M, chunk)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_rows)(W, rows) for rows in chunks)
    D = np.zeros((M, M), dtype=np.float32)
    for rows, vals in zip(chunks, parts):
        for i, v in zip(rows, vals):
            D[i, i + 1:] = v
    return D + D.T


def _medoid(D: np.ndarray, members: np.ndarray) -> int:
    '''Member with the smallest summed distance to the other members (Exhibit 1).'''
    sub = D[np.ix_(members, members)]
    return int(members[np.argmin(sub.sum(axis=1))])


def _kmeanspp(D: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    '''k-means++ seeding with the W2 distance (sec. 2.3, step 1).'''
    M = len(D)
    cent = [int(rng.integers(M))]
    for _ in range(1, k):
        d2 = np.min(D[:, cent], axis=1).astype(float) ** 2
        cent.append(int(rng.choice(M, p=d2 / d2.sum())) if d2.sum() > 0 else int(rng.integers(M)))
    return cent


def wk_means(D: np.ndarray, k: int = 2, n_init: int = 10, max_iter: int = 100, seed: int = 0):
    '''
    WK-means on a precomputed W2 distance matrix.  Runs n_init k-means++
    starts and keeps the one with the smallest total distance to the
    centroids.  Returns (labels, centroid indices, cost).
    '''
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        cent = _kmeanspp(D, k, rng)
        labels = np.argmin(D[:, cent], axis=1)
        for _ in range(max_iter):
            if len(np.unique(labels)) < k:            # empty cluster: reseed
                break
            cent = [_medoid(D, np.flatnonzero(labels == j)) for j in range(k)]
            new = np.argmin(D[:, cent], axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
        if len(np.unique(labels)) < k:
            continue
        cost = float(D[np.arange(len(D)), np.asarray(cent)[labels]].sum())
        if best is None or cost < best[2]:
            best = (labels, np.asarray(cent), cost)
    if best is None:
        raise RuntimeError('WK-means: no start produced k non-empty clusters')
    return best
