"""
uni_2d_wk.py
============
The uni-d 2-WK-means model from Remark 3.1: same algorithm as uni_1d_wk.py, but with
p=2 — distance and barycenter both use the 2-Wasserstein metric (pointwise mean of
sorted order statistics, the closed-form W2 barycenter) instead of 1-Wasserstein
(pointwise median). See uni_1d_wk.py for the shared implementation notes.
"""
import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REGIMES = os.path.dirname(_HERE)
if _REGIMES not in sys.path:
    sys.path.insert(0, _REGIMES)

import numpy as np
from scipy.stats import ecdf, kstest

from Finance.Regimes.data import load_all


# Change lines 50, 156, 237, 251 for h1, h2 
H1 = 15
H2 = 10

class UnivariateWKMeans:
    """
    Univariate (1D) Wasserstein K-Means (WK-means) Clustering — the uni-d 2-WK-means
    model (p=2) from Remark 3.1: distance and barycenter both use the 2-Wasserstein
    metric. p=1 (uni-d 1-WK-means) is also supported for completeness (see
    uni_1d_wk.py), but this file's demo runs the p=2 variant.

    Both cases are closed-form on sorted order statistics: minimizing
    sum_k |c_i - z_i^(k)|^p at each quantile index i independently gives the pointwise
    median (p=1) or pointwise mean (p=2) across the cluster's sorted segments — and
    since a pointwise median/mean of already-sorted vectors is itself sorted, the result
    is always a valid quantile function.
    """
    def __init__(self, k=2, p=2, max_iter=100, tol=1e-5, random_state=None):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None

    @staticmethod
    def lift_stream(returns, h1=H1, h2=H2):
        """
        Lifts 1D log-returns time series into overlapping segments/measures.

        Parameters:
            returns (np.ndarray): 1D array of log-returns (length N).
            h1 (int): Window size (default 5, one trading week of *daily* returns —
                the paper's h1=35/h2=28 are calibrated for hourly data, 35 market-hours
                being one week; on daily data the equivalent week is 5 trading days).
            h2 (int): Overlap size (default 4, one trading day short of h1).

        Returns:
            np.ndarray: Matrix of shape (M, h1) containing discrete empirical measures.
        """
        N = len(returns)
        step = h1 - h2
        M = (N - h2) // step

        segments = []
        for i in range(M):
            start = i * step
            end = start + h1
            if end <= N:
                segments.append(returns[start:end])

        return np.array(segments)

    def _wasserstein_1d_distance(self, u_sorted, v_sorted):
        """Calculates 1D p-Wasserstein distance between sorted empirical samples."""
        if self.p == 1:
            return np.mean(np.abs(u_sorted - v_sorted))
        elif self.p == 2:
            return np.sqrt(np.mean((u_sorted - v_sorted) ** 2))
        else:
            return (np.mean(np.abs(u_sorted - v_sorted) ** self.p)) ** (1.0 / self.p)

    def _update_centroid(self, cluster_points):
        """
        Pointwise Wasserstein barycenter of a cluster's sorted order statistics.

        Minimizing sum_k |c_i - z_i^(k)|^p at each index i separately is solved by the
        pointwise median (p=1) or pointwise mean (p=2) — the assignment step's distance
        metric (self.p) and this update must agree, or the algorithm isn't minimizing
        the objective it thinks it is.
        """
        if self.p == 1:
            return np.median(cluster_points, axis=0)
        elif self.p == 2:
            return np.mean(cluster_points, axis=0)
        raise NotImplementedError(
            f'Wasserstein barycenter update only implemented for p=1 (uni-d 1-WK-means) '
            f'and p=2 (uni-d 2-WK-means), got p={self.p}.'
        )

    def fit(self, X_measures):
        """
        Clusters empirical measures K = {u_1, ..., u_M}.

        Parameters:
            X_measures (np.ndarray): Array of shape (M, h1).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        M, h1 = X_measures.shape

        # Step 0: Pre-sort rows to easily compute 1D Quantiles/Wasserstein distance
        X_sorted = np.sort(X_measures, axis=1)

        # Step 1: Initialize k centroids uniformly at random
        init_indices = np.random.choice(M, size=self.k, replace=False)
        self.centroids_ = X_sorted[init_indices].copy()

        for iteration in range(self.max_iter):
            # Step 2: Assign clusters based on p-Wasserstein distance
            distances = np.zeros((M, self.k))
            for i in range(M):
                for j in range(self.k):
                    distances[i, j] = self._wasserstein_1d_distance(X_sorted[i], self.centroids_[j])

            labels = np.argmin(distances, axis=1)

            # Step 3: Compute 1D Wasserstein Barycentre for each cluster
            new_centroids = np.zeros_like(self.centroids_)
            for j in range(self.k):
                cluster_points = X_sorted[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = self._update_centroid(cluster_points)
                else:
                    # Handle empty cluster by reinitializing
                    new_centroids[j] = X_sorted[np.random.choice(M)]

            # Check stopping criteria based on centroid loss movement
            centroid_shift = np.sum([
                self._wasserstein_1d_distance(self.centroids_[j], new_centroids[j])
                for j in range(self.k)
            ])

            self.centroids_ = new_centroids
            self.labels_ = labels

            if centroid_shift < self.tol:
                break

        return self

    def transform_to_uniform(self, returns, h1=H1, h2=H2):
        """
        Step 1 of the paper's two-step approach:
        Applies empirical cumulative distribution function (eCDF) per cluster
        to yield uniform marginals on [0, 1] for subsequent copula/multivariate analysis.
        """
        X_measures = self.lift_stream(returns, h1=h1, h2=h2)
        if self.labels_ is None:
            self.fit(X_measures)

        transformed_returns = np.zeros_like(returns)
        step = h1 - h2

        # Apply cluster-specific empirical CDF
        for j in range(self.k):
            cluster_indices = np.where(self.labels_ == j)[0]
            if len(cluster_indices) == 0:
                continue

            # Aggregate all return points belonging to cluster j
            cluster_returns = X_measures[cluster_indices].flatten()

            # Compute empirical CDF mapping
            ecdf_res = ecdf(cluster_returns)

            # Transform segment points
            for idx in cluster_indices:
                start = idx * step
                end = start + h1
                transformed_returns[start:end] = ecdf_res.cdf.evaluate(returns[start:end])

        return transformed_returns


def _cluster_ks_pvalue(cluster_segments: np.ndarray) -> float:
    """
    Goodness-of-fit check for a cluster, per the paper's own description (Section 5,
    Step 1): apply the cluster's empirical CDF to its own points and test whether the
    transformed values look like Uniform(0, 1) -- the same ECDF transform used in
    transform_to_uniform, just pooled and run through kstest here.
    """
    if cluster_segments.shape[0] < 2:
        return 1.0
    flat = cluster_segments.ravel()
    transformed = ecdf(flat).cdf.evaluate(flat)
    _, p_value = kstest(transformed, 'uniform')
    return p_value


def select_k(measures: np.ndarray, k_min: int = 2, k_max: int = 6, alpha: float = 0.05,
             p: int = 2, random_state: int | None = None, verbose: bool = True
             ) -> tuple[int, 'UnivariateWKMeans']:
    """
    Fit UnivariateWKMeans at increasing k, Bonferroni-testing every cluster's uniformity
    (via `_cluster_ks_pvalue`), until all clusters pass (fail to reject) or k_max is
    reached. Returns the chosen k and the fitted model.
    """
    model = None
    for k in range(k_min, k_max + 1):
        model = UnivariateWKMeans(k=k, p=p, random_state=random_state)
        model.fit(measures)
        adjusted_alpha = alpha / k   # Bonferroni correction over the k clusters tested
        pvalues = [_cluster_ks_pvalue(measures[model.labels_ == j]) for j in range(k)]
        if verbose:
            pstr = ', '.join(f'{pv:.3f}' for pv in pvalues)
            print(f'  k={k}: cluster p-values = [{pstr}]  (need all > {adjusted_alpha:.4f})')
        if all(pv > adjusted_alpha for pv in pvalues):
            return k, model
    return k_max, model


if __name__ == '__main__':
    TICKERS = ['AAPL', 'AMZN']

    # 1. Pull real log-returns out of the project's equities database
    data    = load_all()
    log_ret = data['equities']['Log_Return'].unstack('Ticker')

    for TICKER in TICKERS:
        print(f'\n=== {TICKER} (uni-d 2-WK-means) ===')
        raw_returns = log_ret[TICKER].dropna().to_numpy()
        print(f'{TICKER}: {len(raw_returns)} daily log-returns')

        # 2. Lift raw returns into overlapping segments
        measures = UnivariateWKMeans.lift_stream(raw_returns, h1=H1, h2=H2)

        # 3. Fit 1D 2-Wasserstein K-Means, choosing k via the KS-uniformity test
        print('Selecting k via KS-uniformity test:')
        k, model = select_k(measures, k_min=2, k_max=6, alpha=0.05, p=2, random_state=42)
        print(f'Chosen k = {k}')

        print("Cluster labels assigned per window:", model.labels_[:10])
        counts = np.bincount(model.labels_)
        for j, n in enumerate(counts):
            mean_ret = measures[model.labels_ == j].mean()
            print(f'  cluster {j}: {n} windows, mean return {mean_ret:.5f}')

        # 4. Transform marginal distributions to Uniform [0, 1] using cluster eCDFs
        uniform_data = model.transform_to_uniform(raw_returns, h1=H1, h2=H2)
        print("Sample transformed uniform returns:", uniform_data[:5])
