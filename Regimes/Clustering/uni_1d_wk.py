import os
import sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_REGIMES = os.path.dirname(_HERE)
if _REGIMES not in sys.path:
    sys.path.insert(0, _REGIMES)

import numpy as np
from scipy.stats import ecdf, kstest

from data import load_all


class UnivariateWKMeans:
    """
    Univariate (1D) Wasserstein K-Means (WK-means) Clustering.
    Uses p-Wasserstein distance and quantile-based 1D Wasserstein barycentres.
    """
    def __init__(self, k=2, p=1, max_iter=100, tol=1e-5, random_state=None):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None

    @staticmethod
    def lift_stream(returns, h1=35, h2=28):
        """
        Lifts 1D log-returns time series into overlapping segments/measures.
        
        Parameters:
            returns (np.ndarray): 1D array of log-returns (length N).
            h1 (int): Window size (default 35, approx. 1 market week in hours).
            h2 (int): Overlap size (default 28, 1 market day short of h1).
            
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
                    # 1D Wasserstein Barycentre is the element-wise average of sorted samples
                    new_centroids[j] = np.mean(cluster_points, axis=0)
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

    def transform_to_uniform(self, returns, h1=35, h2=28):
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
    Leave-one-segment-out goodness-of-fit check for a cluster.

    For each member segment, build an ECDF from every *other* segment's points in the
    cluster, transform the held-out segment's own points through it, and pool the
    transformed values across the cluster. If the cluster is really one regime, the pool
    should look like Uniform(0, 1) -- kstest returns the p-value for that null.
    (Leave-one-out avoids the trivial degeneracy of testing an ECDF against the sample
    that built it.)
    """
    n_c = cluster_segments.shape[0]
    if n_c < 2:
        return 1.0
    flat = cluster_segments.reshape(n_c, -1)
    transformed = []
    for i in range(n_c):
        sorted_others = np.sort(np.delete(flat, i, axis=0).ravel())
        ranks = np.searchsorted(sorted_others, flat[i], side='right')
        transformed.append(ranks / (len(sorted_others) + 1))
    _, p_value = kstest(np.concatenate(transformed), 'uniform')
    return p_value


def select_k(measures: np.ndarray, k_min: int = 2, k_max: int = 6, alpha: float = 0.05,
             p: int = 1, random_state: int | None = None, verbose: bool = True
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
    TICKER = 'AAPL'

    # 1. Pull real log-returns for one ticker out of the project's equities database
    data     = load_all()
    log_ret  = data['equities']['Log_Return'].unstack('Ticker')
    raw_returns = log_ret[TICKER].dropna().to_numpy()
    print(f'{TICKER}: {len(raw_returns)} daily log-returns')

    # 2. Lift raw returns into overlapping segments (h1=35, h2=28 ~ 1 trading week)
    measures = UnivariateWKMeans.lift_stream(raw_returns, h1=35, h2=28)

    # 3. Fit 1D 1-Wasserstein K-Means, choosing k via the KS-uniformity test
    print('Selecting k via KS-uniformity test:')
    k, model = select_k(measures, k_min=2, k_max=6, alpha=0.05, p=1, random_state=42)
    print(f'Chosen k = {k}')

    print("Cluster labels assigned per window:", model.labels_[:10])
    counts = np.bincount(model.labels_)
    for j, n in enumerate(counts):
        mean_ret = measures[model.labels_ == j].mean()
        print(f'  cluster {j}: {n} windows, mean return {mean_ret:.5f}')

    # 4. Transform marginal distributions to Uniform [0, 1] using cluster eCDFs
    uniform_data = model.transform_to_uniform(raw_returns, h1=35, h2=28)
    print("Sample transformed uniform returns:", uniform_data[:5])