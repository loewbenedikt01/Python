'''
Two-state Gaussian HMM with time-varying parameters, estimated adaptively.

Follows Nystrup, Madsen & Lindstrom:
  2016, J. Forecasting, 'Long memory of financial time series and hidden
        Markov models with time-varying parameters', section 4, eqs. (6)-(11);
  2018, Quantitative Finance, 'Dynamic portfolio optimization across hidden
        market regimes', section 2, eqs. (3)-(9).

Model:  y_t | S_t = i  ~  N(mu_i, sigma_i^2),  S_t a 2-state Markov chain with
        transition matrix Gamma = [[g11, 1-g11], [1-g22, g22]].

Estimation (adaptive, recursive):
  * objective: weighted conditional log-likelihood
        l~_t(theta) = sum_n  lambda^(t-n) log Pr(y_n | y_1..y_{n-1}, theta),
    lambda = 1 - 1/N_eff                                   (2018 eqs. 3-4);
  * update, one step per new observation                   (2018 eq. 7):
        theta_t = theta_{t-1} + A * I_t(theta_{t-1})^-1 * grad l~_t(theta_{t-1}),
    A = 1/N_eff;
  * Fisher information, updated recursively                 (2016 eq. 11):
        I_t = I_{t-1} + (1/t) (s_t s_t' - I_{t-1}),
    s_t = grad log Pr(y_t | y_1..y_{t-1}, theta_{t-1});
  * the weighted score uses the last 10 * N_eff observations; older ones carry
    weight < e^-10 (2016 p. 6: 2,500 observations for N_eff = 250);
  * parameters are transformed to be unconstrained (log sigma, logit g_ii),
    as the paper requires for convergence;
  * initialisation: numerical maximisation of the weighted likelihood on the
    first INIT_DAYS observations gives the starting parameters and Hessian
    (2016 footnote 7).

Gradients are central finite differences of the forward-algorithm likelihood;
they give the same score as the Lystig & Hughes (2002) recursion the paper uses.

Inference: forward filter only (2018 eq. 8), never smoothed, so every
probability uses data up to the current day.  States are reported ordered by
variance: state 0 = calm (low vol), state 1 = crisis (high vol).
'''

import logging
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize

logging.getLogger('hmmlearn').setLevel(logging.ERROR)

_LOG_2PI = float(np.log(2.0 * np.pi))
_FD_STEP = 1e-4                                     # finite-difference step on theta
_BOUNDS  = np.array([(-50.0, 50.0), (-50.0, 50.0),  # mu (scaled units)
                     (np.log(0.05), np.log(50.0)),  # log sigma
                     (np.log(0.05), np.log(50.0)),
                     (-12.0, 12.0), (-12.0, 12.0)]) # logit g11, logit g22


@dataclass
class HMMParams:
    pi:    np.ndarray      # (2,)   stationary distribution
    A:     np.ndarray      # (2, 2) transition matrix, rows sum to 1
    mu:    np.ndarray      # (2,)   emission means
    sigma: np.ndarray      # (2,)   emission std devs


# ----
# Parameter transforms
# ----

def to_params(th: np.ndarray) -> HMMParams:
    '''
    Unconstrained theta -> parameters, sorted so state 0 = calm, 1 = crisis.
    '''
    mu, sigma = th[:2], np.exp(th[2:4])
    g = 1.0 / (1.0 + np.exp(-th[4:6]))
    A = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    o = np.argsort(sigma)
    A = A[np.ix_(o, o)]
    pi = np.array([A[1, 0], A[0, 1]]) / (A[1, 0] + A[0, 1])
    return HMMParams(pi=pi, A=A, mu=mu[o], sigma=sigma[o])


def _to_theta(mu, sigma, A) -> np.ndarray:
    g = np.clip(np.diag(A), 1e-5, 1 - 1e-5)
    th = np.r_[mu, np.log(sigma), np.log(g / (1 - g))]
    return np.clip(th, _BOUNDS[:, 0], _BOUNDS[:, 1])


# ----
# Forward algorithm
# ----

def _forward(th: np.ndarray, x: np.ndarray):
    '''
    Normalised forward recursion, chain started in its stationary distribution.
    Returns log Pr(y_n | y_1..y_{n-1}) for every n, and the filtered state
    probabilities after the last observation (unsorted state order).
    '''
    m0, m1, ls0, ls1 = th[:4]
    g00 = 1.0 / (1.0 + np.exp(-th[4]))
    g11 = 1.0 / (1.0 + np.exp(-th[5]))
    l0 = -0.5 * ((x - m0) / np.exp(ls0)) ** 2 - ls0
    l1 = -0.5 * ((x - m1) / np.exp(ls1)) ** 2 - ls1
    mx = np.maximum(l0, l1)
    d0 = np.exp(l0 - mx).tolist()
    d1 = np.exp(l1 - mx).tolist()
    p0 = (1.0 - g11) / (2.0 - g00 - g11)
    c = np.empty(len(d0))
    a0 = p0
    for t in range(len(d0)):
        u0 = p0 * d0[t]
        u1 = (1.0 - p0) * d1[t]
        s = u0 + u1
        c[t] = s
        a0 = u0 / s
        p0 = a0 * g00 + (1.0 - a0) * (1.0 - g11)
    logc = np.log(c) + mx - 0.5 * _LOG_2PI
    return logc, np.array([a0, 1.0 - a0])


def filtered(th: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''
    P(S_T | y_1..y_T), sorted calm/crisis
    '''
    _, alpha = _forward(th, x)
    return alpha[np.argsort(th[2:4])]


def forget_weights(n: int, n_eff: float) -> np.ndarray:
    '''
    lambda^(T-n), lambda = 1 - 1/N_eff, newest observation last (weight 1).
    '''
    lam = 1.0 - 1.0 / n_eff
    return lam ** np.arange(n - 1, -1, -1, dtype=float)


# ----
# Initialisation: numerical maximisation of the weighted likelihood
# ----

def _starting_values(x: np.ndarray, n_init: int, seed: int) -> list[np.ndarray]:
    '''
    Unweighted EM fits from random starts, used only as optimiser starts.
    '''
    X = x.reshape(-1, 1)
    out = []
    for k in range(n_init):
        m = GaussianHMM(n_components=2, covariance_type='diag', n_iter=500, tol=1e-6,
                        min_covar=1e-4, random_state=seed + k, implementation='log')
        try:
            m.fit(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(m.transmat_)):
            out.append(_to_theta(m.means_.ravel(),
                                 np.sqrt(m.covars_.reshape(2, -1)[:, 0]), m.transmat_))
    if not out:
        raise RuntimeError('HMM initialisation: no EM start converged')
    return out


def _wnll(th, x, w) -> float:
    return -float(np.dot(w, _forward(th, x)[0]))


def initialise(x: np.ndarray, n_eff: float, n_init: int = 10, seed: int = 0):
    '''
    theta_t0 = argmax of the weighted likelihood on x (several starts), and
    I_t0 from its Hessian:  grad^2 l~ ~= -(1 - lambda^t)/(1 - lambda) I
    (2016 eq. 9).  Returns (theta, I).
    '''
    x = np.asarray(x, dtype=float)
    w = forget_weights(len(x), n_eff)
    best = None
    for th0 in _starting_values(x, n_init, seed):
        r = minimize(_wnll, th0, args=(x, w), method='L-BFGS-B', bounds=_BOUNDS)
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    th = best.x

    # numerical Hessian of minus the weighted log-likelihood
    k, h = len(th), 1e-3
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = H[j, i] = (_wnll(th + ei + ej, x, w) - _wnll(th + ei - ej, x, w)
                                 - _wnll(th - ei + ej, x, w) + _wnll(th - ei - ej, x, w)) / (4 * h * h)
    lam = 1.0 - 1.0 / n_eff
    I = _make_pd(H * (1.0 - lam) / (1.0 - lam ** len(x)))
    return th, I


def _make_pd(M: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    '''
    Symmetrise and lift non-positive eigenvalues, so I stays invertible.
    '''
    M = 0.5 * (M + M.T)
    v, U = np.linalg.eigh(M)
    vals = np.maximum(v, floor * max(1.0, v.max()))
    return np.dot(np.dot(U, np.diag(vals)), U.T)


# ----
# Recursive update, one observation
# ----

def adaptive_step(th: np.ndarray, I: np.ndarray, x: np.ndarray, t: int, n_eff: float):
    '''
    One step of the recursive adaptive estimator for the newest observation
    x[-1], at time t (number of observations seen so far, 1-based).
    'x' is the scoring window ending with the newest observation.
    Returns (theta_t, I_t, weighted log-likelihood at theta_{t-1}).
    '''
    w = forget_weights(len(x), n_eff)
    k = len(th)
    base, _ = _forward(th, x)
    G = np.empty((len(x), k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = _FD_STEP
        G[:, i] = (_forward(th + e, x)[0] - _forward(th - e, x)[0]) / (2 * _FD_STEP)

    score = np.dot(w, G)
    s_t = G[-1]
    I_new = I + (np.outer(s_t, s_t) - I) / t
    try:
        step = np.linalg.solve(I_new, score) / n_eff
    except np.linalg.LinAlgError:
        step = np.zeros(k)
    th_new = th + step if np.all(np.isfinite(step)) else th
    th_new = np.clip(th_new, _BOUNDS[:, 0], _BOUNDS[:, 1])
    return th_new, I_new, float(np.dot(w, base))
