"""Discrete hidden Markov model in plain NumPy.

Sequences are batched as an (N, T) integer array plus a boolean mask (sequences are
left-aligned; positions beyond a sequence's length are masked out). Implements:
  * scaled forward / backward recursions (the sum-product algorithm on a chain),
  * Baum-Welch (EM) maximum-likelihood training,
  * Viterbi decoding in log space.
"""
from __future__ import annotations

import numpy as np

FLOOR = 1e-8


def _emis(B: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(N, K) emission probabilities of observations `o` under each state."""
    return B[:, o].T


def _emis_all(B: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(N, T, K) emission probabilities for a whole (N, T) observation array."""
    return np.moveaxis(B[:, o], 0, -1)


def forward(pi, A, B, obs, mask):
    """Scaled forward pass. Returns alpha_hat (N,T,K) and scaling factors c (N,T).

    alpha_hat_t is the filtered state distribution p(z_t | x_1..t); c_t = p(x_t | x_1..t-1),
    so log-likelihood = sum(log c). Masked positions copy the previous alpha and use c = 1.
    """
    N, T = obs.shape
    alpha = np.zeros((N, T, len(pi)))
    c = np.ones((N, T))
    a = pi[None, :] * _emis(B, obs[:, 0])
    c0 = a.sum(1)
    alpha[:, 0] = a / c0[:, None]
    c[:, 0] = c0
    for t in range(1, T):
        a = (alpha[:, t - 1] @ A) * _emis(B, obs[:, t])
        m = mask[:, t]
        ct = np.where(m, a.sum(1), 1.0)
        alpha[:, t] = np.where(m[:, None], a / ct[:, None], alpha[:, t - 1])
        c[:, t] = ct
    return alpha, c


def backward(A, B, obs, mask, c):
    """Scaled backward pass using the forward scaling factors."""
    N, T = obs.shape
    beta = np.ones((N, T, A.shape[0]))
    for t in range(T - 2, -1, -1):
        b = ((_emis(B, obs[:, t + 1]) * beta[:, t + 1]) @ A.T) / c[:, t + 1][:, None]
        beta[:, t] = np.where(mask[:, t + 1][:, None], b, 1.0)
    return beta


def posteriors(alpha, beta, mask):
    """Smoothed state posteriors gamma_t = p(z_t | x_1..T), zero at masked positions."""
    g = alpha * beta
    g = g / g.sum(2, keepdims=True)
    return g * mask[:, :, None]


def viterbi(pi, A, B, obs, mask):
    """Most likely state path per sequence (-1 at masked positions) and its log-probability."""
    N, T = obs.shape
    K = len(pi)
    lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
    delta = lpi[None, :] + lB[:, obs[:, 0]].T
    psi = np.zeros((N, T, K), dtype=int)
    ident = np.arange(K)[None, :].repeat(N, 0)
    for t in range(1, T):
        cand = delta[:, :, None] + lA[None]
        new = cand.max(1) + lB[:, obs[:, t]].T
        m = mask[:, t][:, None]
        psi[:, t] = np.where(m, cand.argmax(1), ident)
        delta = np.where(m, new, delta)
    path = np.zeros((N, T), dtype=int)
    path[:, T - 1] = delta.argmax(1)
    rows = np.arange(N)
    for t in range(T - 2, -1, -1):
        path[:, t] = psi[rows, t + 1, path[:, t + 1]]
    return np.where(mask, path, -1), delta.max(1)


class DiscreteHMM:
    def __init__(self, n_states: int, n_symbols: int):
        self.K, self.M = n_states, n_symbols
        self.pi = self.A = self.B = None
        self.ll_history: list[float] = []
        self.converged = False

    # ------------------------------------------------------------ helpers
    @property
    def n_params(self) -> int:
        return (self.K - 1) + self.K * (self.K - 1) + self.K * (self.M - 1)

    def _init(self, rng):
        A = rng.dirichlet(np.ones(self.K) * 2, size=self.K) + np.eye(self.K) * 0.5
        self.A = A / A.sum(1, keepdims=True)
        self.B = rng.dirichlet(np.ones(self.M), size=self.K)
        self.pi = rng.dirichlet(np.ones(self.K))

    def loglik(self, obs, mask) -> float:
        _, c = forward(self.pi, self.A, self.B, obs, mask)
        return float(np.log(c).sum())

    def _em_step(self, obs, mask):
        alpha, c = forward(self.pi, self.A, self.B, obs, mask)
        beta = backward(self.A, self.B, obs, mask, c)
        gamma = posteriors(alpha, beta, mask)
        # expected transition counts: sum_t alpha_t(i) A_ij B_j(x_t+1) beta_t+1(j) / c_t+1
        nxt = _emis_all(self.B, obs[:, 1:]) * beta[:, 1:] / c[:, 1:, None]
        xi = self.A * np.einsum("ntk,ntl,nt->kl", alpha[:, :-1], nxt, mask[:, 1:])
        pi = gamma[:, 0].mean(0)
        trans_mass = gamma[:, :-1][mask[:, 1:]].sum(0)  # time steps that have a successor
        A = xi / np.maximum(trans_mass[:, None], FLOOR)
        onehot = (obs[:, :, None] == np.arange(self.M)[None, None, :]) & mask[:, :, None]
        B = np.einsum("ntk,ntm->km", gamma, onehot.astype(float))
        self.pi = self._norm(pi)
        self.A = self._norm(A)
        self.B = self._norm(B)
        return float(np.log(c).sum())

    @staticmethod
    def _norm(x):
        x = np.maximum(x, FLOOR)
        return x / x.sum(-1, keepdims=True)

    # ----------------------------------------------------------- training
    def fit(self, obs, mask, n_iter=100, tol=1e-5, n_init=3, seed=42):
        """Baum-Welch from `n_init` random starts; keeps the best log-likelihood."""
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_init):
            self._init(rng)
            hist, prev, conv = [], -np.inf, False
            for _ in range(n_iter):
                ll = self._em_step(obs, mask)
                hist.append(ll)
                if abs(ll - prev) < tol * max(abs(ll), 1.0):
                    conv = True
                    break
                prev = ll
            final = self.loglik(obs, mask)
            if best is None or final > best[0]:
                best = (final, self.pi.copy(), self.A.copy(), self.B.copy(), hist, conv)
        _, self.pi, self.A, self.B, self.ll_history, self.converged = best
        self._sort_states()
        return self

    def _sort_states(self):
        """Order states by expected symbol index (symbols are ordered by spend) to fix label switching."""
        order = np.argsort(self.B @ np.arange(self.M))
        self.pi, self.A, self.B = self.pi[order], self.A[np.ix_(order, order)], self.B[order]

    # ----------------------------------------------------------- inference
    def decode(self, obs, mask):
        return viterbi(self.pi, self.A, self.B, obs, mask)

    def filter_smooth(self, obs, mask):
        alpha, c = forward(self.pi, self.A, self.B, obs, mask)
        gamma = posteriors(alpha, backward(self.A, self.B, obs, mask, c), mask)
        return alpha, c, gamma

    def predict_next(self, alpha_last):
        """Next-step state and observation distributions from the filtered last state."""
        s = alpha_last @ self.A
        return s, s @ self.B

    def bic(self, ll: float, n_obs: int) -> float:
        return -2 * ll + self.n_params * np.log(n_obs)

    def aic(self, ll: float) -> float:
        return -2 * ll + 2 * self.n_params
