import itertools

import numpy as np
import pytest

from src import sequences as sq
from src.feature_engineering import build_customer_features
from src.hmm import DiscreteHMM, backward, forward, posteriors, viterbi
from src.preprocessing import clean_transactions

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.2, 0.8]])
B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])


def batch(seqs):
    T = max(map(len, seqs))
    obs = np.zeros((len(seqs), T), dtype=int)
    mask = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        obs[i, : len(s)], mask[i, : len(s)] = s, True
    return obs, mask


def brute(seq):
    """Enumerate every state path: returns total probability, best path, per-step marginals."""
    T, tot, best, bp = len(seq), 0.0, -1.0, None
    marg = np.zeros((T, 2))
    for z in itertools.product(range(2), repeat=T):
        p = PI[z[0]] * B[z[0], seq[0]]
        for t in range(1, T):
            p *= A[z[t - 1], z[t]] * B[z[t], seq[t]]
        tot += p
        if p > best:
            best, bp = p, z
        for t in range(T):
            marg[t, z[t]] += p
    return tot, bp, marg / tot


SEQS = [[0, 1, 2, 2], [2, 2], [1], [0, 0, 1, 2]]


def test_forward_matches_brute_force_with_variable_lengths():
    obs, mask = batch(SEQS)
    _, c = forward(PI, A, B, obs, mask)
    for i, s in enumerate(SEQS):
        assert np.isclose(np.log(c[i]).sum(), np.log(brute(s)[0]))


def test_posteriors_match_brute_force_and_sum_to_one():
    obs, mask = batch(SEQS)
    alpha, c = forward(PI, A, B, obs, mask)
    g = posteriors(alpha, backward(A, B, obs, mask, c), mask)
    for i, s in enumerate(SEQS):
        assert np.allclose(g[i, : len(s)], brute(s)[2])
        assert np.allclose(g[i, : len(s)].sum(1), 1)
        assert (g[i, len(s):] == 0).all()


def test_viterbi_matches_brute_force():
    obs, mask = batch(SEQS)
    path, logp = viterbi(PI, A, B, obs, mask)
    for i, s in enumerate(SEQS):
        tot, bp, _ = brute(s)
        assert tuple(path[i, : len(s)]) == bp
        assert (path[i, len(s):] == -1).all()
        assert np.isfinite(logp[i])


def test_baum_welch_never_decreases_likelihood():
    rng = np.random.default_rng(0)
    seqs = [list(rng.integers(0, 3, size=rng.integers(3, 9))) for _ in range(60)]
    obs, mask = batch(seqs)
    m = DiscreteHMM(2, 3)
    m._init(np.random.default_rng(1))
    lls = [m._em_step(obs, mask) for _ in range(25)]
    assert all(b >= a - 1e-8 for a, b in zip(lls, lls[1:]))
    for p in (m.pi, m.A, m.B):
        assert np.allclose(p.sum(-1), 1)


def test_fit_recovers_sticky_states():
    rng = np.random.default_rng(3)
    seqs = []
    for _ in range(300):  # one low-activity and one high-activity regime, rarely switching
        z, s = int(rng.random() < .5), []
        for _ in range(10):
            s.append(int(rng.choice(3, p=B[z])))
            z = z if rng.random() < .9 else 1 - z
        seqs.append(s)
    obs, mask = batch(seqs)
    m = DiscreteHMM(2, 3).fit(obs, mask, n_init=3)
    assert m.A[0, 0] > .75 and m.A[1, 1] > .75
    assert m.B[0] @ np.arange(3) < m.B[1] @ np.arange(3)  # states ordered by expected spend


def test_sequences_align_with_customer_table(raw):
    clean, _, _ = clean_transactions(raw)
    cust, _ = build_customer_features(clean)
    seqs = sq.prepare_sequences(clean)
    assert seqs["ids"] == cust["CustomerID"].tolist()
    assert set(np.unique(seqs["obs"][seqs["mask"]])) <= {0, 1, 2}
    assert (seqs["mask"].sum(1) == seqs["lengths"]).all()
    first = seqs["obs"][seqs["lengths"] > 0, 0]
    assert (first > 0).all()  # a sequence starts at the first purchase month


def test_sweep_backtest_and_payload(raw):
    import json
    clean, _, _ = clean_transactions(raw)
    seqs = sq.prepare_sequences(clean)
    rows, models = sq.sweep(seqs, ks=(2, 3))
    assert [r["k"] for r in rows] == [2, 3]
    bt = sq.backtest(seqs, 2, n_init=1)
    assert {r["Method"] for r in bt["table"]} >= {"HMM"} and bt["n_evaluated"] > 0
    p = sq.build_payload(seqs, models, rows, 2, bt)
    json.dumps(p, allow_nan=False)
    d = sq.customer_detail(seqs, models[2], int(np.argmax(seqs["lengths"])))
    assert len(d["viterbi"]) == len(d["obs"]) and np.isclose(sum(d["next_obs"]), 1)
