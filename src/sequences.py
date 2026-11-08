"""Monthly purchase sequences per customer and the HMM analysis built on them.

Each customer's sequence runs from their first purchase month to the last *complete* month in the
data. Each month is one of three observed symbols: no purchase / low spend / high spend, where the
low-high split is the median of all positive customer-month spends. Hidden states are the
customer's unobserved engagement level. This describes behaviour; it does not identify churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import DiscreteHMM

SYMBOLS = ["No purchase", "Low spend", "High spend"]
STATE_NAMES = {
    2: ["Inactive", "Active"],
    3: ["Dormant", "Casual", "Engaged"],
    4: ["Dormant", "Casual", "Regular", "High-spend"],
    5: ["Dormant", "Fading", "Casual", "Regular", "High-spend"],
    6: ["Dormant", "Fading", "Casual", "Occasional", "Regular", "High-spend"],
}


def state_names(k: int) -> list[str]:
    return [f"{n} (state {i})" for i, n in enumerate(STATE_NAMES.get(k, [f"State {i}" for i in range(k)]))]


def prepare_sequences(clean: pd.DataFrame) -> dict:
    """Clean transactions -> padded symbol sequences aligned with the sorted customer table."""
    months = clean["InvoiceDate"].dt.to_period("M")
    spend = clean.groupby(["CustomerID", months])["LineRevenue"].sum().unstack(fill_value=0.0)
    spend = spend.reindex(columns=pd.period_range(months.min(), months.max(), freq="M"), fill_value=0.0)
    last = clean["InvoiceDate"].max()
    dropped = last.day < last.days_in_month - 2  # final month is only partly observed
    if dropped:
        spend = spend.iloc[:, :-1]
    S = spend.to_numpy(float)
    pos = S[S > 0]
    thr = float(np.median(pos)) if pos.size else 0.0
    sym = np.where(S <= 0, 0, np.where(S <= thr, 1, 2))

    N, T = S.shape
    first = np.where((S > 0).any(1), (S > 0).argmax(1), T)  # T = no purchase in a complete month
    lengths = T - first
    obs = np.zeros((N, T), dtype=int)
    for i in range(N):
        if lengths[i]:
            obs[i, : lengths[i]] = sym[i, first[i]:]
    mask = np.arange(T)[None, :] < lengths[:, None]
    return {"ids": spend.index.astype(str).tolist(), "months": [str(m) for m in spend.columns], "obs": obs,
            "mask": mask, "start": first, "lengths": lengths, "threshold": thr, "dropped_partial": bool(dropped),
            "dropped_month": str(months.max()) if dropped else None}


def _valid(seqs: dict) -> np.ndarray:
    return seqs["lengths"] > 0


def fit_hmm(seqs: dict, k: int, n_init: int = 3) -> DiscreteHMM:
    v = _valid(seqs)
    return DiscreteHMM(k, len(SYMBOLS)).fit(seqs["obs"][v], seqs["mask"][v], n_init=n_init)


def sweep(seqs: dict, ks=(2, 3, 4, 5)) -> tuple[list[dict], dict[int, DiscreteHMM]]:
    v = _valid(seqs)
    obs, mask = seqs["obs"][v], seqs["mask"][v]
    n_obs = int(mask.sum())
    rows, models = [], {}
    for k in ks:
        m = fit_hmm(seqs, k)
        ll = m.loglik(obs, mask)
        models[k] = m
        rows.append({"k": k, "LogLik": ll, "AIC": m.aic(ll), "BIC": m.bic(ll, n_obs), "Parameters": m.n_params,
                     "EM iterations": len(m.ll_history), "Converged": m.converged})
    return rows, models


def backtest(seqs: dict, k: int, n_init: int = 3) -> dict:
    """Hold out each customer's last observed month and predict it from the earlier months.

    Compares the HMM with two simple baselines on accuracy and log-loss. Only customers with at
    least 3 observed months are scored; all models are trained on the shortened sequences only.
    """
    L = seqs["lengths"]
    keep = L >= 2
    obs, mask = seqs["obs"][keep].copy(), seqs["mask"][keep].copy()
    Ls = L[keep]
    rows = np.arange(len(Ls))
    target = obs[rows, Ls - 1].copy()
    mask[rows, Ls - 1] = False                       # hide the held-out month
    obs[rows, Ls - 1] = 0
    ev = Ls >= 3                                      # need >= 2 visible months to score

    model = DiscreteHMM(k, len(SYMBOLS)).fit(obs, mask, n_init=n_init)
    alpha = model.filter_smooth(obs, mask)[0]
    last = alpha[rows, Ls - 2]
    _, p_hmm = model.predict_next(last)

    M = len(SYMBOLS)
    marg = np.bincount(obs[mask], minlength=M) + 1.0
    marg /= marg.sum()
    cnt = np.ones((M, M))
    for t in range(obs.shape[1] - 1):
        ok = mask[:, t + 1]
        np.add.at(cnt, (obs[ok, t], obs[ok, t + 1]), 1)
    chain = cnt / cnt.sum(1, keepdims=True)
    prev = obs[rows, Ls - 2]
    preds = {"HMM": p_hmm, "Observed-symbol Markov chain": chain[prev],
             "Overall frequency (no sequence info)": np.tile(marg, (len(Ls), 1))}
    table = []
    for name, p in preds.items():
        p = np.clip(p[ev], 1e-9, 1)
        y = target[ev]
        table.append({"Method": name, "Accuracy": float((p.argmax(1) == y).mean()),
                      "LogLoss": float(-np.log(p[np.arange(len(y)), y]).mean())})
    return {"k": k, "n_evaluated": int(ev.sum()), "table": table,
            "actual_share": (np.bincount(target[ev], minlength=M) / ev.sum()).tolist()}


def customer_detail(seqs: dict, model: DiscreteHMM, idx: int) -> dict:
    """Forward-backward and Viterbi quantities for one customer."""
    L = int(seqs["lengths"][idx])
    if L == 0:
        raise ValueError("This customer has no purchase in a complete month, so there is no sequence.")
    obs, mask = seqs["obs"][idx:idx + 1, :L], seqs["mask"][idx:idx + 1, :L]
    alpha, c, gamma = model.filter_smooth(obs, mask)
    path, logp = model.decode(obs, mask)
    s, o = model.predict_next(alpha[0, -1])
    start = int(seqs["start"][idx])
    return {"id": seqs["ids"][idx], "months": seqs["months"][start:start + L], "obs": obs[0].tolist(),
            "filtered": alpha[0].tolist(), "smoothed": gamma[0].tolist(), "scaling": c[0].tolist(),
            "viterbi": path[0].tolist(), "log_likelihood": float(np.log(c).sum()), "viterbi_logprob": float(logp[0]),
            "next_state": s.tolist(), "next_obs": o.tolist()}


def build_payload(seqs: dict, models: dict, sweep_rows: list[dict], k: int, bt: dict) -> dict:
    model = models[k]
    v = _valid(seqs)
    obs, mask = seqs["obs"], seqs["mask"]
    N, T = obs.shape
    path = np.full((N, T), -1)
    nxt = np.full((N, len(SYMBOLS)), np.nan)
    if v.any():
        path[v], _ = model.decode(obs[v], mask[v])
        alpha, _, _ = model.filter_smooth(obs[v], mask[v])
        _, p = model.predict_next(alpha[np.arange(v.sum()), seqs["lengths"][v] - 1])
        nxt[v] = p
    cal = seqs["start"][:, None] + np.arange(T)[None, :]
    ok = mask & (cal < T)
    mix = np.zeros((T, model.K), dtype=int)
    np.add.at(mix, (cal[ok], path[ok]), 1)
    last_state = [int(path[i, seqs["lengths"][i] - 1]) if seqs["lengths"][i] else None for i in range(N)]
    best_k = min(sweep_rows, key=lambda r: r["BIC"])["k"]
    return {
        "meta": {"months": seqs["months"], "n_customers": N, "n_sequences": int(v.sum()), "n_observations": int(mask.sum()),
                 "threshold": seqs["threshold"], "symbols": SYMBOLS, "dropped_partial": seqs["dropped_partial"],
                 "dropped_month": seqs["dropped_month"], "k": k, "ks": [r["k"] for r in sweep_rows], "k_bic": best_k,
                 "state_names": state_names(k), "converged": model.converged},
        "sweep": sweep_rows,
        "model": {"pi": model.pi.tolist(), "A": model.A.tolist(), "B": model.B.tolist(), "ll_history": model.ll_history},
        "state_mix": mix.tolist(),
        "customers": {"id": seqs["ids"], "length": seqs["lengths"].tolist(), "last_state": last_state,
                      "next_obs": [None if np.isnan(r[0]) else r.tolist() for r in nxt]},
        "backtest": bt,
    }
