"""Single-objective (GA, PSO, greedy) and multi-objective (NSGA-II, MOPSO) optimisers on random keys.

All optimisers share: the same decoder (feasibility by construction), the same evaluation budget,
the same per-instance normaliser and the same warm start (previous plan encoded as keys).
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .objectives import Normalizer, nondominated, objectives, scalarize, tchebycheff, indicators


@dataclass
class Result:
    name: str
    keys: np.ndarray
    F: np.ndarray                 # raw objectives of the returned plan
    score: float                  # scalarised normalised score (SO) or Tchebycheff of compromise (MO)
    history: list
    time_s: float
    evals: int
    conv_iter: int
    front_keys: np.ndarray = None
    front_F: np.ndarray = None
    extra: dict = field(default_factory=dict)


class Problem:
    def __init__(self, inst, weights=None, seed=0, n_probe=60):
        self.inst = inst
        self.w = np.full(4, 0.25) if weights is None else np.asarray(weights, float) / np.sum(weights)
        rng = np.random.default_rng(seed)
        probe = [rng.random(2 * inst.n) for _ in range(n_probe)]
        F = [self.raw(z) for z in probe]
        F.append(objectives(inst, inst.decode(inst.greedy_keys(), dev_rule="earliest")))
        self.norm = Normalizer().fit(F)
        self.evals = 0

    def raw(self, z):
        return objectives(self.inst, self.inst.decode(z))

    def F(self, Z):
        self.evals += len(Z)
        return np.array([self.raw(z) for z in Z])

    def score(self, F):
        return scalarize(self.norm(F), self.w)


def _conv_iter(h):
    h = np.asarray(h)
    thr = h[-1] + 0.01 * (h[0] - h[-1])
    return int(np.argmax(h <= thr + 1e-12))


def greedy(prob):
    t0 = time.perf_counter()
    z = prob.inst.greedy_keys()
    plan = prob.inst.decode(z, dev_rule="earliest")
    F = objectives(prob.inst, plan)
    return Result("Greedy", z, F, float(prob.score(F[None])[0]), [0.0], time.perf_counter() - t0, 1, 0,
                  extra=dict(dev_rule="earliest"))


def run_ga(prob, pop=60, gens=100, seed=0, init=None, cx=0.9, elite=2, tour=3):
    rng, t0, dim = np.random.default_rng(seed), time.perf_counter(), 2 * prob.inst.n
    P = rng.random((pop, dim))
    if init is not None:
        P[0] = init
    F = prob.F(P)
    S = prob.score(F)
    hist = [float(S.min())]
    mut = 1.5 / prob.inst.n
    for _ in range(gens):
        order = np.argsort(S)
        el = P[order[:elite]].copy()
        T = rng.integers(0, pop, (pop, tour))
        parents = P[T[np.arange(pop), S[T].argmin(1)]]
        mates = parents[rng.permutation(pop)]
        mask = (rng.random((pop, dim)) < 0.5) & (rng.random((pop, 1)) < cx)
        kids = np.where(mask, mates, parents)
        m = rng.random((pop, dim)) < mut
        kids[m] = np.clip(kids[m] + rng.normal(0, 0.15, m.sum()), 0, 1)
        kids[:elite] = el
        P = kids
        F = prob.F(P)
        S = prob.score(F)
        hist.append(float(min(S.min(), hist[-1])))
    b = S.argmin()
    return Result("GA", P[b].copy(), F[b], float(S[b]), hist, time.perf_counter() - t0, pop * (gens + 1),
                  _conv_iter(hist))


def run_pso(prob, swarm=60, gens=100, seed=0, init=None, w=0.72, c1=1.5, c2=1.5, vmax=0.2):
    rng, t0, dim = np.random.default_rng(seed), time.perf_counter(), 2 * prob.inst.n
    X = rng.random((swarm, dim))
    V = rng.uniform(-vmax, vmax, (swarm, dim))
    if init is not None:
        X[0] = init
    F = prob.F(X)
    S = prob.score(F)
    pb, pbS, pbF = X.copy(), S.copy(), F.copy()
    g = pbS.argmin()
    gb, gbS, gbF = pb[g].copy(), pbS[g], pbF[g].copy()
    hist = [float(gbS)]
    for _ in range(gens):
        V = w * V + c1 * rng.random(X.shape) * (pb - X) + c2 * rng.random(X.shape) * (gb - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0, 1)
        F = prob.F(X)
        S = prob.score(F)
        better = S < pbS
        pb[better], pbS[better], pbF[better] = X[better], S[better], F[better]
        g = pbS.argmin()
        if pbS[g] < gbS:
            gb, gbS, gbF = pb[g].copy(), pbS[g], pbF[g].copy()
        hist.append(float(gbS))
    return Result("PSO", gb, gbF, float(gbS), hist, time.perf_counter() - t0, swarm * (gens + 1), _conv_iter(hist))


# ====================================================================== multi-objective
def _fast_nds(F):
    n = len(F)
    dominated = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for i in range(n):
        le = np.all(F[i] <= F, 1) & np.any(F[i] < F, 1)
        ge = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        dominated[i] = list(np.where(le)[0])
        cnt[i] = ge.sum()
    fronts, cur = [], list(np.where(cnt == 0)[0])
    rank = np.zeros(n, int)
    r = 0
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            rank[i] = r
            for j in dominated[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur, r = nxt, r + 1
    return fronts, rank


def _crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def _sbx_pm(P1, P2, rng, eta_c=15, eta_m=20, pc=0.9):
    n, dim = P1.shape
    u = rng.random((n, dim))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u) + 1e-12)) ** (1 / (eta_c + 1)))
    do = (rng.random((n, 1)) < pc) & (rng.random((n, dim)) < 0.5)
    C = np.where(do, 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), P1)
    m = rng.random((n, dim)) < 1.0 / dim
    u = rng.random((n, dim))
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    C = np.where(m, C + delta, C)
    return np.clip(C, 0, 1)


def run_nsga2(prob, pop=60, gens=100, seed=0, init=None):
    rng, t0, dim = np.random.default_rng(seed), time.perf_counter(), 2 * prob.inst.n
    P = rng.random((pop, dim))
    if init is not None:
        P[0] = init
    F = prob.F(P)
    hist = []
    for _ in range(gens):
        Fn = prob.norm(F)
        fronts, rank = _fast_nds(Fn)
        crowd = np.zeros(pop)
        for fr in fronts:
            crowd[fr] = _crowding(Fn[fr])
        a, b = rng.integers(0, pop, (2, pop))
        win = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])), a, b)
        par = P[win]
        kids = _sbx_pm(par, par[rng.permutation(pop)], rng)
        KF = prob.F(kids)
        U, UF = np.vstack([P, kids]), np.vstack([F, KF])
        Un = prob.norm(UF)
        fronts, _ = _fast_nds(Un)
        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= pop:
                sel += fr
            else:
                cd = _crowding(Un[fr])
                sel += [fr[i] for i in np.argsort(-cd)[:pop - len(sel)]]
                break
        P, F = U[sel], UF[sel]
        hist.append(hypervolume(prob.norm(F[nondominated(prob.norm(F))])))
    nd = nondominated(prob.norm(F))
    return _finish_mo("NSGA-II", prob, P[nd], F[nd], hist, t0, pop * (gens + 1))


def run_mopso(prob, swarm=60, gens=100, seed=0, init=None, w=0.4, c1=1.5, c2=1.5, vmax=0.2, archive=60, pm=0.1):
    rng, t0, dim = np.random.default_rng(seed), time.perf_counter(), 2 * prob.inst.n
    X = rng.random((swarm, dim))
    if init is not None:
        X[0] = init
    V = np.zeros_like(X)
    F = prob.F(X)
    pb, pbF = X.copy(), F.copy()
    A, AF = _archive_update(np.empty((0, dim)), np.empty((0, 4)), X, F, prob, archive)
    hist = []
    for it in range(gens):
        cd = _crowding(prob.norm(AF)) if len(AF) > 2 else np.ones(len(AF))
        a, b = rng.integers(0, len(A), (2, swarm))
        lead = A[np.where(cd[a] >= cd[b], a, b)]
        V = w * V + c1 * rng.random(X.shape) * (pb - X) + c2 * rng.random(X.shape) * (lead - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0, 1)
        rate = pm * (1 - it / gens)
        mu = rng.random(X.shape) < rate / 2
        X[mu] = rng.random(mu.sum())
        F = prob.F(X)
        Fn, pbn = prob.norm(F), prob.norm(pbF)
        dom_new = np.all(Fn <= pbn, 1) & np.any(Fn < pbn, 1)
        incomp = ~dom_new & ~(np.all(pbn <= Fn, 1) & np.any(pbn < Fn, 1))
        upd = dom_new | (incomp & (rng.random(swarm) < 0.5))
        pb[upd], pbF[upd] = X[upd], F[upd]
        A, AF = _archive_update(A, AF, X, F, prob, archive)
        hist.append(hypervolume(prob.norm(AF)))
    return _finish_mo("MOPSO", prob, A, AF, hist, t0, swarm * (gens + 1))


def _archive_update(A, AF, X, F, prob, cap):
    U, UF = np.vstack([A, X]), np.vstack([AF, F])
    _, uniq = np.unique(np.round(UF, 9), axis=0, return_index=True)
    U, UF = U[uniq], UF[uniq]
    nd = nondominated(prob.norm(UF))
    U, UF = U[nd], UF[nd]
    while len(U) > cap:
        cd = _crowding(prob.norm(UF))
        k = int(np.argmin(cd))
        U, UF = np.delete(U, k, 0), np.delete(UF, k, 0)
    return U, UF


def _finish_mo(name, prob, Z, F, hist, t0, evals):
    k = compromise(prob, Z, F)
    return Result(name, Z[k].copy(), F[k], float(tchebycheff(prob.norm(F[k]), prob.w)[0]),
                  hist, time.perf_counter() - t0, evals, _conv_iter(np.max(hist) - np.asarray(hist)) if hist else 0,
                  front_keys=Z, front_F=F)


def compromise(prob, Z, F, churn_tol=None):
    """Decision rule: feasible (no overload) and within the learnt churn tolerance, then minimum
    augmented Tchebycheff distance to the ideal under the learnt preference weights."""
    ok = np.ones(len(F), bool)
    for i, z in enumerate(Z):
        ok[i] = indicators(prob.inst, prob.inst.decode(z))["overload_h"] <= 1e-6
    tol = getattr(prob, "churn_tol", None) if churn_tol is None else churn_tol
    if tol is not None:
        t = tol
        while not (ok & (F[:, 3] <= t)).any() and t < 1.0:
            t += 0.05
        ok &= F[:, 3] <= t + 1e-12
    if not ok.any():
        ok[:] = True
    idx = np.where(ok)[0]
    return int(idx[np.argmin(tchebycheff(prob.norm(F[idx]), prob.w))])


# ====================================================================== indicators
def hypervolume(Fn, ref=1.1):
    P = np.asarray(Fn, float)
    if len(P) == 0:
        return 0.0
    P = np.minimum(P, ref)
    P = P[nondominated(P)]
    return float(_hv(P, np.full(P.shape[1], ref)))


def _hv(P, ref):
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 1:
        return float(ref[0] - P[:, 0].min())
    o = np.argsort(P[:, -1])
    P = P[o]
    vol = 0.0
    for i in range(len(P)):
        top = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        h = top - P[i, -1]
        if h > 0:
            sub = P[:i + 1, :-1]
            sub = sub[nondominated(sub)]
            vol += h * _hv(sub, ref[:-1])
    return vol


def igd_plus(Fn, ref_front):
    A, R = np.asarray(Fn, float), np.asarray(ref_front, float)
    d = np.sqrt((np.maximum(A[None] - R[:, None], 0) ** 2).sum(2))
    return float(d.min(1).mean())
