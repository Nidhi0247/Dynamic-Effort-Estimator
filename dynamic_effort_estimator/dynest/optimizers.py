import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import CLUSTERS
MATCH_BONUS = 1.25   
@dataclass
class OptResult:
    name: str
    genes: np.ndarray
    fitness: float
    history: list
    time_s: float
    evals: int
    metrics: dict
    conv_iter: int

class SprintProblem:
    W = dict(over=10.0, imb=1.0, delay=0.5, span=0.3, move=0.3, cost=1.0)

    def __init__(self, ids, effort, cluster, priority, team, sprint_days=10, focus=0.8, prev=None, n_sprints=None):
        self.ids = list(ids)
        self.effort = np.asarray(effort, float)
        self.n = len(self.ids)
        self.priority = np.asarray(priority, float)
        self.dev_names = list(team["name"])
        self.D = len(team)
        spec = [CLUSTERS.index(s) if s in CLUSTERS else -1 for s in team["specialty"]]
        eff = np.array([[team["speed"].iloc[d] * (MATCH_BONUS if spec[d] == k else 1.0) for k in range(3)]
                        for d in range(self.D)])
        self.cost = self.effort[:, None] / eff[:, np.asarray(cluster, int)].T      
        self.cap = sprint_days * focus
        total = self.effort.sum() / eff.mean()
        self.S = n_sprints or max(2, math.ceil(total / (self.D * self.cap * 0.8)))
        self.ub = np.r_[np.full(self.n, self.S), np.full(self.n, self.D)]
        self.pmax = max(self.priority.max(), 1)
        self.prev = None
        if prev:   
            sp = np.array([prev.get(i, (-1, -1))[0] for i in self.ids])
            dv = np.array([prev.get(i, (-1, -1))[1] for i in self.ids])
            self.prev = np.r_[sp, dv] if (sp >= 0).any() else None

    @classmethod
    def from_estimates(cls, table, team, sprint_days=10, focus=0.8, **kw):
        chunk = 0.75 * sprint_days * focus
        ids, eff, cl, pr = [], [], [], []
        for r in table.itertuples():
            k = max(1, math.ceil(r.effort / chunk))
            for j in range(k):
                ids.append(r.id if k == 1 else f"{r.id}#{j + 1}")
                eff.append(r.effort / k)
                cl.append(r.cluster)
                pr.append(r.priority)
        return cls(ids, eff, cl, pr, team, sprint_days=sprint_days, focus=focus, **kw)

    def warm_start(self, rng):
        if self.prev is None:
            return None
        rand = rng.integers(0, self.ub)
        return self.clip(np.where(self.prev >= 0, self.prev, rand))

    def clip(self, g):
        return np.minimum(np.maximum(g, 0), self.ub - 1).astype(int)

    def evaluate(self, g):
        n = self.n
        sp, dv = g[:n], g[n:]
        c = self.cost[np.arange(n), dv]
        load = np.bincount(sp * self.D + dv, weights=c, minlength=self.S * self.D)
        over = np.maximum(load - self.cap, 0).sum() / c.sum()
        dl = load.reshape(self.S, self.D).sum(0)
        imb = dl.std() / (dl.mean() + 1e-9)
        delay = (self.priority * sp * c).sum() / (c.sum() * self.S * self.pmax)
        span = (sp.max() + 1) / self.S
        move = 0.0
        if self.prev is not None:
            known = self.prev[:n] >= 0
            move = float((known & ((sp != self.prev[:n]) | (dv != self.prev[n:]))).mean())
        w = self.W
        return (w["over"] * over + w["imb"] * imb + w["delay"] * delay + w["span"] * span
                + w["move"] * move + w["cost"] * c.sum() / self.effort.sum())

    def metrics(self, g):
        n = self.n
        sp, dv = g[:n], g[n:]
        c = self.cost[np.arange(n), dv]
        load = np.bincount(sp * self.D + dv, weights=c, minlength=self.S * self.D)
        dl = load.reshape(self.S, self.D).sum(0)
        used = sp.max() + 1
        moved = int(((self.prev[:n] >= 0) & ((sp != self.prev[:n]) | (dv != self.prev[n:]))).sum()) if self.prev is not None else 0
        return dict(overload_days=float(np.maximum(load - self.cap, 0).sum()),
                    leveling_efficiency=float(dl.mean() / dl.max()) if dl.max() > 0 else 1.0,
                    utilization=float(c.sum() / (self.D * used * self.cap)),
                    sprints_used=int(used), effective_effort=float(c.sum()), tasks_moved=moved)

    def plan(self, g):
        n = self.n
        return pd.DataFrame(dict(id=self.ids, requirement=[i.split("#")[0] for i in self.ids], sprint=g[:n] + 1, developer=[self.dev_names[d] for d in g[n:]],
                                 effective_days=self.cost[np.arange(n), g[n:]].round(2)))

    def plan_map(self, g):
        return {i: (int(g[k]), int(g[self.n + k])) for k, i in enumerate(self.ids)}


def _conv_iter(h):
    final = h[-1]
    thr = final + 0.01 * (h[0] - final)
    return int(next(i for i, v in enumerate(h) if v <= thr + 1e-12))


def _finish(p, name, best, hist, t0, evals):
    return OptResult(name, best.copy(), float(p.evaluate(best)), hist, time.perf_counter() - t0, evals,
                     p.metrics(best), _conv_iter(hist))


def greedy_plan(p):
    load = np.zeros((p.S, p.D))
    sp, dv = np.zeros(p.n, int), np.zeros(p.n, int)
    for i in np.lexsort((-p.effort, -p.priority)):
        for s in range(p.S):
            ok = [d for d in range(p.D) if load[s, d] + p.cost[i, d] <= p.cap]
            if ok:
                d = min(ok, key=lambda d: p.cost[i, d])
                break
        else:
            s, d = p.S - 1, int(np.argmin(load[-1]))
        sp[i], dv[i] = s, d
        load[s, d] += p.cost[i, d]
    return np.r_[sp, dv]


def run_ga(p, pop=60, iters=120, seed=0, init=None, cx=0.9, elite=2):
    rng, t0, n2 = np.random.default_rng(seed), time.perf_counter(), 2 * p.n
    mut = 1.5 / p.n
    P = rng.integers(0, p.ub, (pop, n2))
    if init is not None:
        P[0] = init
    fit = np.array([p.evaluate(g) for g in P])
    hist = [float(fit.min())]
    for _ in range(iters):
        order = np.argsort(fit)
        elites = P[order[:elite]].copy()
        tour = rng.integers(0, pop, (pop, 3))
        parents = P[tour[np.arange(pop), fit[tour].argmin(1)]]
        mates = parents[rng.permutation(pop)]
        mask = (rng.random((pop, n2)) < 0.5) & (rng.random((pop, 1)) < cx)
        kids = np.where(mask, mates, parents)
        m = rng.random((pop, n2)) < mut
        kids[m] = rng.integers(0, np.broadcast_to(p.ub, kids.shape)[m])
        kids[:elite] = elites
        P, fit = kids, np.array([p.evaluate(g) for g in kids])
        hist.append(float(fit.min()))
    return _finish(p, "GA", P[fit.argmin()], hist, t0, pop * (iters + 1))


def run_pso(p, swarm=60, iters=120, seed=0, init=None, w=0.72, c1=1.5, c2=1.5):
    rng, t0, n2 = np.random.default_rng(seed), time.perf_counter(), 2 * p.n
    hi = p.ub - 1.0
    dec = lambda X: p.clip(np.rint(X))
    X = rng.uniform(0, 1, (swarm, n2)) * hi
    V = rng.uniform(-1, 1, (swarm, n2)) * 0.2 * hi
    if init is not None:
        X[0] = init
    pb, pf = X.copy(), np.array([p.evaluate(g) for g in dec(X)])
    gi = pf.argmin()
    gb, gf = pb[gi].copy(), pf[gi]
    hist = [float(gf)]
    for _ in range(iters):
        V = w * V + c1 * rng.random(X.shape) * (pb - X) + c2 * rng.random(X.shape) * (gb - X)
        V = np.clip(V, -0.3 * hi, 0.3 * hi)
        X = np.clip(X + V, 0, hi)
        f = np.array([p.evaluate(g) for g in dec(X)])
        better = f < pf
        pb[better], pf[better] = X[better], f[better]
        if pf.min() < gf:
            gi = pf.argmin()
            gb, gf = pb[gi].copy(), pf[gi]
        hist.append(float(gf))
    return _finish(p, "PSO", dec(gb[None])[0], hist, t0, swarm * (iters + 1))


def run_both(p, seed=0, pop=60, iters=120, warm=True):
    """GA and PSO executed concurrently (Step 7)."""
    init = p.warm_start(np.random.default_rng(seed)) if warm else None
    with ThreadPoolExecutor(2) as ex:
        fg = ex.submit(run_ga, p, pop, iters, seed, init)
        fp = ex.submit(run_pso, p, pop, iters, seed, init)
        return {"GA": fg.result(), "PSO": fp.result()}


def benchmark(p, runs=5, pop=60, iters=120, seed=0):
    """Repeated runs -> summary table comparing GA vs PSO (+ greedy reference)."""
    rows, first = [], None
    for r in range(runs):
        res = run_both(p, seed + r, pop, iters, warm=False)
        first = first or res
        for k, v in res.items():
            rows.append(dict(algorithm=k, fitness=v.fitness, time_s=v.time_s, conv_iter=v.conv_iter,
                             leveling=v.metrics["leveling_efficiency"], utilization=v.metrics["utilization"],
                             overload=v.metrics["overload_days"]))
    df = pd.DataFrame(rows)
    summ = df.groupby("algorithm").agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    g = greedy_plan(p)
    gm = p.metrics(g)
    summ.loc["Greedy (reference)"] = [p.evaluate(g), np.nan, 0, np.nan, 0, np.nan, gm["leveling_efficiency"], np.nan,
                                      gm["utilization"], np.nan, gm["overload_days"], np.nan]
    return summ, first, df
