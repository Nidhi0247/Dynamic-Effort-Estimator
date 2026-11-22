"""Data-trained Mamdani FIS (replaces the hand-written 125-rule base).

* Membership functions: strong fuzzy partitions (ordered centres; term t is the triangle
  (c[t-1], c[t], c[t+1]) with shoulders at both ends).  Memberships of adjacent terms always
  sum to 1, so coverage and distinguishability hold by construction (interpretability).
* Rules: Wang-Mendel generation from data (weighted vote per antecedent cell, support pruning).
* Tuning: a real-coded GA moves the partition centres (input and output); rules are regenerated
  for every candidate data base, and fitness is MAE on an internal, time-later hold-out.
* Inference: min-AND, min-implication, max-aggregation, centroid defuzzification.
"""
import itertools
import numpy as np

_NAMES = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["low", "fairly_low", "fairly_high", "high"],
    5: ["very_low", "low", "medium", "high", "very_high"],
    6: ["very_low", "low", "fairly_low", "fairly_high", "high", "very_high"],
    7: ["very_low", "low", "fairly_low", "medium", "fairly_high", "high", "very_high"],
    9: ["extremely_low", "very_low", "low", "fairly_low", "medium", "fairly_high", "high", "very_high",
        "extremely_high"],
}


def term_names(n):
    return _NAMES.get(n, [f"T{i}" for i in range(n)])


def partition_memberships(x, centres):
    """(N,) values -> (N, T) memberships of a strong triangular partition."""
    x = np.asarray(x, float)[:, None]
    c = np.asarray(centres, float)
    T = len(c)
    mu = np.zeros((x.shape[0], T))
    for t in range(T):
        if t == 0:
            left = np.ones_like(x[:, 0])
        else:
            left = np.clip((x[:, 0] - c[t - 1]) / max(c[t] - c[t - 1], 1e-9), 0, 1)
        if t == T - 1:
            right = np.ones_like(x[:, 0])
        else:
            right = np.clip((c[t + 1] - x[:, 0]) / max(c[t + 1] - c[t], 1e-9), 0, 1)
        mu[:, t] = np.minimum(left, right)
    return mu


def _repair(c, lo, hi, min_gap):
    c = np.clip(np.sort(c), lo, hi)
    for i in range(1, len(c)):
        c[i] = max(c[i], c[i - 1] + min_gap)
    if c[-1] > hi:                              # shift back from the right if pushed over the edge
        c[-1] = hi
        for i in range(len(c) - 2, -1, -1):
            c[i] = min(c[i], c[i + 1] - min_gap)
    return c


class LearnedMamdani:
    def __init__(self, input_names, n_terms=(3, 3, 3), n_out=7, min_support=2, resolution=201, seed=0):
        self.input_names = list(input_names)
        self.n_terms = list(n_terms)
        self.n_out = n_out
        self.min_support = min_support
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        self.rules = []          # list of dict(ante=(t1,..), cons=o, support=n, weight=w)

    # ------------------------------------------------------------------ data base
    def _init_db(self, X, y):
        self.lo = np.quantile(X, 0.01, axis=0)
        self.hi = np.quantile(X, 0.99, axis=0)
        same = self.hi - self.lo < 1e-9
        self.hi[same] = self.lo[same] + 1.0
        self.ylo, self.yhi = np.quantile(y, 0.01), np.quantile(y, 0.99)
        if self.yhi - self.ylo < 1e-9:
            self.yhi = self.ylo + 1.0
        self.in_centres = [np.quantile(X[:, i], np.linspace(0.05, 0.95, k)) for i, k in enumerate(self.n_terms)]
        self.in_centres = [_repair(c, self.lo[i], self.hi[i], 0.04 * (self.hi[i] - self.lo[i]))
                           for i, c in enumerate(self.in_centres)]
        self.out_centres = _repair(np.quantile(y, np.linspace(0.05, 0.95, self.n_out)), self.ylo, self.yhi,
                                   0.03 * (self.yhi - self.ylo))
        self._set_output_grid()

    def _set_output_grid(self):
        self.ygrid = np.linspace(self.ylo, self.yhi, self.resolution)
        self.out_mu = partition_memberships(self.ygrid, self.out_centres).T        # (O, Y)

    def _genome(self):
        parts = [(c - self.lo[i]) / (self.hi[i] - self.lo[i]) for i, c in enumerate(self.in_centres)]
        parts.append((self.out_centres - self.ylo) / (self.yhi - self.ylo))
        return np.concatenate(parts)

    def _set_genome(self, g):
        k = 0
        for i, n in enumerate(self.n_terms):
            seg = self.lo[i] + g[k:k + n] * (self.hi[i] - self.lo[i])
            self.in_centres[i] = _repair(seg, self.lo[i], self.hi[i], 0.04 * (self.hi[i] - self.lo[i]))
            k += n
        seg = self.ylo + g[k:k + self.n_out] * (self.yhi - self.ylo)
        self.out_centres = _repair(seg, self.ylo, self.yhi, 0.03 * (self.yhi - self.ylo))
        self._set_output_grid()

    # ------------------------------------------------------------------ rule base
    def _memberships(self, X):
        return [partition_memberships(X[:, i], self.in_centres[i]) for i in range(len(self.n_terms))]

    def _wang_mendel(self, X, y):
        mus = self._memberships(X)
        ante = np.stack([m.argmax(1) for m in mus], 1)                                # (N, I)
        deg_in = np.prod([m[np.arange(len(X)), ante[:, i]] for i, m in enumerate(mus)], axis=0)
        # consequent = output term with the highest membership of the cell's membership-weighted mean target
        # (a vote over terms collapses noisy data onto the most populated middle term)
        cells = {}
        for a, d, yy in zip(map(tuple, ante), deg_in, y):
            c = cells.setdefault(a, [0.0, 0.0, 0])
            c[0] += d * yy
            c[1] += d
            c[2] += 1
        rules = []
        for a, (sy, sw, n) in cells.items():
            if n >= self.min_support and sw > 0:
                ybar = sy / sw
                mu = partition_memberships(np.array([ybar]), self.out_centres)[0]
                rules.append(dict(ante=a, cons=int(mu.argmax()), support=n, weight=float(sw), mean=float(ybar)))
        self.rules = sorted(rules, key=lambda r: r["ante"])
        self.default = float(np.median(y))
        self._compile()

    def _compile(self):
        self._A = np.array([r["ante"] for r in self.rules], int).reshape(-1, len(self.n_terms))
        self._C = np.array([r["cons"] for r in self.rules], int)

    # ------------------------------------------------------------------ inference
    def infer(self, X, return_strengths=False):
        X = np.atleast_2d(np.asarray(X, float))
        mus = self._memberships(X)
        n = len(X)
        if len(self.rules) == 0:
            crisp = np.full(n, self.default)
            return (crisp, np.zeros((n, 0)), np.zeros((n, self.resolution))) if return_strengths else crisp
        strength = np.minimum.reduce([mus[i][:, self._A[:, i]] for i in range(len(self.n_terms))])   # (N, R)
        term = np.zeros((n, self.n_out))
        for o in range(self.n_out):
            sel = self._C == o
            if sel.any():
                term[:, o] = strength[:, sel].max(1)
        agg = np.max(np.minimum(term[:, :, None], self.out_mu[None]), axis=1)             # (N, Y)
        area = agg.sum(1)
        crisp = np.where(area > 1e-12, (agg * self.ygrid).sum(1) / np.maximum(area, 1e-12), self.default)
        return (crisp, strength, agg) if return_strengths else crisp

    def predict(self, X):
        return self.infer(X)

    # ------------------------------------------------------------------ training
    def fit(self, X, y, tune=True, pop=24, gens=30, holdout=0.25):
        """X, y must be in time order; the last `holdout` share is the GA's internal fitness set."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._init_db(X, y)
        self._wang_mendel(X, y)
        if not tune or len(X) < 40:
            return self
        cut = int(len(X) * (1 - holdout))
        Xa, ya, Xb, yb = X[:cut], y[:cut], X[cut:], y[cut:]

        def fitness(g):
            self._set_genome(g)
            self._wang_mendel(Xa, ya)
            return float(np.abs(self.infer(Xb) - yb).mean())

        g0 = self._genome()
        dim = len(g0)
        P = np.clip(g0 + self.rng.normal(0, 0.08, (pop, dim)), 0, 1)
        P[0] = g0
        F = np.array([fitness(g) for g in P])
        self.tune_history = [float(F.min())]
        for _ in range(gens):
            order = np.argsort(F)
            elite = P[order[:2]].copy()
            t = self.rng.integers(0, pop, (pop, 2))
            parents = P[np.where(F[t[:, 0]] < F[t[:, 1]], t[:, 0], t[:, 1])]
            mates = parents[self.rng.permutation(pop)]
            a = self.rng.uniform(-0.25, 1.25, (pop, dim))                   # BLX-alpha crossover
            kids = np.clip(parents + a * (mates - parents), 0, 1)
            m = self.rng.random((pop, dim)) < 1.0 / dim
            kids[m] = np.clip(kids[m] + self.rng.normal(0, 0.1, m.sum()), 0, 1)
            kids[:2] = elite
            P = kids
            F = np.array([fitness(g) for g in P])
            self.tune_history.append(float(F.min()))
        self._set_genome(P[F.argmin()])
        self._wang_mendel(X, y)                                                # final rules on all data
        return self

    # ------------------------------------------------------------------ explanation
    def rule_text(self, r):
        parts = [f"{n} IS {term_names(k)[t]}" for n, k, t in zip(self.input_names, self.n_terms, r["ante"])]
        return "IF " + " AND ".join(parts) + f" THEN effort IS {term_names(self.n_out)[r['cons']]}"

    def rule_table(self):
        import pandas as pd
        return pd.DataFrame([dict(rule=self.rule_text(r), support=r["support"], weight=round(r["weight"], 3))
                             for r in self.rules])

    def explain(self, x, top=5, min_strength=1e-3):
        crisp, s, agg = self.infer(np.atleast_2d(x), return_strengths=True)
        order = np.argsort(-s[0]) if s.shape[1] else []
        fired = [dict(rule=self.rule_text(self.rules[i]), firing_strength=float(s[0, i]),
                      support=self.rules[i]["support"]) for i in order[:top] if s[0, i] > min_strength]
        return float(crisp[0]), fired, agg[0]

    def coverage(self, X):
        """Share of inputs for which at least one rule fires (uncovered inputs use the default)."""
        _, s, _ = self.infer(X, return_strengths=True)
        return float((s.max(1) > 1e-9).mean()) if s.shape[1] else 0.0
