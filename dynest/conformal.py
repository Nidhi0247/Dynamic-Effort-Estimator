"""Learned uncertainty (S6): normalized, Mondrian, rolling split-conformal intervals.

Requirement level (log-effort space):
    score_i = |log y_i - log yhat_i| / sigma_hat(x_i)
    sigma_hat is a gradient-boosting model of |residual|, trained on the first half of the
    calibration window; quantiles of the score are taken on the second half (no reuse).
    Quantiles are per group (SOM cluster) when the group has enough points, else global.
Project level:
    score_s = |log(sum actual) - log(sum predicted)| over historical sprints; interval
    E * exp(+-q).  This captures residual correlation inside a sprint.
"""
from collections import deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _q(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


class ConformalIntervals:
    def __init__(self, alpha=0.05, min_group=30, window=2000, seed=0):
        self.alpha, self.min_group, self.window, self.seed = alpha, min_group, window, seed

    def fit(self, F_cal, resid_log_cal, groups_cal=None):
        """F_cal: difficulty features (time-ordered); resid_log_cal: log(actual) - log(pred)."""
        F_cal, r = np.asarray(F_cal, float), np.asarray(resid_log_cal, float)
        g = np.zeros(len(r), int) if groups_cal is None else np.asarray(groups_cal, int)
        half = len(r) // 2
        self.sigma_model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=8, learning_rate=0.05,
                                                         min_samples_leaf=20, random_state=self.seed)
        self.sigma_model.fit(F_cal[:half], np.abs(r[:half]))
        self.floor = max(float(np.quantile(np.abs(r[:half]), 0.1)), 1e-3)
        sig = self._sigma(F_cal[half:])
        scores = np.abs(r[half:]) / sig
        self.global_scores = deque(scores, maxlen=self.window)
        self.group_scores = {}
        for gi, sc in zip(g[half:], scores):
            self.group_scores.setdefault(int(gi), deque(maxlen=self.window)).append(sc)
        return self

    def _sigma(self, F):
        return np.maximum(self.sigma_model.predict(np.asarray(F, float)), self.floor)

    def quantile(self, group=None):
        gs = self.group_scores.get(int(group)) if group is not None else None
        if gs is not None and len(gs) >= self.min_group:
            return _q(gs, self.alpha)
        return _q(self.global_scores, self.alpha)

    def interval(self, pred, F, groups=None):
        """pred in hours (>0). Returns (low, high, sigma_log)."""
        pred = np.asarray(pred, float)
        sig = self._sigma(F)
        groups = np.zeros(len(pred), int) if groups is None else np.asarray(groups, int)
        q = np.array([self.quantile(g) for g in groups])
        half = q * sig
        return pred * np.exp(-half), pred * np.exp(half), sig

    def update(self, F, resid_log, groups=None):
        """Rolling conformal: add newly observed residuals (resolved issues)."""
        sig = self._sigma(np.atleast_2d(F))
        sc = np.abs(np.atleast_1d(resid_log)) / sig
        groups = np.zeros(len(sc), int) if groups is None else np.atleast_1d(groups)
        for gi, s in zip(groups, sc):
            self.global_scores.append(s)
            self.group_scores.setdefault(int(gi), deque(maxlen=self.window)).append(s)


class ProjectConformal:
    """Project/scope-level interval for the sum of n requirement estimates.

    Learnt from validation (actual, predicted, project) triples: for a scope of size n we draw many groups
    of n issues *within one project* (keeping same-project error correlation), compute
    log(sum actual / sum predicted), and take its alpha/2 and 1-alpha/2 quantiles.  The band therefore
    narrows with n only as much as the data supports, and it can be asymmetric.
    """

    def __init__(self, alpha=0.05, draws=2000, window=5000, seed=0):
        self.alpha, self.draws, self.window, self.seed = alpha, draws, window, seed

    def fit(self, actual, pred, project):
        a, p = np.asarray(actual, float), np.asarray(pred, float)
        ok = (a > 0) & (p > 0)
        self.a, self.p, self.g = list(a[ok]), list(p[ok]), list(np.asarray(project)[ok])
        self._cache = {}
        return self

    def _bucket(self, n):
        return int(min(max(n, 1), 400)) if n <= 20 else int(round(min(n, 400) / 10) * 10)

    def quantiles(self, n):
        n = self._bucket(n)
        if n in self._cache:
            return self._cache[n]
        a, p, g = np.array(self.a), np.array(self.p), np.array(self.g)
        rng = np.random.default_rng(self.seed + n)
        groups = [np.where(g == x)[0] for x in np.unique(g)]
        sizes = np.array([len(x) for x in groups], float)
        pick = rng.choice(len(groups), self.draws, p=sizes / sizes.sum())
        r = np.empty(self.draws)
        for k, gi in enumerate(pick):
            idx = rng.choice(groups[gi], n, replace=True)
            r[k] = np.log(a[idx].sum() / p[idx].sum())
        q = (float(np.quantile(r, self.alpha / 2)), float(np.quantile(r, 1 - self.alpha / 2)))
        self._cache[n] = q
        return q

    def interval(self, total, n=20):
        lo, hi = self.quantiles(n)
        return total * np.exp(lo), total * np.exp(hi)

    def update(self, actual, pred, project):
        if actual > 0 and pred > 0:
            self.a.append(actual)
            self.p.append(pred)
            self.g.append(project)
            for lst in (self.a, self.p, self.g):
                del lst[:-self.window]
            self._cache = {}
