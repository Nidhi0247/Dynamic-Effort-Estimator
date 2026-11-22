"""Self-Organising Map (S3).  Trained on learned requirement features; clusters are discovered, not fixed.

Each neuron stores an empirical-Bayes cluster prior of the training target (log hours per SP):
    prior_j = (n_j * mean_j + m * global_mean) / (n_j + m)
with the shrinkage strength m chosen on validation by the pipeline.
"""
import numpy as np


class SOM:
    def __init__(self, rows=4, cols=4, dim=5, seed=0):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.random((rows * cols, dim))
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], float)
        self.prior = None
        self.qe_history = []

    def fit(self, X, epochs=40, lr0=0.5, sigma0=None, max_samples=4000):
        X = np.asarray(X, float)
        if len(X) > max_samples:                               # keep training time bounded on big projects
            X = X[self.rng.choice(len(X), max_samples, replace=False)]
        self.W = X[self.rng.choice(len(X), len(self.W), replace=len(X) < len(self.W))].copy()
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        total, t = epochs * len(X), 0
        for ep in range(epochs):
            for x in X[self.rng.permutation(len(X))]:
                frac = t / total
                lr, sig = lr0 * (1 - frac) + 0.01, sigma0 * (1 - frac) + 0.3
                b = np.argmin(((self.W - x) ** 2).sum(1))
                d2 = ((self.grid - self.grid[b]) ** 2).sum(1)
                self.W += lr * np.exp(-d2 / (2 * sig ** 2))[:, None] * (x - self.W)
                t += 1
            self.qe_history.append(self.quantization_error(X))
        return self

    def bmu(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.argmin(((X[:, None, :] - self.W[None]) ** 2).sum(2), axis=1)

    def quantization_error(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return float(np.linalg.norm(X - self.W[self.bmu(X)], axis=1).mean())

    def topographic_error(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        two = np.argsort(d, 1)[:, :2]
        g = self.grid
        return float((np.abs(g[two[:, 0]] - g[two[:, 1]]).sum(1) > 1).mean())

    def fit_priors(self, X, target, m=10.0):
        b, target = self.bmu(X), np.asarray(target, float)
        self.global_mean = float(target.mean())
        self.counts = np.bincount(b, minlength=len(self.W))
        sums = np.bincount(b, weights=target, minlength=len(self.W))
        self.m = m
        self.prior = (sums + m * self.global_mean) / (self.counts + m)
        return self

    def update_prior(self, x, target):
        """Online: fold a newly resolved issue into its neuron's prior."""
        b = int(self.bmu(x)[0])
        s = self.prior[b] * (self.counts[b] + self.m) - self.m * self.global_mean + target
        self.counts[b] += 1
        self.prior[b] = (s + self.m * self.global_mean) / (self.counts[b] + self.m)

    def transform(self, X):
        b = self.bmu(X)
        return b, self.prior[b]

    def name_neurons(self, feature_names, top=2):
        """Data-driven cluster names: the features whose neuron weight is highest relative to the map mean."""
        z = (self.W - self.W.mean(0)) / (self.W.std(0) + 1e-9)
        self.names = [" + ".join(feature_names[j] for j in np.argsort(-z[i])[:top]) for i in range(len(self.W))]
        return self.names

    def umatrix(self):
        U = np.zeros((self.rows, self.cols))
        W = self.W.reshape(self.rows, self.cols, -1)
        for r in range(self.rows):
            for c in range(self.cols):
                nb = [W[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                      if 0 <= rr < self.rows and 0 <= cc < self.cols]
                U[r, c] = np.mean([np.linalg.norm(W[r, c] - n) for n in nb])
        return U
