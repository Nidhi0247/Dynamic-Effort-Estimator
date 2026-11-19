import numpy as np
from .data import CLUSTERS

class SOM:
    def __init__(self, rows=4, cols=4, dim=5, seed=0):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.random((rows * cols, dim))
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], float)
        self.prior = None
        self.qe_history = []

    def fit(self, X, epochs=40, lr0=0.5, sigma0=None):
        X = np.asarray(X, float)
        self.W = X[self.rng.choice(len(X), len(self.W), replace=True)].copy()
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        total, t = epochs * len(X), 0
        for ep in range(epochs):
            for x in X[self.rng.permutation(len(X))]:
                frac = t / total
                lr, sig = lr0 * (1 - frac) + 0.01, sigma0 * (1 - frac) + 0.3
                b = np.argmin(((self.W - x) ** 2).sum(1))
                d2 = ((self.grid - self.grid[b]) ** 2).sum(1)
                h = np.exp(-d2 / (2 * sig ** 2))[:, None]
                self.W += lr * h * (x - self.W)
                t += 1
            self.qe_history.append(self.quantization_error(X))
        return self

    def bmu(self, X):
        X = np.atleast_2d(X)
        return np.argmin(((X[:, None, :] - self.W[None]) ** 2).sum(2), axis=1)

    def quantization_error(self, X):
        return float(np.linalg.norm(X - self.W[self.bmu(X)], axis=1).mean())

    def neuron_labels(self):
        return np.argmax(self.W[:, :3], axis=1)

    def fit_priors(self, X, target):
        b, target = self.bmu(X), np.asarray(target, float)
        g = target.mean()
        self.prior = np.array([target[b == i].mean() if (b == i).any() else g
                               for i in range(len(self.W))])

    def transform(self, X):
        b = self.bmu(X)
        lab = self.neuron_labels()[b]
        return b, lab, self.prior[b]

    def umatrix(self):
        U = np.zeros((self.rows, self.cols))
        W = self.W.reshape(self.rows, self.cols, -1)
        for r in range(self.rows):
            for c in range(self.cols):
                nb = [W[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                      if 0 <= rr < self.rows and 0 <= cc < self.cols]
                U[r, c] = np.mean([np.linalg.norm(W[r, c] - n) for n in nb])
        return U

    @staticmethod
    def label_name(i):
        return CLUSTERS[int(i)]
