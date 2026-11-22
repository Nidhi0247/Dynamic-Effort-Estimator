import itertools
import numpy as np


class ANFIS:
    def __init__(self, n_inputs=3, n_mfs=3, seed=0):
        self.I, self.M = n_inputs, n_mfs
        self.rules = np.array(list(itertools.product(range(n_mfs), repeat=n_inputs)))   # (R, I)
        self.R = len(self.rules)
        self.ii = np.arange(n_inputs)[None, :]
        self.C = np.tile(np.linspace(0, 1, n_mfs), (n_inputs, 1))
        self.S = np.full((n_inputs, n_mfs), 0.5 / n_mfs + 0.1)
        self.theta = np.zeros(self.R * (n_inputs + 1))
        self.history = []

    def _fire(self, X):
        mu = np.exp(-((X[:, :, None] - self.C[None]) ** 2) / (2 * self.S[None] ** 2))   # (N,I,M)
        rm = mu[:, self.ii, self.rules]                                                   # (N,R,I)
        w = rm.prod(2)
        return rm, w, w.sum(1, keepdims=True) + 1e-12

    def _design(self, X):
        rm, w, S = self._fire(X)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        A = ((w / S)[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)
        return A, rm, w, S, Xa

    def predict(self, X):
        return self._design(np.asarray(X, float))[0] @ self.theta

    def _lse(self, A, t, ridge=1e-6):
        return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ t)

    def fit(self, X, t, X_val=None, t_val=None, epochs=60, lr=0.02, ridge=1e-6):
        X, t = np.asarray(X, float), np.asarray(t, float)
        best = (np.inf, None)
        for ep in range(epochs):
            A, rm, w, S, Xa = self._design(X)
            self.theta = self._lse(A, t, ridge)                                  
            y = A @ self.theta
            err = y - t
            rmse = float(np.sqrt((err ** 2).mean()))
            vr = float(np.sqrt(((self.predict(X_val) - t_val) ** 2).mean())) if X_val is not None else rmse
            self.history.append((rmse, vr))
            if vr < best[0]:
                best = (vr, (self.C.copy(), self.S.copy(), self.theta.copy()))
            # backward pass: gradient on premise parameters
            P = self.theta.reshape(self.R, self.I + 1)
            f = Xa @ P.T                                                   # (N,R)
            dw = err[:, None] * (f - y[:, None]) / S                       # dE/dw_r
            g = dw[:, :, None] * w[:, :, None] / (rm + 1e-12)              # dE/dmu_ri
            xc = X[:, None, :] - self.C[self.ii, self.rules][None]         # (N,R,I)
            s_ = self.S[self.ii, self.rules][None]
            gc_r = (g * rm * xc / s_ ** 2).sum(0)
            gs_r = (g * rm * xc ** 2 / s_ ** 3).sum(0)
            gc, gs = np.zeros_like(self.C), np.zeros_like(self.S)
            idx = (np.broadcast_to(self.ii, self.rules.shape), self.rules)
            np.add.at(gc, idx, gc_r)
            np.add.at(gs, idx, gs_r)
            step = lr * (1 - ep / epochs)
            self.C -= step * gc / (np.linalg.norm(gc) + 1e-9)
            self.S = np.clip(self.S - step * gs / (np.linalg.norm(gs) + 1e-9), 0.05, 1.0)
        self.C, self.S, self.theta = best[1]
        self.init_online(X, ridge=max(ridge, 1e-6))
        return self

    # ---------------------------------------------------------------- online learning (RLS)
    def init_online(self, X, ridge=1e-6, forget=1.0):
        """Prepare recursive least squares on the consequents for the current premise parameters."""
        A = self._design(np.asarray(X, float))[0]
        self.P = np.linalg.inv(A.T @ A + ridge * np.eye(A.shape[1]))
        self.forget = forget

    def partial_fit(self, X, t):
        """Update the consequent parameters with new (input, target) pairs, one at a time (RLS)."""
        A = self._design(np.atleast_2d(np.asarray(X, float)))[0]
        lam = getattr(self, "forget", 1.0)
        for a, y in zip(A, np.atleast_1d(np.asarray(t, float))):
            Pa = self.P @ a
            k = Pa / (lam + a @ Pa)
            self.theta = self.theta + k * (y - a @ self.theta)
            self.P = (self.P - np.outer(k, Pa)) / lam
        return self
