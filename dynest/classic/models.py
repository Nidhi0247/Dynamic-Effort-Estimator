"""Estimators for the classic project-level benchmark (E8).

Every estimator shares one protocol, so differences come from the model alone:
  * the preprocessing (log transform of skewed predictors, one-hot categories, standardisation) is fitted on
    the training fold only;
  * the target is log effort, or log effort per size unit (the "density" hyper-parameter);
  * hyper-parameters come from random search with the same budget for every tuned model, scored by inner
    k-fold cross-validation MAE inside the training fold;
  * the back-transform is the MAE shift (weighted median of the log ratio) learnt on the inner out-of-fold
    predictions;
  * the 95% interval is split-conformal on the same inner out-of-fold log residuals (one recipe for all models).

The proposed estimator, NeuroFuzzy, is the DynEst core on tabular project data:
    SOM over all predictors -> shrunk cluster prior (out-of-fold) -> ANFIS on [prior, top-ranked predictors]
    -> out-of-fold complexity score c_hat -> learned Mamdani (Wang-Mendel + GA-tuned strong partitions) on
    [c_hat, next-ranked predictors] -> log effort.
The predictor ranking (|Spearman rho| with the target) is learnt on the training fold.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..anfis import ANFIS
from ..calibration import mae_shift
from ..learned_mamdani import LearnedMamdani
from ..som import SOM

COCOMO81_MODES = {"organic": (3.2, 1.05), "semidetached": (3.0, 1.12), "embedded": (2.8, 1.20)}   # Boehm 1981


# ====================================================================== preprocessing
class Prep:
    def __init__(self, cats=()):
        self.cats = list(cats)

    def fit(self, X):
        self.cat = [c for c in self.cats if c in X]
        self.num = [c for c in X if c not in self.cat]
        N = X[self.num].astype(float)
        self.logc = [c for c in self.num if (N[c] >= 0).all() and N[c].nunique() > 2
                     and stats.skew(N[c]) > 1.0]
        self.levels = {c: sorted(X[c].astype(str).unique()) for c in self.cat}
        M = self._raw(X)
        self.mu, self.sd = M.mean(0), M.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def _raw(self, X):
        N = X[self.num].astype(float).copy()
        for c in self.logc:
            N[c] = np.log1p(N[c])
        parts, names = [N.values], list(self.num)
        for c in self.cat:
            v = X[c].astype(str).values
            for lv in self.levels[c]:
                parts.append((v == lv).astype(float)[:, None])
                names.append(f"{c}={lv}")
        self.names = names
        return np.hstack(parts)

    def transform(self, X):
        return (self._raw(X) - self.mu) / self.sd


def _folds(n, k, rng):
    k = max(2, min(k, n))
    return np.array_split(rng.permutation(n), k)


def _logsize(X, ds):
    return np.log(X[ds.size].astype(float).values)


# ====================================================================== cores (fit -> predict log effort)
class Core:
    def __init__(self, ds, seed=0, **hp):
        self.ds, self.seed, self.hp = ds, seed, hp

    def _target(self, X, y):
        self.offset = self.hp.get("density", False)
        return np.log(y) - (_logsize(X, self.ds) if self.offset else 0.0)

    def _untarget(self, X, z):
        return z + (_logsize(X, self.ds) if self.offset else 0.0)


class MedianCore(Core):
    def fit(self, X, y):
        self.m = float(np.median(np.log(y)))
        return self

    def predict_log(self, X):
        return np.full(len(X), self.m)


class SizeRegCore(Core):
    """Function-point (or KLOC) regression: log effort = a + b log size (OLS)."""
    def fit(self, X, y):
        self.b, self.a = np.polyfit(_logsize(X, self.ds), np.log(y), 1)
        return self

    def predict_log(self, X):
        return self.a + self.b * _logsize(X, self.ds)


class CocomoNominalCore(Core):
    """Intermediate COCOMO-81 with Boehm's published mode constants and multipliers (no fitting)."""
    def fit(self, X, y):
        return self

    def predict_log(self, X):
        mode = X[self._mode_col()].astype(str).str.lower().values
        ab = np.array([COCOMO81_MODES.get(m, COCOMO81_MODES["semidetached"]) for m in mode])
        em = np.log(X[self.ds.em].astype(float).values).sum(1)
        return np.log(ab[:, 0]) + ab[:, 1] * _logsize(X, self.ds) + em

    def _mode_col(self):
        return "dev_mode" if "dev_mode" in self.ds.X else "mode"


class CocomoLocalCore(Core):
    """Boehm's local calibration: published multipliers fixed, a and b refitted (log-linear OLS)."""
    def fit(self, X, y):
        em = np.log(X[self.ds.em].astype(float).values).sum(1)
        self.b, self.a = np.polyfit(_logsize(X, self.ds), np.log(y) - em, 1)
        return self

    def predict_log(self, X):
        em = np.log(X[self.ds.em].astype(float).values).sum(1)
        return self.a + self.b * _logsize(X, self.ds) + em


class CocomoRefitCore(Core):
    """COCOMO-81 functional form with every multiplier's exponent re-estimated (ridge in log space)."""
    def _D(self, X):
        return np.column_stack([_logsize(X, self.ds), np.log(X[self.ds.em].astype(float).values)])

    def fit(self, X, y):
        D = self._D(X)
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-9
        self.m = Ridge(alpha=self.hp.get("alpha", 1.0)).fit((D - self.mu) / self.sd, np.log(y))
        return self

    def predict_log(self, X):
        return self.m.predict((self._D(X) - self.mu) / self.sd)


class SklearnCore(Core):
    def fit(self, X, y):
        self.p = Prep(self.ds.cats).fit(X)
        self.m = self.build().fit(self.p.transform(X), self._target(X, y))
        return self

    def predict_log(self, X):
        return self._untarget(X, self.m.predict(self.p.transform(X)))


class RidgeCore(SklearnCore):
    def build(self):
        return Ridge(alpha=self.hp["alpha"])


class CARTCore(SklearnCore):
    def build(self):
        return DecisionTreeRegressor(max_depth=self.hp["depth"], min_samples_leaf=self.hp["leaf"],
                                     random_state=self.seed)


class SVRCore(SklearnCore):
    def build(self):
        return SVR(C=self.hp["C"], gamma=self.hp["gamma"], epsilon=self.hp["eps"])


class RFCore(SklearnCore):
    def build(self):
        return RandomForestRegressor(n_estimators=300, max_features=self.hp["mf"], min_samples_leaf=self.hp["leaf"],
                                     random_state=self.seed, n_jobs=1)


class GBMCore(SklearnCore):
    def build(self):
        return GradientBoostingRegressor(n_estimators=self.hp["n"], learning_rate=self.hp["lr"],
                                         max_depth=self.hp["depth"], subsample=self.hp["sub"],
                                         random_state=self.seed)


class MLPCore(SklearnCore):
    def build(self):
        return MLPRegressor(hidden_layer_sizes=self.hp["hidden"], alpha=self.hp["alpha"],
                            learning_rate_init=self.hp["lr"], max_iter=3000, early_stopping=False,
                            random_state=self.seed)


class ShallowTreeCore(SklearnCore):
    def build(self):
        return DecisionTreeRegressor(max_depth=self.hp["depth"], min_samples_leaf=self.hp["leaf"], random_state=self.seed)


class EBMCore(SklearnCore):
    def build(self):
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(learning_rate=self.hp["lr"], interactions=self.hp["inter"],
                                            max_leaves=self.hp["leaves"], min_samples_leaf=self.hp["leaf"],
                                            outer_bags=4, n_jobs=1, random_state=self.seed)


class ABECore(Core):
    """Analogy-based estimation: k nearest projects (Euclidean, standardised predictors), tuned adaptation."""
    def fit(self, X, y):
        self.p = Prep(self.ds.cats).fit(X)
        self.Z, self.y, self.s = self.p.transform(X), np.asarray(y, float), X[self.ds.size].astype(float).values
        return self

    def predict_log(self, X):
        Z, s = self.p.transform(X), X[self.ds.size].astype(float).values
        d = np.sqrt(((Z[:, None, :] - self.Z[None]) ** 2).sum(2))
        k = min(self.hp["k"], len(self.y))
        nn = np.argsort(d, 1)[:, :k]
        Y, D = self.y[nn], np.take_along_axis(d, nn, 1)
        a = self.hp["adapt"]
        if a == "mean":
            e = Y.mean(1)
        elif a == "median":
            e = np.median(Y, 1)
        elif a == "idw":
            w = 1 / (D + 1e-6)
            e = (w * Y).sum(1) / w.sum(1)
        else:                                                        # size-ratio adaptation
            e = (Y * (s[:, None] / self.s[nn])).mean(1)
        return np.log(np.maximum(e, 1e-9))


class NeuroFuzzyCore(Core):
    """SOM -> ANFIS -> learned Mamdani (the DynEst core, see module docstring)."""

    def fit(self, X, y):
        h, rng = self.hp, np.random.default_rng(self.seed)
        self.p = Prep(self.ds.cats).fit(X)
        Z = self.p.transform(X)
        t = self._target(X, y)
        n = len(t)
        rho = np.array([abs(stats.spearmanr(Z[:, j], t).statistic) if Z[:, j].std() > 0 else 0.0
                        for j in range(Z.shape[1])])
        self.order = list(np.argsort(-np.nan_to_num(rho)))
        folds = _folds(n, 5, rng)

        # S3: SOM (unsupervised) + shrunk prior; the prior fed to ANFIS is out-of-fold
        if h["use_som"]:
            g = h["grid"]
            self.som = SOM(g, g, Z.shape[1], seed=self.seed).fit(Z, epochs=30)
            self.som.fit_priors(Z, t, m=h["m"])
            b = self.som.bmu(Z)
            prior = np.zeros(n)
            for f in folds:
                o = np.ones(n, bool)
                o[f] = False
                gm = t[o].mean()
                cnt = np.bincount(b[o], minlength=len(self.som.W))
                sm = np.bincount(b[o], weights=t[o], minlength=len(self.som.W))
                prior[f] = ((sm + h["m"] * gm) / (cnt + h["m"]))[b[f]]
            self.som.name_neurons(self.p.names)
        k_a = h["n_anfis"] - (1 if h["use_som"] else 0)
        self.a_idx = self.order[:k_a]
        self.m_idx = self.order[k_a:k_a + h["n_mam_extra"]]
        A = self._anfis_in(Z, prior if h["use_som"] else None)

        # S4: ANFIS -> c_hat (out-of-fold for the Mamdani stage)
        self.t_mu, self.t_sd = float(t.mean()), float(t.std() + 1e-9)
        ts = (t - self.t_mu) / self.t_sd
        if h["use_anfis"]:
            self.a_lo, self.a_hi = A.min(0), A.max(0)
            self.a_hi = np.where(self.a_hi - self.a_lo < 1e-9, self.a_lo + 1, self.a_hi)
            S = self._s01(A)
            c_hat = np.zeros(n)
            for f in folds:
                o = np.ones(n, bool)
                o[f] = False
                a = ANFIS(S.shape[1], h["mfs"], seed=self.seed).fit(S[o], ts[o], epochs=h["epochs"], ridge=h["ridge"])
                c_hat[f] = a.predict(S[f])
            self.anfis = ANFIS(S.shape[1], h["mfs"], seed=self.seed).fit(S, ts, epochs=h["epochs"], ridge=h["ridge"])
            c_hat = c_hat * self.t_sd + self.t_mu
        else:
            c_hat = A[:, 0]

        # S5: learned Mamdani on [c_hat, next-ranked predictors]; transparent: [SOM prior, top-2 predictors]
        if h.get("transparent"):
            self.m_idx = self.order[:2]
            M = np.column_stack([prior] + [Z[:, j] for j in self.m_idx])
            names = ["similar-project prior"] + [self.p.names[j] for j in self.m_idx]
            terms = [h["terms0"]] + [3] * len(self.m_idx)          # Kitchenham has a single predictor
        if h["use_mamdani"] and not h.get("transparent"):
            M = np.column_stack([c_hat] + [Z[:, j] for j in self.m_idx])
            names = ["c_hat"] + [self.p.names[j] for j in self.m_idx]
            terms = [h["terms0"]] + [3] * len(self.m_idx)
        if h["use_mamdani"]:
            perm = rng.permutation(n)                              # GA hold-out is the tail: shuffle first
            self.fis = LearnedMamdani(names, terms, h["n_out"], h["min_support"], seed=self.seed)
            self.fis.fit(M[perm], t[perm], tune=True, gens=h["gens"])
        else:
            self.lin = np.polyfit(c_hat, t, 1) if not h["use_anfis"] else None
        return self

    def _anfis_in(self, Z, prior):
        cols = ([prior] if prior is not None else []) + [Z[:, j] for j in self.a_idx]
        return np.column_stack(cols)

    def _s01(self, A):
        return np.clip((A - self.a_lo) / (self.a_hi - self.a_lo), -0.25, 1.25)

    def _c_hat(self, Z):
        h = self.hp
        prior = self.som.transform(Z)[1] if h["use_som"] else None
        A = self._anfis_in(Z, prior)
        if h["use_anfis"]:
            return self.anfis.predict(self._s01(A)) * self.t_sd + self.t_mu
        return A[:, 0]

    def predict_log(self, X):
        Z = self.p.transform(X)
        if self.hp.get("transparent"):
            prior = self.som.transform(Z)[1]
            return self._untarget(X, self.fis.predict(np.column_stack([prior] + [Z[:, j] for j in self.m_idx])))
        c = self._c_hat(Z)
        if self.hp["use_mamdani"]:
            z = self.fis.predict(np.column_stack([c] + [Z[:, j] for j in self.m_idx]))
        elif self.lin is not None:
            z = np.polyval(self.lin, c)
        else:
            z = c
        return self._untarget(X, z)


# ====================================================================== hyper-parameter spaces
def _lu(rng, lo, hi):
    return float(10 ** rng.uniform(lo, hi))


def _ch(rng, opts):
    return opts[rng.integers(len(opts))]


NF_DEFAULT = dict(use_som=True, use_anfis=True, use_mamdani=True, grid=3, m=3.0, n_anfis=3, mfs=2, ridge=0.1,
                  epochs=30, n_mam_extra=1, terms0=5, n_out=7, min_support=1, gens=15, density=True)


def nf_space(rng, n):
    g_max = 4 if n >= 60 else 3
    return dict(NF_DEFAULT, grid=int(rng.integers(2, g_max + 1)), m=_ch(rng, [1.0, 3.0, 10.0]),
                n_anfis=_ch(rng, [2, 3]), mfs=_ch(rng, [2, 3]) if n >= 40 else 2, ridge=_lu(rng, -3, 1),
                n_mam_extra=_ch(rng, [0, 1, 2]), terms0=_ch(rng, [3, 5, 7]), n_out=_ch(rng, [5, 7, 9]),
                min_support=_ch(rng, [1, 2, 3]), density=_ch(rng, [True, False]))


SPACES = {
    "Ridge": lambda r, n: dict(alpha=_lu(r, -3, 2), density=_ch(r, [True, False])),
    "CART": lambda r, n: dict(depth=_ch(r, [2, 3, 4, 5, None]), leaf=_ch(r, [1, 2, 3, 5]), density=_ch(r, [True, False])),
    "SVR": lambda r, n: dict(C=_lu(r, -1, 2.5), gamma=_lu(r, -3, 0), eps=float(r.uniform(0.01, 0.3)),
                             density=_ch(r, [True, False])),
    "RF": lambda r, n: dict(mf=_ch(r, [0.33, 0.66, 1.0]), leaf=_ch(r, [1, 2, 3, 5]), density=_ch(r, [True, False])),
    "GBM": lambda r, n: dict(n=_ch(r, [100, 300]), lr=_ch(r, [0.03, 0.1]), depth=_ch(r, [2, 3]),
                             sub=_ch(r, [0.7, 1.0]), density=_ch(r, [True, False])),
    "MLP": lambda r, n: dict(hidden=_ch(r, [(8,), (16,), (32,), (16, 8)]), alpha=_lu(r, -4, 0.5),
                             lr=_ch(r, [1e-3, 1e-2]), density=_ch(r, [True, False])),
    "ShallowTree": lambda r, n: dict(depth=_ch(r, [1, 2, 3]), leaf=_ch(r, [2, 3, 5, 8]), density=_ch(r, [True, False])),
    "EBM": lambda r, n: dict(lr=_lu(r, -2.5, -1.3), inter=_ch(r, [0, 3, 5]), leaves=_ch(r, [2, 3]),
                             leaf=_ch(r, [2, 3, 5]), density=_ch(r, [True, False])),
    "ABE": lambda r, n: dict(k=_ch(r, [1, 2, 3, 4, 5]), adapt=_ch(r, ["mean", "median", "idw", "size"])),
    "COCOMO refit": lambda r, n: dict(alpha=_lu(r, -2, 2)),
    "NeuroFuzzy": nf_space,
}


# ====================================================================== estimator wrapper (tuning, shift, conformal)
class Estimator:
    def __init__(self, name, core, space=None, fixed=None, fit_shift=True, alpha=0.05):
        self.name, self.core, self.space, self.fixed = name, core, space, fixed or {}
        self.fit_shift, self.alpha = fit_shift, alpha

    def _oof(self, X, y, ds, hp, folds, seed):
        z = np.zeros(len(y))
        for f in folds:
            o = np.ones(len(y), bool)
            o[f] = False
            z[f] = self.core(ds, seed, **hp).fit(X.iloc[o], y[o]).predict_log(X.iloc[f])
        return z

    def fit(self, X, y, ds, n_trials=10, inner=3, seed=0):
        X, y = X.reset_index(drop=True), np.asarray(y, float)
        rng = np.random.default_rng(seed)
        folds = _folds(len(y), inner, rng)
        if self.space is None:
            cands = [dict(self.fixed)]
        else:
            first = dict(NF_DEFAULT) if self.space is nf_space else self.space(np.random.default_rng(seed + 999), len(y))
            cands = [dict(first, **self.fixed)] + [dict(self.space(rng, len(y)), **self.fixed)
                                                   for _ in range(n_trials - 1)]
        best = None
        for hp in cands:
            try:
                z = self._oof(X, y, ds, hp, folds, seed)
            except Exception:                                       # e.g. a degenerate config on a tiny fold
                continue
            if not np.isfinite(z).all():
                continue
            s = mae_shift(y, np.exp(z)) if self.fit_shift else 0.0
            mae = float(np.abs(y - np.exp(z + s)).mean())
            if best is None or mae < best[0]:
                best = (mae, hp, z, s)
        if best is None:
            raise RuntimeError(f"{self.name}: no configuration could be fitted")
        self.inner_mae, self.hp, z, self.shift = best
        r = np.log(y) - (z + self.shift)
        n = len(r)
        lvl = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(np.abs(r), lvl, method="higher"))
        self.model = self.core(ds, seed, **self.hp).fit(X, y)
        return self

    def predict(self, X):
        z = self.model.predict_log(X.reset_index(drop=True)) + self.shift
        return np.exp(z), np.exp(z - self.q), np.exp(z + self.q)


def estimators(ds):
    """The model set applicable to dataset ds (COCOMO variants only where the multipliers exist)."""
    size_name = "FP regression (log-log)" if ds.kind == "FP" else "KLOC power law (log-log)"
    E = [Estimator("NeuroFuzzy (proposed)", NeuroFuzzyCore, SPACES["NeuroFuzzy"]),
         Estimator("NeuroFuzzy-T (proposed, transparent)", NeuroFuzzyCore, SPACES["NeuroFuzzy"],
                   fixed=dict(transparent=True, use_som=True, use_anfis=False, use_mamdani=True)),
         Estimator("Median", MedianCore),
         Estimator(size_name, SizeRegCore)]
    if ds.em:
        E += [Estimator("COCOMO-81 nominal", CocomoNominalCore, fit_shift=False),
              Estimator("COCOMO-81 local calib.", CocomoLocalCore),
              Estimator("COCOMO-81 refit", CocomoRefitCore, SPACES["COCOMO refit"])]
    E += [Estimator("ABE (analogy)", ABECore, SPACES["ABE"]),
          Estimator("Ridge (log)", RidgeCore, SPACES["Ridge"]),
          Estimator("CART", CARTCore, SPACES["CART"]),
          Estimator("Shallow tree (depth<=3)", ShallowTreeCore, SPACES["ShallowTree"]),
          Estimator("EBM (glass-box)", EBMCore, SPACES["EBM"]),
          Estimator("SVR", SVRCore, SPACES["SVR"]),
          Estimator("Random forest", RFCore, SPACES["RF"]),
          Estimator("GBM", GBMCore, SPACES["GBM"]),
          Estimator("MLP", MLPCore, SPACES["MLP"])]
    return E


ABLATIONS = {"NF -SOM": dict(use_som=False), "NF -ANFIS": dict(use_anfis=False),
             "NF -Mamdani": dict(use_mamdani=False), "ANFIS only": dict(use_som=False, use_mamdani=False)}


def ablations(best_hp):
    """Ablations reuse the proposed model's tuned configuration with one component switched off."""
    return [Estimator(k, NeuroFuzzyCore, fixed={**best_hp, **v}) for k, v in ABLATIONS.items()]
