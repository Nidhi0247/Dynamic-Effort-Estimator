"""Trained baselines (B0-B7).  Same windows, same target (log hours per story point), same features,
same random-search budget, model selection on the validation window only.

Every model implements  fit(train_df, val_df) -> self  and  predict(df) -> hours.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import mae_shift

SP, Y = "sp", "hours"


def _density(df):
    return np.log(df[Y].values / df[SP].values)


class _Base:
    name = "base"

    def __init__(self, features, n_trials=40, seed=0):
        self.features, self.n_trials, self.seed = list(features), n_trials, seed

    def _X(self, df):
        return df[self.features].astype(float).values

    def predict(self, df):
        return df[SP].values * np.exp(self._predict_log(df) + self.smear)

    def _set_smear(self, val):
        self.smear = mae_shift(val[Y].values, val[SP].values * np.exp(self._predict_log(val)))


class MedianDensity(_Base):
    """B0: median hours per story point of the training window."""
    name = "B0 Median density"

    def fit(self, tr, va):
        self.m = float(np.median(_density(tr)))
        self.smear = 0.0
        self._set_smear(va)
        return self

    def _predict_log(self, df):
        return np.full(len(df), self.m)


class SPVelocity(_Base):
    """B1: per-project velocity (hours per SP), recent window weighted, pooled fallback."""
    name = "B1 SP velocity"

    def fit(self, tr, va, recent=200):
        d = pd.DataFrame(dict(p=tr["project_id"].values, y=_density(tr)))
        self.global_ = float(d.y.median())
        self.by_p = d.groupby("p").y.apply(lambda s: float(s.tail(recent).median())).to_dict()
        self.smear = 0.0
        self._set_smear(va)
        return self

    def _predict_log(self, df):
        return np.array([self.by_p.get(p, self.global_) for p in df["project_id"]])


class _Tuned(_Base):
    """Random search over `space()`; best validation MAE (hours) wins; refit on train."""

    def space(self, rng):
        raise NotImplementedError

    def build(self, **hp):
        raise NotImplementedError

    def fit(self, tr, va):
        rng = np.random.default_rng(self.seed)
        Xtr, ytr = self._X(tr), _density(tr)
        best = (np.inf, None)
        for _ in range(self.n_trials):
            hp = self.space(rng)
            m = self.build(**hp).fit(Xtr, ytr)
            self.model, self.smear = m, 0.0
            self._set_smear(va)
            mae = np.abs(self.predict(va) - va[Y].values).mean()
            if mae < best[0]:
                best = (mae, (m, self.smear, hp))
        self.model, self.smear, self.hp = best[1]
        return self

    def _predict_log(self, df):
        return self.model.predict(self._X(df))


class RidgeB(_Tuned):
    name = "B2 Ridge"

    def space(self, rng):
        return dict(alpha=float(10 ** rng.uniform(-3, 3)))

    def build(self, alpha):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


class RandomForestB(_Tuned):
    name = "B3 Random forest"

    def space(self, rng):
        return dict(max_depth=int(rng.choice([4, 6, 8, 12, 16])), min_samples_leaf=int(rng.choice([2, 5, 10, 20])),
                    max_features=float(rng.choice([0.3, 0.5, 0.8, 1.0])))

    def build(self, **hp):
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=self.seed, **hp)


class GBMB(_Tuned):
    name = "B4 Gradient boosting"

    def space(self, rng):
        return dict(learning_rate=float(10 ** rng.uniform(-2.3, -0.7)), max_leaf_nodes=int(rng.choice([8, 15, 31])),
                    l2_regularization=float(10 ** rng.uniform(-3, 1)), min_samples_leaf=int(rng.choice([10, 20, 40])))

    def build(self, **hp):
        return HistGradientBoostingRegressor(max_iter=500, early_stopping=True, validation_fraction=0.15,
                                             random_state=self.seed, **hp)


class MLPB(_Tuned):
    name = "B5 MLP"

    def space(self, rng):
        h = int(rng.choice([16, 32, 64]))
        return dict(hidden=(h,) if rng.random() < 0.5 else (h, h // 2), alpha=float(10 ** rng.uniform(-4, 0)),
                    lr=float(10 ** rng.uniform(-3.5, -2)))

    def build(self, hidden, alpha, lr):
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                                                            learning_rate_init=lr, early_stopping=True,
                                                            max_iter=500, random_state=self.seed))


class CocomoLocal(_Base):
    """B6: COCOMO II form E = A * SP^B * prod_j EM_j(level_j), locally calibrated.

    Each cost driver is binned into 5 levels (VL..VH) at training quintiles; log A, B and every
    log EM_j(level) (nominal = reference level) are fitted by ridge-regularised least squares.
    """
    name = "B6 COCOMO II (local)"

    def __init__(self, drivers, ridge=1.0, **kw):
        super().__init__(drivers, **kw)
        self.ridge = ridge

    def _design(self, df):
        cols = [np.ones(len(df)), np.log(df[SP].values)]
        for f, edges in zip(self.features, self.edges):
            lvl = np.digitize(df[f].astype(float).values, edges)                # 0..4, 2 = nominal
            for L in (0, 1, 3, 4):
                cols.append((lvl == L).astype(float))
        return np.column_stack(cols)

    def fit(self, tr, va):
        self.edges = [np.quantile(tr[f].astype(float), [0.2, 0.4, 0.6, 0.8]) for f in self.features]
        X, y = self._design(tr), np.log(tr[Y].values)
        R = self.ridge * np.eye(X.shape[1])
        R[0, 0] = R[1, 1] = 0
        self.beta = np.linalg.solve(X.T @ X + R, X.T @ y)
        self.smear = 0.0
        self.smear = mae_shift(va[Y].values, np.exp(self._design(va) @ self.beta))
        return self

    def predict(self, df):
        return np.exp(self._design(df) @ self.beta + self.smear)

    def effort_multipliers(self):
        out = {}
        for k, f in enumerate(self.features):
            b = self.beta[2 + 4 * k: 6 + 4 * k]
            out[f] = dict(VL=np.exp(b[0]), L=np.exp(b[1]), N=1.0, H=np.exp(b[2]), VH=np.exp(b[3]))
        return dict(A=float(np.exp(self.beta[0])), B=float(self.beta[1]), EM=out)


class CAEA(_Base):
    """B7: Constructive Agile Estimation Algorithm.

    Adjusted story points ASP = SP * (1 + sum_f theta_f(level_f)), effort = velocity * ASP, with
    each CAEA factor rated low / medium / high at training terciles of the attribute mapped to it.

    mode="fit":   velocity and the factor values theta are fitted by NNLS on training data (proxy).
    mode="paper": theta comes from configs/caea_constants.yaml (transcribed from Bhalerao & Ingle 2009 /
                  Alshammari 2022); only the velocity is fitted.
    """

    def __init__(self, factor_map, mode="fit", constants_file=None, **kw):
        super().__init__(list(factor_map.values()), **kw)
        self.factor_map, self.mode = dict(factor_map), mode
        self.constants_file = constants_file
        self.name = f"B7 CAEA ({mode})"

    def _levels(self, df):
        return {fac: np.digitize(df[col].astype(float).values, self.edges[fac])
                for fac, col in self.factor_map.items()}

    def fit(self, tr, va):
        self.edges = {fac: np.quantile(tr[col].astype(float), [1 / 3, 2 / 3]) for fac, col in self.factor_map.items()}
        lv = self._levels(tr)
        sp, y = tr[SP].values.astype(float), tr[Y].values.astype(float)
        if self.mode == "paper":
            consts = yaml.safe_load(Path(self.constants_file).read_text())
            vals = consts["factor_values"]                       # {factor: {low:, medium:, high:}}
            if any(v is None for f in vals.values() for v in f.values()):
                raise ValueError("CAEA paper constants not transcribed yet (see configs/caea_constants.yaml)")
            names = ["low", "medium", "high"]
            self.theta = {fac: np.array([vals[fac][n] for n in names], float) / consts["normaliser"]
                          for fac in self.factor_map}
            asp = sp * (1 + sum(self.theta[f][lv[f]] for f in self.factor_map))
            self.v = float(np.median(y / asp))
        else:
            cols = [sp]
            for fac in self.factor_map:
                for L in range(3):
                    cols.append(sp * (lv[fac] == L))
            coef, _ = nnls(np.column_stack(cols) / y[:, None], np.ones(len(y)))   # relative-error NNLS
            self.v = float(coef[0])
            k = 1
            self.theta = {}
            for fac in self.factor_map:
                self.theta[fac] = coef[k:k + 3] / max(self.v, 1e-9)
                k += 3
        self.smear = 0.0
        self.smear = mae_shift(va[Y].values, self.predict(va))
        return self

    def predict(self, df):
        lv = self._levels(df)
        asp = df[SP].values.astype(float) * (1 + sum(self.theta[f][lv[f]] for f in self.factor_map))
        return np.maximum(self.v * asp * np.exp(self.smear), 1e-3)


def all_baselines(features, cocomo_drivers, caea_map, n_trials=40, seed=0, caea_constants=None):
    models = [MedianDensity(features), SPVelocity(features),
              RidgeB(features, n_trials, seed), RandomForestB(features, max(n_trials // 2, 1), seed),
              GBMB(features, n_trials, seed), MLPB(features, max(n_trials // 2, 1), seed),
              CocomoLocal(cocomo_drivers, seed=seed), CAEA(caea_map, mode="fit", seed=seed)]
    if caea_constants and Path(caea_constants).exists():
        models.append(CAEA(caea_map, mode="paper", constants_file=caea_constants, seed=seed))
    return models
