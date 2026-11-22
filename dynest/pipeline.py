"""DynEst-R estimation pipeline (S1-S6), fully trained on real history.

S1 FeatureBuilder (measured as-of features + text PCA)      fitted on the training window
S2 AuxModels  -> clarity_hat, integration_hat, profile_hat     trained on training-window labels
S3 SOM on [text, profile_hat, integration_hat] -> cluster prior (shrunk mean log hours/SP)
S4 ANFIS on (chosen inputs incl. the SOM prior) -> complexity score c_hat  (out-of-fold for S5)
S5 Learned Mamdani on (c_hat, clarity_hat, familiarity) -> log hours/SP     (Wang-Mendel + GA)
S6 Conformal intervals (requirement and project level)        calibrated on the validation window
Online: update() folds newly resolved issues into ANFIS (RLS), SOM priors and conformal scores.

Ablation switches (cfg): use_som, use_anfis, mamdani in {"learned", "expert", "none"}, use_aux.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .anfis import ANFIS
from .aux_models import AuxModels
from .calibration import mae_shift
from .conformal import ConformalIntervals, ProjectConformal
from .features import FeatureBuilder, MEASURED
from .learned_mamdani import LearnedMamdani
from .mamdani import MamdaniFIS
from .som import SOM

DEFAULT = dict(som_grid=4, som_m=10.0, som_epochs=20, anfis_inputs=("integration_hat", "som_prior", "log_sp"),
               anfis_mfs=3, anfis_epochs=40, mamdani_inputs=("c_hat", "clarity_hat", "familiarity_log"),
               mamdani_terms=(3, 3, 3), mamdani_out=7, min_support=3, mamdani_tune=True, mamdani_gens=20,
               use_som=True, use_anfis=True, mamdani="learned", use_aux=True, text_backend="minilm",
               alpha=0.05, oof_folds=5, seed=0)

SPACE = dict(som_grid=[3, 4, 5], som_m=[3.0, 10.0, 30.0], anfis_mfs=[2, 3],
             anfis_inputs=[("integration_hat", "som_prior", "log_sp"), ("integration_hat", "som_prior", "txt0"),
                           ("som_prior", "log_sp", "txt0")],
             mamdani_terms=[(3, 3, 3), (4, 3, 3), (5, 3, 3), (4, 4, 3)], mamdani_out=[5, 7, 9], min_support=[2, 3, 5])


def log_density(df):
    return np.log(df.effort_h.values / df.sp.values)


class _Scaler01:
    def fit(self, X):
        self.lo, self.hi = np.quantile(X, 0.01, 0), np.quantile(X, 0.99, 0)
        self.hi = np.where(self.hi - self.lo < 1e-9, self.lo + 1, self.hi)
        return self

    def __call__(self, X):
        return np.clip((X - self.lo) / (self.hi - self.lo), 0, 1)


class DynEstR:
    def __init__(self, cfg=None):
        self.cfg = {**DEFAULT, **(cfg or {})}

    # ================================================================== base (S1, S2)
    def fit_base(self, train, components):
        c = self.cfg
        self.fb = FeatureBuilder(c["text_backend"], 16, c["seed"]).fit(train)
        Xtr = self.fb.transform(train)
        if c["use_aux"]:
            self.aux = AuxModels(seed=c["seed"]).fit(Xtr, train, components)
        else:
            self.aux = None
        return self

    def _base(self, df):
        X = self.fb.transform(df)
        if self.aux is not None:
            A = self.aux.transform(X)
        else:                                                    # A5: raw counts stand in for S2
            A = pd.DataFrame(dict(clarity_hat=X.desc_words_log, integration_hat=np.log1p(X.n_links_t + X.n_components_t)),
                             index=X.index)
        return pd.concat([X, A], axis=1)

    # ================================================================== head (S3-S6)
    def fit(self, train, val, components, base_ready=False):
        train = train.sort_values("resolved_at").reset_index(drop=True)
        val = val.sort_values("resolved_at").reset_index(drop=True)
        if not base_ready:
            self.fit_base(train, components)
        c = self.cfg
        B_tr, B_va = self._base(train), self._base(val)
        y_tr, y_va = log_density(train), log_density(val)
        self.y_mu, self.y_sd = float(y_tr.mean()), float(y_tr.std() + 1e-9)
        self.prof_cols = self.aux.profile_cols if self.aux is not None else []

        # S3 SOM
        self.som_cols = self.fb.text_cols + self.prof_cols + ["integration_hat"]
        self.som_scaler = StandardScaler().fit(B_tr[self.som_cols].values)
        Ztr = self.som_scaler.transform(B_tr[self.som_cols].values)
        g = c["som_grid"]
        self.som = SOM(g, g, Ztr.shape[1], seed=c["seed"]).fit(Ztr, epochs=c["som_epochs"])
        self.som.fit_priors(Ztr, y_tr, m=c["som_m"])
        self.som.name_neurons(self.som_cols)
        for B in (B_tr, B_va):
            self._add_som(B)

        # S4 ANFIS (out-of-fold on train for the Mamdani stage)
        self.anfis_cols = [x for x in c["anfis_inputs"] if c["use_som"] or x != "som_prior"]
        if c["use_anfis"]:
            self.ascale = _Scaler01().fit(B_tr[self.anfis_cols].values)
            Xa_tr, Xa_va = self.ascale(B_tr[self.anfis_cols].values), self.ascale(B_va[self.anfis_cols].values)
            t_tr, t_va = (y_tr - self.y_mu) / self.y_sd, (y_va - self.y_mu) / self.y_sd
            oof = np.zeros(len(train))
            folds = np.array_split(np.arange(len(train)), c["oof_folds"])
            for f in folds:
                m = np.ones(len(train), bool)
                m[f] = False
                a = ANFIS(len(self.anfis_cols), c["anfis_mfs"], seed=c["seed"]).fit(Xa_tr[m], t_tr[m],
                                                                                   epochs=c["anfis_epochs"])
                oof[f] = a.predict(Xa_tr[f])
            self.anfis = ANFIS(len(self.anfis_cols), c["anfis_mfs"], seed=c["seed"]).fit(Xa_tr, t_tr, Xa_va, t_va,
                                                                                       epochs=c["anfis_epochs"])
            B_tr["c_hat"] = oof * self.y_sd + self.y_mu
            B_va["c_hat"] = self._anfis(B_va)
        else:                                                    # A2: cluster prior is the complexity score
            B_tr["c_hat"], B_va["c_hat"] = B_tr.som_prior, B_va.som_prior

        # S5 Mamdani
        self.mcols = list(c["mamdani_inputs"])
        if c["mamdani"] == "learned":
            self.fis = LearnedMamdani(self.mcols, c["mamdani_terms"], c["mamdani_out"], c["min_support"],
                                      seed=c["seed"])
            self.fis.fit(B_tr[self.mcols].values, y_tr, tune=c["mamdani_tune"], gens=c["mamdani_gens"])
        elif c["mamdani"] == "expert":                          # A4: PBL hand-written rule base + calibration
            self.fis = MamdaniFIS()
            self.escale = _Scaler01().fit(B_tr[self.mcols].values)
            crisp = self._expert_crisp(B_va)
            self.cal = np.polyfit(crisp, y_va, 2)
        z_va = self._head(B_va)
        self.smear = mae_shift(val.effort_h.values, val.sp.values * np.exp(z_va))

        # S6 conformal (validation window, time-ordered)
        self.conf_cols = MEASURED + ["clarity_hat", "integration_hat", "c_hat"]
        r_va = y_va - (z_va + self.smear)
        self.conf = ConformalIntervals(alpha=c["alpha"], seed=c["seed"]).fit(B_va[self.conf_cols].values, r_va,
                                                                             B_va.som_bmu.values)
        pv = val.sp.values * np.exp(z_va + self.smear)
        self.proj_conf = ProjectConformal(alpha=c["alpha"], seed=c["seed"]).fit(val.effort_h.values, pv,
                                                                               val.project_id.values)
        self.val_mae = float(np.abs(pv - val.effort_h.values).mean())
        return self

    # ------------------------------------------------------------------ internals
    def _add_som(self, B):
        Z = self.som_scaler.transform(B[self.som_cols].values)
        bmu, prior = self.som.transform(Z)
        B["som_bmu"] = bmu
        B["som_prior"] = prior if self.cfg["use_som"] else self.som.global_mean
        return B

    def _anfis(self, B):
        return self.anfis.predict(self.ascale(B[self.anfis_cols].values)) * self.y_sd + self.y_mu

    def _expert_crisp(self, B):
        X = 10 * self.escale(B[self.mcols].values)
        # complexity up, clarity (higher = clearer) and experience (familiarity) as in the PBL rule base
        return self.fis.infer_batch(X[:, 0], X[:, 1], X[:, 2])[0]

    def _head(self, B):
        m = self.cfg["mamdani"]
        if m == "learned":
            return self.fis.predict(B[self.mcols].values)
        if m == "expert":
            return np.polyval(self.cal, self._expert_crisp(B))
        return B.c_hat.values                                    # A3: ANFIS output used directly

    def transform(self, df):
        B = self._base(df.reset_index(drop=True))
        self._add_som(B)
        B["c_hat"] = self._anfis(B) if self.cfg["use_anfis"] else B.som_prior
        return B

    # ================================================================== public API
    def predict(self, df, intervals=True):
        df = df.reset_index(drop=True)
        B = self.transform(df)
        z = self._head(B) + self.smear
        hours = df.sp.values * np.exp(z)
        out = pd.DataFrame(dict(issue_id=df.issue_id.values, hours=hours, log_density=z, c_hat=B.c_hat.values,
                                clarity_hat=B.clarity_hat.values, integration_hat=B.integration_hat.values,
                                cluster=B.som_bmu.values, cluster_name=[self.som.names[b] for b in B.som_bmu]))
        if intervals:
            lo, hi, sig = self.conf.interval(hours, B[self.conf_cols].values, B.som_bmu.values)
            out["low"], out["high"], out["sigma_log"] = lo, hi, sig
        for col in self.prof_cols:
            out[col] = B[col].values
        return out

    def project_interval(self, total_hours, n_requirements=20):
        return self.proj_conf.interval(total_hours, n_requirements)

    def explain(self, row, top=5):
        df = pd.DataFrame([row]) if not isinstance(row, pd.DataFrame) else row.head(1)
        B = self.transform(df)
        info = dict(cluster=self.som.names[int(B.som_bmu.iloc[0])], c_hat=float(B.c_hat.iloc[0]),
                    clarity_hat=float(B.clarity_hat.iloc[0]), integration_hat=float(B.integration_hat.iloc[0]))
        if self.cfg["mamdani"] == "learned":
            z, fired, agg = self.fis.explain(B[self.mcols].values[0], top=top)
            info.update(rules=fired, aggregated=agg, ygrid=self.fis.ygrid, crisp=z)
        return info

    def update(self, resolved):
        """Online learning from newly resolved issues (time-ordered)."""
        if len(resolved) == 0:
            return self
        B = self.transform(resolved)
        y = log_density(resolved.reset_index(drop=True))
        if self.cfg["use_anfis"]:
            self.anfis.partial_fit(self.ascale(B[self.anfis_cols].values), (y - self.y_mu) / self.y_sd)
        Z = self.som_scaler.transform(B[self.som_cols].values)
        for zi, yi in zip(Z, y):
            self.som.update_prior(zi[None], yi)
        z = self._head(self.transform(resolved)) + self.smear
        self.conf.update(B[self.conf_cols].values, y - z, B.som_bmu.values)
        return self


def tune(train, val, components, n_trials=8, base_cfg=None, seed=0, log=None):
    """Random search over SPACE on validation MAE (hours).  The base (S1/S2) is fitted once."""
    rng = np.random.default_rng(seed)
    base = DynEstR(base_cfg).fit_base(train.sort_values("resolved_at").reset_index(drop=True), components)
    best = None
    for k in range(n_trials):
        cfg = dict(base.cfg)
        if k > 0:
            for key, opts in SPACE.items():
                cfg[key] = opts[rng.integers(len(opts))]
        m = DynEstR(cfg)
        m.fb, m.aux = base.fb, base.aux
        m.fit(train, val, components, base_ready=True)
        if log:
            log(f"    trial {k}: val MAE {m.val_mae:.2f} h  {({x: cfg[x] for x in SPACE})}")
        if best is None or m.val_mae < best.val_mae:
            best = m
    return best
