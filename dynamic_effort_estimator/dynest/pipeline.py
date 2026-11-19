from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .data import FEATURES, CLUSTERS, load_history
from .som import SOM
from .anfis import ANFIS
from .mamdani import MamdaniFIS, rule_text
Z95 = 1.96
@dataclass
class ProjectEstimate:
    table: pd.DataFrame          
    effort: float                
    sigma: float
    low: float
    high: float
    team_size: int
    focus: float
    duration_days: float
    duration_low: float
    duration_high: float
    rule_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    @property
    def band(self):
        return Z95 * self.sigma

    def team_for(self, target_days):
        return int(np.ceil(self.effort / (target_days * self.focus)))

def _metrics(actual, pred):
    rel = np.abs(actual - pred) / actual
    return dict(MMRE=float(rel.mean()), PRED25=float((rel <= 0.25).mean()),
                MAE=float(np.abs(actual - pred).mean()))

class EffortPipeline:
    def __init__(self, seed=0):
        self.seed = seed
        self.fis = MamdaniFIS()
        self.fitted = False

    def fit(self, hist=None):
        hist = load_history() if hist is None else hist
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(hist))
        n_tr, n_va = int(.65 * len(hist)), int(.15 * len(hist))
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        self.history = hist

        F = hist[FEATURES].values / 10
        rating = hist["assessed_complexity"].values
        self.som = SOM(4, 4, len(FEATURES), seed=self.seed).fit(F[tr])
        self.som.fit_priors(F[tr], rating[tr])
        _, _, prior = self.som.transform(F)
        X = np.c_[hist["complexity"], hist["integration"], prior] / 10
        self.anfis = ANFIS(3, 3).fit(X[tr], rating[tr] / 10, X[va], rating[va] / 10)

        r_hat = self._rating(X)
        crisp, _, _ = self.fis.infer_batch(r_hat, hist["clarity"], hist["experience"])
        dens = (hist["actual_effort"] / hist["size_points"]).values
        self.cal = np.polyfit(crisp[va], dens[va], 2)          
        pred_va = hist["size_points"].values[va] * np.clip(np.polyval(self.cal, crisp[va]), 0.05, None)
        self.sigma_log = float(np.std(np.log(hist["actual_effort"].values[va] / pred_va)))

        size = hist["size_points"].values
        pred_te = size[te] * np.clip(np.polyval(self.cal, crisp[te]), 0.05, None)
        B = np.c_[np.ones(len(hist)), hist[["complexity", "integration", "clarity", "experience"]].values]
        beta = np.linalg.lstsq(B[tr], dens[tr], rcond=None)[0]
        base_te = size[te] * np.clip(B[te] @ beta, 0.05, None)
        act = hist["actual_effort"].values[te]
        self.metrics = pd.DataFrame({
            "SOM+ANFIS+Mamdani (proposed)": _metrics(act, pred_te),
            "Linear regression baseline": _metrics(act, base_te)}).T
        self.anfis_rmse_test = float(np.sqrt(((r_hat[te] - rating[te]) ** 2).mean()))
        self.som_qe = self.som.quantization_error(F[te])
        self.fitted = True
        return self

    def _rating(self, X):
        return np.clip(self.anfis.predict(X) * 10, 0, 10)

    def estimate_requirements(self, df):
        df = df.copy().reset_index(drop=True)
        F = df[FEATURES].astype(float).values / 10
        bmu, lab, prior = self.som.transform(F)                                         
        rating = self._rating(np.c_[df["complexity"], df["integration"], prior] / 10)   
        crisp, strengths, _ = self.fis.infer_batch(rating, df["clarity"], df["experience"])  
        inten = np.clip(np.polyval(self.cal, crisp), 0.05, None)
        df["som_neuron"], df["cluster"] = bmu, lab
        df["cluster_name"] = [CLUSTERS[i] for i in lab]
        df["complexity_rating"] = rating
        df["intensity"] = inten
        df["effort"] = inten * df["size_points"].astype(float)
        df["sigma"] = df["effort"] * self.sigma_log * (1 + (10 - df["clarity"].astype(float)) / 20)
        self._last_strengths = strengths
        return df

    def explain(self, row):
        r = self.estimate_requirements(pd.DataFrame([row]))
        _, fired, agg = self.fis.explain(r["complexity_rating"][0], row["clarity"], row["experience"])
        return r.iloc[0], fired, agg

    def estimate_project(self, df, team_size=4, focus=0.8):
        df = df[df["active"]] if "active" in df else df
        if len(df) == 0:
            empty = pd.DataFrame(columns=list(df.columns) + ["effort", "sigma"])
            return ProjectEstimate(empty, 0, 0, 0, 0, team_size, focus, 0, 0, 0)
        t = self.estimate_requirements(df)
        s = self._last_strengths
        eff = float(t["effort"].sum())
        sigma = float(np.sqrt((t["sigma"] ** 2).sum()) + 0.03 * eff)           
        cap = team_size * focus
        rs = pd.DataFrame({"rule": [rule_text(r) for r in self.fis.rules],
                           "mean_firing": s.mean(0), "max_firing": s.max(0)})
        rs = rs[rs["max_firing"] > 1e-3].sort_values("mean_firing", ascending=False).reset_index(drop=True)
        return ProjectEstimate(t, eff, sigma, max(eff - Z95 * sigma, 0), eff + Z95 * sigma, team_size, focus, eff / cap, max(eff - Z95 * sigma, 0) / cap, (eff + Z95 * sigma) / cap, rs)
