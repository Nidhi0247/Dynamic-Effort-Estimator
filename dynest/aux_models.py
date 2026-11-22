"""S2 auxiliary attribute models: learned replacements for the human 0-10 ratings.

clarity_hat      predicts clarity_label     (-log1p of later rewrites + clarification questions)
integration_hat  predicts integration_label (log1p of links + components at resolution)
profile_hat      predicts the issue's component-topic mix: NMF topics over the component labels of
                 training-window issues (k chosen by reconstruction error), i.e. functional profiles
                 are discovered from the data instead of the fixed UI/backend/data trio.
All labels exist only for issues resolved inside the training window.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingRegressor


def _gbm(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                         random_state=seed)


class AuxModels:
    def __init__(self, k_range=(3, 4, 5, 6), seed=0):
        self.k_range, self.seed = k_range, seed

    def fit(self, X, frame, components):
        """X: S1 features (train); frame: train rows (labels); components: Issue_ID -> list of component names."""
        self.cols = list(X.columns)
        self.clarity = _gbm(self.seed).fit(X.values, frame.clarity_label.values)
        self.integration = _gbm(self.seed).fit(X.values, frame.integration_label.values)
        self._fit_profiles(X, frame, components)
        return self

    def _fit_profiles(self, X, frame, components):
        comps = frame.issue_id.map(components)
        has = comps.map(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0).values
        vocab = pd.Series([c for v in comps[has] for c in v]).value_counts()
        vocab = vocab[vocab >= 5].index.tolist()
        self.vocab = vocab
        if len(vocab) < 4 or has.sum() < 50:
            self.k, self.topics, self.profile = 0, None, None
            self.topic_names = []
            return
        idx = {c: i for i, c in enumerate(vocab)}
        M = np.zeros((has.sum(), len(vocab)))
        for r, v in enumerate(comps[has]):
            for c in v:
                if c in idx:
                    M[r, idx[c]] = 1.0
        best = None
        for k in self.k_range:
            if k >= len(vocab):
                break
            nmf = NMF(k, init="nndsvda", random_state=self.seed, max_iter=400).fit(M)
            err = nmf.reconstruction_err_ / max(np.linalg.norm(M), 1e-9)
            score = err + 0.02 * k                                  # parsimony: each topic must buy 2% error
            if best is None or score < best[0]:
                best = (score, k, nmf)
        _, self.k, self.topics = best
        W = self.topics.transform(M)
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        keep = W.sum(1) > 0
        self.profile = [_gbm(self.seed).fit(X.values[has][keep], W[keep, j]) for j in range(self.k)]
        H = self.topics.components_
        self.topic_names = [" / ".join(vocab[i] for i in np.argsort(-H[j])[:2]) for j in range(self.k)]

    def transform(self, X):
        A = X[self.cols].values
        out = pd.DataFrame(index=X.index)
        out["clarity_hat"] = self.clarity.predict(A)
        out["integration_hat"] = self.integration.predict(A)
        if self.k:
            P = np.column_stack([m.predict(A) for m in self.profile]).clip(min=0)
            P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
            for j in range(self.k):
                out[f"profile{j}"] = P[:, j]
        return out

    @property
    def profile_cols(self):
        return [f"profile{j}" for j in range(self.k)]
