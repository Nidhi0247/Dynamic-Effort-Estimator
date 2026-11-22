"""S1 measured features (counts and as-of values, no ratings) + text representation."""
import numpy as np
import pandas as pd

from .text import TextEncoder

# Jira priority scheme order (measurement mapping of a categorical field, not a rating)
_PRIORITY = {"blocker": 5, "highest": 5, "critical": 4, "high": 4, "major": 3, "medium": 3, "normal": 3,
             "minor": 2, "low": 2, "trivial": 1, "lowest": 1}
TYPES = ["Bug", "Story", "Task", "Improvement", "New Feature", "Sub-task", "Epic", "Technical task"]

MEASURED = ["log_sp", "priority_ord", "n_components_t", "n_links_t", "desc_words_log", "ac_lines", "comments_t",
            "reporter_prior_log", "familiarity_log", "tenure_log", "age_log"] + [f"type_{t}" for t in TYPES]


def priority_ord(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 3.0
    s = str(v).lower()
    for k, val in _PRIORITY.items():
        if k in s:
            return float(val)
    return 3.0


def measured(df):
    out = pd.DataFrame(index=df.index)
    out["log_sp"] = np.log(df.sp.astype(float).clip(lower=0.5))
    out["priority_ord"] = df.priority_t.map(priority_ord)
    out["n_components_t"] = df.n_components_t.astype(float)
    out["n_links_t"] = df.n_links_t.astype(float)
    out["desc_words_log"] = np.log1p(df.desc_words.astype(float))
    out["ac_lines"] = df.ac_lines.astype(float).clip(upper=30)
    out["comments_t"] = df.comments_t.astype(float).clip(upper=50)
    out["reporter_prior_log"] = np.log1p(df.reporter_prior.astype(float))
    out["familiarity_log"] = np.log1p(df.familiarity.astype(float))
    out["tenure_log"] = np.log1p(df.tenure_days.astype(float))
    out["age_log"] = np.log1p(df.age_days_t.astype(float).clip(lower=0))
    for t in TYPES:
        out[f"type_{t}"] = (df.issuetype_t.astype(str) == t).astype(float)
    return out.fillna(0.0)                     # missing counts/durations mean "none recorded"


class FeatureBuilder:
    """Fitted on the training window only (text PCA/SVD); transform() is pure."""

    def __init__(self, text_backend="minilm", text_dim=16, seed=0):
        self.enc = TextEncoder(text_backend, text_dim, seed)

    def fit(self, train):
        self.enc.fit(train.text.fillna(""))
        self.text_cols = [f"txt{i}" for i in range(self.enc.dim)]
        return self

    def transform(self, df):
        X = measured(df)
        T = self.enc.transform(df.text.fillna(""))
        for i, c in enumerate(self.text_cols):
            X[c] = T[:, i] if i < T.shape[1] else 0.0
        return X

    @property
    def columns(self):
        return MEASURED + self.text_cols
