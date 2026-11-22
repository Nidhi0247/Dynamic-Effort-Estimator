"""Execution model (S7): every scheduling parameter is learnt from logged effort.

Efficiency (ridge mixed model, partial pooling):
    log(actual_i / ref_i) = mu + a_d + b_{d,k} + beta_wip * wip_i + beta_re * reassigned_i
                            + beta_fam * log1p(fam_i) + beta_ten * log1p(tenure_i) + e
    eta_{d,k} = exp(-(a_d + b_{d,k}))      (reference developer = 1; unseen developers -> 1)
    kappa     = max(beta_wip, 0) * median(actual)         hours lost per extra concurrent requirement
    handover  = max(exp(beta_re) - 1, 0)                   extra share of done work on reassignment
Capacity: logged hours per developer and sprint (from `timespent` changelog increments), forecast
with simple exponential smoothing whose alpha is chosen by one-step-ahead error on training sprints.
"""
import numpy as np
import pandas as pd


class EfficiencyModel:
    def __init__(self, lam_dev=5.0, lam_dk=10.0, lam_x=1.0):
        self.lam_dev, self.lam_dk, self.lam_x = lam_dev, lam_dk, lam_x

    def fit(self, df, n_clusters):
        """df columns: dev, cluster, actual, ref, wip, reassigned, familiarity, tenure."""
        d = df.dropna(subset=["dev"]).copy()
        d = d[(d.actual > 0) & (d.ref > 0)]
        self.K = n_clusters
        self.devs = sorted(d.dev.unique())
        di = {v: i for i, v in enumerate(self.devs)}
        nD, n = len(self.devs), len(d)
        X_dev = np.zeros((n, nD))
        X_dk = np.zeros((n, nD * n_clusters))
        rows = np.arange(n)
        dv = d.dev.map(di).values
        X_dev[rows, dv] = 1
        X_dk[rows, dv * n_clusters + d.cluster.values.astype(int)] = 1
        Z = np.column_stack([d.wip.values, d.reassigned.values, np.log1p(d.familiarity.values),
                             np.log1p(d.tenure.values)])
        X = np.column_stack([np.ones(n), Z, X_dev, X_dk])
        y = np.log(d.actual.values / d.ref.values)
        pen = np.r_[0.0, np.full(Z.shape[1], self.lam_x), np.full(nD, self.lam_dev), np.full(nD * n_clusters, self.lam_dk)]
        beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
        self.mu = beta[0]
        self.beta_wip, self.beta_re, self.beta_fam, self.beta_ten = beta[1:5]
        a = beta[5:5 + nD]
        b = beta[5 + nD:].reshape(nD, n_clusters)
        self.log_eta = -(a[:, None] + b)
        self.kappa = float(max(self.beta_wip, 0.0) * np.median(d.actual.values))
        self.handover = float(max(np.exp(self.beta_re) - 1, 0.0))
        self.resid_sd = float(np.std(y - X @ beta))
        self.n = n
        return self

    def eta(self, devs):
        """(D, K) efficiency matrix for the given developer ids (unseen -> pooled reference 1.0)."""
        idx = {v: i for i, v in enumerate(self.devs)}
        out = np.ones((len(devs), self.K))
        for r, d in enumerate(devs):
            if d in idx:
                out[r] = np.exp(self.log_eta[idx[d]])
        return out

    def summary(self):
        return dict(n=self.n, mu=float(self.mu), beta_wip=float(self.beta_wip), beta_reassigned=float(self.beta_re),
                    beta_familiarity=float(self.beta_fam), beta_tenure=float(self.beta_ten),
                    kappa_h=self.kappa, handover=self.handover, resid_sd=self.resid_sd)


def logged_hours_by_sprint(ts_changes, sprint_bounds, issue_owner=None):
    """Hours logged per (author, sprint) from `timespent` changelog increments.

    ts_changes   : changelog rows of field 'timespent' (From_Value / To_Value in seconds, Author_ID)
    sprint_bounds: DataFrame with columns sprint_no, start, end (non-overlapping, sorted)
    issue_owner  : optional Series issue_id -> assignee used when Author_ID is missing
    """
    c = ts_changes.copy()
    c["inc"] = (pd.to_numeric(c.To_Value, errors="coerce").fillna(0)
                - pd.to_numeric(c.From_Value, errors="coerce").fillna(0)) / 3600.0
    c = c[c.inc > 0]
    if issue_owner is not None:
        c["Author_ID"] = c["Author_ID"].fillna(c.Issue_ID.map(issue_owner))
    starts = sprint_bounds.start.values.astype("datetime64[ns]")
    k = np.searchsorted(starts, c.Creation_Date.values.astype("datetime64[ns]"), side="right") - 1
    c["sprint_no"] = k
    c = c[(k >= 0) & (k < len(sprint_bounds))]
    return c.groupby(["Author_ID", "sprint_no"]).inc.sum().unstack(fill_value=0.0)


class CapacityForecaster:
    """Per-developer simple exponential smoothing of logged hours per sprint; alpha learnt on history."""

    def fit(self, hours, alphas=(0.1, 0.2, 0.3, 0.5, 0.7)):
        """hours: DataFrame dev x sprint_no (training sprints only)."""
        H = hours.values
        best = (np.inf, 0.3)
        for a in alphas:
            err, lvl = [], H[:, 0].copy()
            for s in range(1, H.shape[1]):
                err.append(np.abs(H[:, s] - lvl))
                lvl = a * H[:, s] + (1 - a) * lvl
            e = np.mean(err) if err else np.inf
            if e < best[0]:
                best = (e, a)
        self.alpha, self.mae = best[1], best[0]
        lvl = H[:, 0].copy()
        for s in range(1, H.shape[1]):
            lvl = self.alpha * H[:, s] + (1 - self.alpha) * lvl
        self.level = pd.Series(lvl, index=hours.index)
        return self

    def forecast(self, devs, horizon):
        lv = np.array([self.level.get(d, np.median(self.level.values) if len(self.level) else 0.0) for d in devs])
        return np.repeat(lv[:, None], horizon, axis=1)
