"""Non-parametric statistics for paired comparisons (projects / streams / issues as units)."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None]).sum()
    eq = (x[:, None] == y[None]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def a12_magnitude(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def rank_biserial_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(p) - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def bootstrap_ci(x, stat=np.median, level=0.95, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3 or np.allclose(x, x[0]):
        v = float(stat(x)) if len(x) else np.nan
        return v, v
    res = stats.bootstrap((x,), stat, confidence_level=level, n_resamples=n, method="BCa",
                          random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def compare(table, lower_is_better=True):
    """table: rows = units (paired), columns = methods.  Friedman + pairwise Wilcoxon/Holm + effect sizes."""
    t = table.dropna()
    out = dict(n_units=len(t))
    if t.shape[1] >= 3 and len(t) >= 3:
        out["friedman_stat"], out["friedman_p"] = map(float, stats.friedmanchisquare(*[t[c] for c in t]))
    ranks = t.rank(axis=1, ascending=lower_is_better).mean()
    out["mean_rank"] = ranks.to_dict()
    rows = []
    for a, b in itertools.combinations(t.columns, 2):
        d = t[a] - t[b]
        p = float(stats.wilcoxon(t[a], t[b]).pvalue) if (d != 0).any() and len(t) >= 5 else np.nan
        lo, hi = bootstrap_ci(d.values)
        rows.append(dict(a=a, b=b, median_diff=float(d.median()), ci_low=lo, ci_high=hi, p=p,
                         A12=a12(t[a], t[b]), r_rb=rank_biserial_paired(t[a], t[b])))
    pw = pd.DataFrame(rows)
    if len(pw):
        pw["p_holm"] = holm(pw["p"].fillna(1.0))
        pw["magnitude"] = pw["A12"].map(a12_magnitude)
    out["pairwise"] = pw
    out["cd_nemenyi"] = nemenyi_cd(t.shape[1], len(t))
    return out


def nemenyi_cd(k, n, alpha=0.05):
    if k < 2 or n < 2:
        return np.nan
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))
