"""Accuracy metrics and leakage-free (time-ordered) splitters."""
import numpy as np
import pandas as pd


def metrics(actual, pred, train_actual=None, low=None, high=None):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ae = np.abs(a - p)
    mre = ae / a
    bre = ae / np.minimum(a, p)
    ibre = ae / np.maximum(a, p)
    out = dict(n=len(a), MAE=ae.mean(), MdAE=np.median(ae), RMSE=np.sqrt(((a - p) ** 2).mean()),
               MMRE=mre.mean(), MdMRE=np.median(mre), PRED25=(mre <= 0.25).mean(),
               MBRE=bre.mean(), MIBRE=ibre.mean(),
               LMAE=np.abs(np.log(a) - np.log(p)).mean())
    if train_actual is not None:
        out["SA"] = standardized_accuracy(a, p, train_actual)
    if low is not None and high is not None:
        lo, hi = np.asarray(low, float), np.asarray(high, float)
        out["coverage"] = ((a >= lo) & (a <= hi)).mean()
        out["rel_width"] = np.median((hi - lo) / p)
        al = 0.05                                                   # Winkler interval score, relative to mean actual
        score = (hi - lo) + (2 / al) * (lo - a) * (a < lo) + (2 / al) * (a - hi) * (a > hi)
        out["interval_score_rel"] = score.mean() / a.mean()
    return {k: float(v) for k, v in out.items()}


def standardized_accuracy(actual, pred, train_actual, runs=1000, seed=0):
    """SA = 1 - MAE / MAE_P0 (Shepperd & MacDonell 2012); P0 = random guessing from training actuals."""
    rng = np.random.default_rng(seed)
    a = np.asarray(actual, float)
    pool = np.asarray(train_actual, float)
    guesses = pool[rng.integers(0, len(pool), (runs, len(a)))]
    mae_p0 = np.abs(guesses - a[None]).mean()
    return float(1 - np.abs(a - np.asarray(pred, float)).mean() / mae_p0)


def rolling_origins(times, n_origins=3, train_min=0.5, val_frac=0.15, test_frac=0.15, gap=pd.Timedelta(0)):
    """Yield (train_idx, val_idx, test_idx) for a time-ordered sample (times = resolution timestamps).

    Origin k trains on everything resolved before its cut, validates on the next val_frac and tests on
    the following test_frac.  Origins are spaced evenly between train_min and 1 - val - test.
    """
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    order = np.argsort(t.values, kind="stable")
    n = len(order)
    last = 1 - val_frac - test_frac
    fracs = np.linspace(train_min, last, n_origins) if n_origins > 1 else [last]
    for f in fracs:
        a, b, c = int(f * n), int((f + val_frac) * n), int((f + val_frac + test_frac) * n)
        tr = order[:a]
        if gap > pd.Timedelta(0):
            cutoff = t.iloc[order[a]] - gap
            tr = tr[t.iloc[tr].values < cutoff.to_datetime64()]
        yield tr, order[a:b], order[b:c]
