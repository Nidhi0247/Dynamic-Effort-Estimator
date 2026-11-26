"""E8: classic project-level benchmark (Function Points / COCOMO datasets).

Datasets (public PROMISE copies, data/classic_raw): Albrecht, Kemerer, Desharnais, China, Maxwell, Kitchenham,
Miyazaki94, COCOMO81, NASA93.  Protocol: R x 10-fold cross-validation per dataset (same folds for every model);
inside each training fold every tuned model gets the same random-search budget scored by inner 3-fold CV,
and the same back-transform and split-conformal 95% interval recipe (dynest/classic/models.py).

Compared: NeuroFuzzy (SOM -> ANFIS -> learned Mamdani, proposed) and its ablations; Median; FP regression or
KLOC power law; COCOMO-81 nominal / local calibration / refit (COCOMO81, NASA93); analogy (ABE); Ridge; CART;
SVR; random forest; GBM; MLP.  Kitchenham's recorded first human estimate is reported as a reference.

Units of analysis: projects within a dataset (paired Wilcoxon on absolute errors averaged over repeats, Holm),
and datasets across the benchmark (Friedman + Nemenyi on SA and on mean |log error|).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynest import stats  # noqa: E402
from dynest.classic.datasets import load_all, LOADERS  # noqa: E402
from dynest.classic.models import estimators, ablations, Estimator, NeuroFuzzyCore, nf_space  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
PROPOSED = "NeuroFuzzy (proposed)"
TRANSPARENT = "NeuroFuzzy-T (proposed, transparent)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def task(name, rep, k, test_idx, trials, inner):
    warnings.filterwarnings("ignore")
    ds = load_all([name])[name]
    te = np.asarray(test_idx)
    tr = np.setdiff1d(np.arange(ds.n), te)
    Xtr, ytr, Xte = ds.X.iloc[tr], ds.y[tr], ds.X.iloc[te]
    seed = 1000 * rep + k
    rows, hp = [], None
    models = estimators(ds)
    for e in models:
        try:
            e.fit(Xtr, ytr, ds, n_trials=trials, inner=inner, seed=seed)
        except Exception as ex:                                         # recorded, never silently dropped
            rows += [dict(dataset=name, rep=rep, fold=k, obs=int(i), model=e.name, actual=ds.y[i], pred=np.nan,
                          low=np.nan, high=np.nan, error=str(ex)[:80]) for i in te]
            continue
        if e.name == PROPOSED:
            hp = e.hp
            models += ablations(hp)
        p, lo, hi = e.predict(Xte)
        rows += [dict(dataset=name, rep=rep, fold=k, obs=int(i), model=e.name, actual=ds.y[i], pred=a, low=b, high=c)
                 for i, a, b, c in zip(te, p, lo, hi)]
    if ds.expert is not None:
        rows += [dict(dataset=name, rep=rep, fold=k, obs=int(i), model="Expert first estimate (reference)",
                      actual=ds.y[i], pred=ds.expert[i], low=np.nan, high=np.nan) for i in te]
    return rows, dict(dataset=name, rep=rep, fold=k, **{f"hp_{a}": str(b) for a, b in (hp or {}).items()})


def p0_mae(y):
    """Exact MAE of random guessing (every other project equally likely): mean_i mean_{j!=i} |y_i - y_j|."""
    y = np.asarray(y, float)
    d = np.abs(y[:, None] - y[None])
    return float(d.sum() / (len(y) * (len(y) - 1)))


def summarise(P, D):
    rows = []
    for (name, model, rep), g in P.groupby(["dataset", "model", "rep"]):
        g = g.dropna(subset=["pred"])
        if len(g) < D[name].n:
            continue
        a, p = g.actual.values, g.pred.values
        ae = np.abs(a - p)
        r = dict(dataset=name, model=model, rep=rep, MAE=ae.mean(), MdAE=np.median(ae),
                 SA=1 - ae.mean() / p0_mae(D[name].y), LMAE=np.abs(np.log(a / p)).mean(),
                 MMRE=(ae / a).mean(), MdMRE=np.median(ae / a), PRED25=(ae / a <= 0.25).mean(),
                 MBRE=(ae / np.minimum(a, p)).mean(), MIBRE=(ae / np.maximum(a, p)).mean())
        if g.low.notna().all():
            lo, hi, al = g.low.values, g.high.values, 0.05
            r["coverage"] = ((a >= lo) & (a <= hi)).mean()
            r["rel_width"] = np.median((hi - lo) / p)
            score = (hi - lo) + (2 / al) * (lo - a) * (a < lo) + (2 / al) * (a - hi) * (a > hi)
            r["interval_score_rel"] = score.mean() / a.mean()
        rows.append(r)
    S = pd.DataFrame(rows)
    return S.groupby(["dataset", "model"]).mean(numeric_only=True).drop(columns="rep").reset_index()


def per_dataset_tests(P):
    out = []
    for name, g in P.dropna(subset=["pred"]).groupby("dataset"):
        g = g.assign(ae=(g.actual - g.pred).abs(), lae=np.abs(np.log(g.actual / g.pred)))
        for metric in ("ae", "lae"):
            t = g.pivot_table(index="obs", columns="model", values=metric, aggfunc="mean").dropna(axis=1)
            if PROPOSED not in t:
                continue
            res = []
            for m in t.columns:
                if m == PROPOSED:
                    continue
                d = t[PROPOSED] - t[m]
                p = float(stats.stats.wilcoxon(t[PROPOSED], t[m]).pvalue) if (d != 0).any() else 1.0
                res.append(dict(dataset=name, metric=metric.upper(), other=m, n=len(t),
                                median_diff=float(d.median()), p=p, r_rb=stats.rank_biserial_paired(t[PROPOSED], t[m])))
            res = pd.DataFrame(res)
            res["p_holm"] = stats.holm(res.p.values)
            out.append(res)
    return pd.concat(out, ignore_index=True)


def cross_dataset(S, metric, lower):
    t = S.pivot_table(index="dataset", columns="model", values=metric).dropna(axis=1)
    t = t[[c for c in t.columns if not c.startswith("Expert")]]
    return t, stats.compare(t, lower_is_better=lower)


def rule_bases(D, trials, inner):
    """Proposed model tuned and fitted on each complete dataset: its learnt rules (for the interpretability section)."""
    out = []
    variants = ((PROPOSED, {}), (TRANSPARENT, dict(transparent=True, use_som=True, use_anfis=False, use_mamdani=True)))
    for name, ds in D.items():
        for vname, fixed in variants:
            e = Estimator(vname, NeuroFuzzyCore, nf_space, fixed=fixed).fit(ds.X, ds.y, ds, n_trials=trials,
                                                                            inner=inner, seed=7)
            m = e.model
            if not e.hp["use_mamdani"]:
                continue
            Z = m.p.transform(ds.X)
            first = m.som.transform(Z)[1] if fixed else m._c_hat(Z)
            tab = m.fis.rule_table()
            tab.insert(0, "dataset", name)
            tab.insert(1, "variant", vname)
            tab["n_rules"] = len(m.fis.rules)
            tab["coverage"] = m.fis.coverage(np.column_stack([first] + [Z[:, j] for j in m.m_idx]))
            tab["mamdani_inputs"] = ", ".join(m.fis.input_names)
            tab["target"] = "log effort per size unit" if e.hp["density"] else "log effort"
            out.append(tab)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.repeats, a.folds, a.trials, a.datasets = 1, 3, 3, ["Albrecht", "COCOMO81"]
    D = load_all(a.datasets)
    jobs = []
    for name, ds in D.items():
        for rep in range(a.repeats):
            perm = np.random.default_rng(100 + rep).permutation(ds.n)           # same folds for every model
            for k, te in enumerate(np.array_split(perm, min(a.folds, ds.n))):
                jobs.append((name, rep, k, te.tolist()))
    jobs.sort(key=lambda j: -D[j[0]].n)                                          # big datasets first
    log(f"{len(D)} datasets, {len(jobs)} outer folds, trials={a.trials}, inner={a.inner}")
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(*j, a.trials, a.inner) for j in jobs)
    P = pd.DataFrame([r for rows, _ in res for r in rows])
    H = pd.DataFrame([h for _, h in res])
    tag = "_smoke" if a.smoke else ""
    P.to_csv(RESULTS / f"e8_predictions{tag}.csv", index=False)
    H.to_csv(RESULTS / f"e8_hyperparams{tag}.csv", index=False)
    if "error" in P:
        bad = P.dropna(subset=["error"]).groupby(["dataset", "model"]).size()
        if len(bad):
            log("fit failures:\n" + bad.to_string())

    S = summarise(P, D)
    S.to_csv(RESULTS / f"e8_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    for name in D:
        s = S[S.dataset == name].sort_values("MAE")
        print(f"\n=== {name} (n={D[name].n}, unit {D[name].unit})")
        print(s[["model", "MAE", "MdAE", "SA", "LMAE", "MdMRE", "PRED25", "coverage", "rel_width",
                 "interval_score_rel"]].round(3).to_string(index=False))

    T = per_dataset_tests(P)
    T.to_csv(RESULTS / f"e8_pairwise{tag}.csv", index=False)
    print("\nProposed vs each model, per dataset (AE; negative median_diff / r_rb = proposed better):")
    show = T[T.metric == "AE"].copy()
    show["sig"] = np.where(show.p_holm < 0.05, np.where(show.r_rb < 0, "better", "worse"), "")
    print(show.pivot_table(index="other", columns="dataset", values="r_rb").round(2).to_string())
    print(show.groupby("other").sig.agg(lambda s: f"{(s == 'better').sum()} better / {(s == 'worse').sum()} worse")
          .to_string())

    for metric, lower in (("SA", False), ("LMAE", True), ("MdAE", True)):
        t, out = cross_dataset(S, metric, lower)
        print(f"\n[{metric} across {out['n_units']} datasets] Friedman p={out.get('friedman_p', np.nan):.3g} "
              f"CD={out['cd_nemenyi']:.2f}")
        print("mean ranks:", {k: round(v, 2) for k, v in sorted(out["mean_rank"].items(), key=lambda x: x[1])})
        out["pairwise"].to_csv(RESULTS / f"e8_cross_{metric}{tag}.csv", index=False)
        pd.Series(out["mean_rank"]).to_csv(RESULTS / f"e8_ranks_{metric}{tag}.csv")

    log("fitting rule bases on complete datasets")
    R = rule_bases(D, a.trials, a.inner)
    R.to_csv(RESULTS / f"e8_rules{tag}.csv", index=False)
    print(R.groupby(["dataset", "variant"])[["n_rules", "coverage", "mamdani_inputs", "target"]].first().to_string())
    log("done")


if __name__ == "__main__":
    main()
