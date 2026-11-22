"""E2: estimation on projects never seen in training (leave-one-project-out, LOPO).

For each of the 22 projects: train + validate on the other 21 (time-ordered split), test on every
resolved requirement of the held-out project.  DynEst-R and all baselines get the same tuning budget
on that validation split.  Unit of analysis = held-out project.
"""
from common import args, dataset, log, save
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dynest import tune
from dynest.baselines import all_baselines
from dynest.evaluation import metrics
from dynest import stats
from e1_accuracy import CAEA_MAP, COCOMO, with_features


def fold(p, tr, va, te, components, trials, backend, seed):
    m = tune(tr, va, components, n_trials=trials, base_cfg=dict(text_backend=backend), seed=seed)
    rows = []
    pr = m.predict(te)
    rows.append(dict(model="DynEst-R (proposed)", project_id=p,
                     **metrics(te.effort_h, pr.hours, tr.effort_h, pr.low, pr.high)))
    feats = list(m.fb.columns) + ["clarity_hat", "integration_hat"] + m.prof_cols
    TR, VA, TE = with_features(m, tr), with_features(m, va), with_features(m, te)
    for b in all_baselines(feats, COCOMO, CAEA_MAP, n_trials=trials, seed=seed):
        b.fit(TR, VA)
        rows.append(dict(model=b.name, project_id=p, **metrics(TE.hours, b.predict(TE), TR.hours)))
    return rows


def main():
    a = args([(["--trials"], dict(type=int, default=6))])
    ds = dataset()
    folds = list(ds.lopo())
    if a.smoke:
        folds = folds[:2]
    log(f"LOPO folds: {len(folds)}")
    if a.backend == "minilm":                    # fill the embedding cache once; workers then only read it
        from dynest.text import embed
        embed(list(ds.rows.text.fillna("")))
    res = Parallel(n_jobs=min(a.jobs, len(folds)), verbose=10)(          # prints "Done k out of 22"
        delayed(fold)(p, tr, va, te, ds.components, 2 if a.smoke else a.trials, a.backend, i)
        for i, (p, tr, va, te) in enumerate(folds))
    R = pd.DataFrame([r for rows in res for r in rows])
    R["project_key"] = R.project_id.map(ds.keys)
    save(R, "e2_unseen_projects.csv")
    print("\nUnseen projects (mean over held-out projects):")
    print(R.groupby("model")[["MAE", "MdAE", "LMAE", "SA", "MdMRE", "PRED25"]].mean().sort_values("MAE").round(3).to_string())
    for metric, low in (("MAE", True), ("LMAE", True), ("SA", False)):
        t = R.pivot_table(index="project_id", columns="model", values=metric)
        out = stats.compare(t, lower_is_better=low)
        pw = out["pairwise"]
        pw = pw[(pw.a == "DynEst-R (proposed)") | (pw.b == "DynEst-R (proposed)")]
        print(f"\n[{metric}] projects={out['n_units']} Friedman p={out.get('friedman_p', np.nan):.3g} CD={out['cd_nemenyi']:.2f}")
        print("mean ranks:", {k: round(v, 2) for k, v in sorted(out["mean_rank"].items(), key=lambda x: x[1])})
        print(pw[["a", "b", "median_diff", "p_holm", "A12", "magnitude"]].round(3).to_string(index=False))
        save(out["pairwise"], f"e2_pairwise_{metric}.csv")


if __name__ == "__main__":
    main()
