"""E1b: project/sprint-level uncertainty.  Does the 95% band for the *sum* of a real sprint's requirements
contain the real total?  Groups = real (project, sprint) scopes in each origin's test window.

Compares the size-aware within-project resampling band (dynest.conformal.ProjectConformal) with the
naive band calibrated on past sprint totals (the previous implementation).
"""
from common import args, fitted_model, log, origin_splits, save
import numpy as np
import pandas as pd

from dynest.conformal import ProjectConformal, _q


def main():
    a = args()
    rows = []
    for k, (f, tr, va, te) in enumerate(origin_splits(a.smoke)):
        m = fitted_model(k, tr, va, a.backend, a.smoke)
        pv, pt = m.predict(va, intervals=False).hours.values, m.predict(te, intervals=False).hours.values
        new = ProjectConformal(alpha=m.cfg["alpha"], seed=k).fit(va.effort_h.values, pv, va.project_id.values)
        g = pd.DataFrame(dict(s=va.sprint_id.values, a=va.effort_h.values, p=pv)).dropna(subset=["s"])
        agg = g.groupby("s").agg(a=("a", "sum"), p=("p", "sum"), n=("a", "size"))
        agg = agg[agg.n >= 2]
        q_old = _q(np.abs(np.log(agg.a / agg.p)), m.cfg["alpha"])
        t = pd.DataFrame(dict(pr=te.project_id.values, s=te.sprint_id.values, a=te.effort_h.values, p=pt)).dropna(subset=["s"])
        for (pr, s), grp in t.groupby(["pr", "s"]):
            n = len(grp)
            if n < 2:
                continue
            A, P = grp.a.sum(), grp.p.sum()
            lo, hi = new.interval(P, n)
            rows.append(dict(origin=k, project=pr, sprint=s, n=n, actual=A, pred=P,
                             new_cover=lo <= A <= hi, new_width=(hi - lo) / P,
                             old_cover=P * np.exp(-q_old) <= A <= P * np.exp(q_old), old_width=2 * np.sinh(q_old)))
        log(f"origin {k}: {sum(r['origin'] == k for r in rows)} test sprints")
    R = pd.DataFrame(rows)
    save(R, "e1b_project_intervals.csv")
    R["size"] = pd.cut(R.n, [1, 5, 10, 20, 1000], labels=["2-5", "6-10", "11-20", ">20"])
    S = R.groupby("size", observed=True).agg(sprints=("n", "size"), new_coverage=("new_cover", "mean"),
                                             new_rel_width=("new_width", "median"), old_coverage=("old_cover", "mean"),
                                             old_rel_width=("old_width", "median"))
    S.loc["all"] = [len(R), R.new_cover.mean(), R.new_width.median(), R.old_cover.mean(), R.old_width.median()]
    print("\nSprint-level 95% intervals on held-out sprints (coverage target 0.95):")
    print(S.round(3).to_string())
    save(S.reset_index(), "e1b_project_intervals_summary.csv")


if __name__ == "__main__":
    main()
