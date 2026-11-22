"""E0: construct validity of the effort proxy against logged work time (Jira Timespent).

For every TAWOS project with >= MIN issues carrying a positive Timespent, the proxy is recomputed from the
status history of ALL the project's issues (so the WIP sharing sees every concurrent ticket), and compared on the
issues with logged time with:
    effort_h          WIP-adjusted active business hours (the proxy used in the study)
    active_bh         active business hours without WIP sharing (sensitivity variant)
    In_Progress_Minutes, Total_Effort_Minutes, Resolution_Time_Minutes   TAWOS-provided time measures
    Story_Point       the team's own size estimate
Statistic: Spearman rho per project (unit-free), BCa CI of the median over projects; the pooled within-project
rank correlation; and the share of logged-time issues the proxy covers (> 0 active hours).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynest.realdata.effort import effort_proxy  # noqa: E402
from dynest.realdata.store import Store  # noqa: E402
from dynest.stats import bootstrap_ci  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MIN = 50
MEASURES = ["effort_h", "active_bh", "In_Progress_Minutes", "Total_Effort_Minutes", "Resolution_Time_Minutes",
            "Story_Point"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    st = Store()
    iss = st.issues
    keys = st.projects.set_index("ID").Project_Key
    logged = iss[(iss.Timespent > 0) & iss.Resolution_Date.notna()]
    cnt = logged.groupby("project_id").size()
    projects = cnt[cnt >= MIN].index.tolist()
    log(f"{len(projects)} projects with >= {MIN} resolved issues with logged time ({int(cnt[projects].sum())} issues)")
    ch = st.changelog(fields=["status"], issue_ids=set(iss[iss.project_id.isin(projects)].issue_id))
    rows, per = [], []
    for p in projects:
        pi = iss[iss.project_id == p]
        eff, _ = effort_proxy(ch[ch.Issue_ID.isin(set(pi.issue_id))], pi)
        d = pi.set_index("issue_id").join(eff[["effort_h", "active_bh"]])
        d = d[(d.Timespent > 0) & d.Resolution_Date.notna()].copy()
        d[["effort_h", "active_bh"]] = d[["effort_h", "active_bh"]].fillna(0.0)
        d["project"] = keys.get(p, p)
        rows.append(d[["project", "Timespent"] + MEASURES])
        r = dict(project=keys.get(p, p), n=len(d), proxy_coverage=float((d.effort_h > 0).mean()),
                 median_logged_h=float(d.Timespent.median() / 60), median_proxy_h=float(d.effort_h.median()))
        for m in MEASURES:
            x = d[m].astype(float)
            ok = x.notna() & (x > 0) if m == "Story_Point" else x.notna()
            r[f"rho_{m}"] = float(ss.spearmanr(x[ok], d.Timespent[ok]).statistic) if ok.sum() >= 10 else np.nan
            r[f"n_{m}"] = int(ok.sum())
        cov = d.effort_h > 0                                  # correlation where the proxy measured some work
        r["rho_effort_h_covered"] = float(ss.spearmanr(d.effort_h[cov], d.Timespent[cov]).statistic) \
            if cov.sum() >= 10 else np.nan
        per.append(r)
        log(f"  {r['project']:12s} n={r['n']:5d} cov={r['proxy_coverage']:.2f} "
            + " ".join(f"{m[:10]}={r[f'rho_{m}']:+.2f}" for m in MEASURES))
    P = pd.DataFrame(per)
    A = pd.concat(rows)
    P.to_csv(RESULTS / "e0_proxy_validity_projects.csv", index=False)

    print("\nMedian Spearman rho with logged time over projects (BCa 95% CI):")
    for m in MEASURES + ["effort_h_covered"]:
        v = P[f"rho_{m}"].dropna().values
        lo, hi = bootstrap_ci(v)
        print(f"  {m:26s} median={np.median(v):+.3f} [{lo:+.3f}, {hi:+.3f}]  positive in {(v > 0).sum()}/{len(v)} projects")
    print("\nPooled within-project rank correlation (ranks computed inside each project):")
    R = A.groupby("project").rank(pct=True)
    for m in MEASURES:
        ok = A[m].notna() & ((A[m] > 0) if m == "Story_Point" else True)
        print(f"  {m:26s} rho={ss.spearmanr(R[m][ok], R.Timespent[ok]).statistic:+.3f} (n={int(ok.sum())})")
    print(f"\nProxy covers {(A.effort_h > 0).mean():.1%} of logged-time issues; "
          f"median ratio proxy/logged (Timespent read as minutes) = "
          f"{np.median(A.effort_h[A.effort_h > 0] / (A.Timespent[A.effort_h > 0] / 60)):.2f}")
    print(P[["project", "n", "proxy_coverage", "median_logged_h", "median_proxy_h"]
            + [f"rho_{m}" for m in MEASURES]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
